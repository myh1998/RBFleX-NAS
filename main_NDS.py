import numpy as np
import torchvision
import torchvision.transforms as transforms
import numexpr as ne
import torch
from scipy import stats
import random
from NDS import NDS
import config
from torchvision.datasets import ImageFolder

#######################################
# Hyperparameter
# - batch_size_NE: batch size for this NAS
# - Num_Networks: the number of networks in one trial
# - maxtrials: the number of trials
# - N_GAMMA: the number of network sampled randomly for self-detecting hyperparameter
#######################################
print('==> Preparing hyperparameters..')
batch_size_NE = config.batch_size_NE
Num_Networks = config.Num_Networks
maxtrials = config.maxtrials
N_GAMMA = config.N_GAMMA
NDS_SPACE = config.NDS_SPACE
dataset_path = config.dataset_path
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#######################################
# Reproducibility
#######################################
print('==> Reproducibility..')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)
torch.manual_seed(1)

#######################################
# Normalization 
# - Column-wise normalization
####################################### 
def normalize(x, axis=None):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    x_max[x_max == x_min] = 1
    x_min[x_max == x_min] = 0
    return (x - x_min) / (x_max - x_min)

#######################################
# Dataset
# - please refer pytorch reference to use other datesets
#######################################
print('==> Preparing data..')
if config.dataset == 'cifar10':
  transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  trainset = torchvision.datasets.CIFAR10(
          root=dataset_path, train=True, download=True, transform=transform_train)
  trainloader = torch.utils.data.DataLoader(
          trainset, batch_size=batch_size_NE, shuffle=True, num_workers=2, pin_memory=True)

elif config.dataset == 'cifar100':
  transform_train = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
  ])
  cifar100_training = torchvision.datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transform_train)
  trainloader = torch.utils.data.DataLoader(cifar100_training, shuffle=True, num_workers=4, batch_size=batch_size_NE, pin_memory=True)
  
elif config.dataset == 'ImageNet16-120':
  train_root = config.train_root
  norma = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

  train_transform = transforms.Compose([
      transforms.RandomResizedCrop(224), 
      transforms.ToTensor(),
      norma,
  ])
  trainset = ImageFolder(root=train_root,
                          transform=train_transform)
  trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=batch_size_NE, shuffle=True, num_workers=2, pin_memory=True)

# x: images used for this NAS
data_iterator = iter(trainloader)
x, target = next(data_iterator)



print('Loading...NDS '+NDS_SPACE)
searchspace = NDS(NDS_SPACE)
print('Num of networks: ', len(searchspace))
print('DONE')

# Model
print('==> Building model..')
if len(searchspace) < 1000:
  maxtrials = 1
for r in range(maxtrials):
  if len(searchspace) > 10000:
    batch_space = random.sample(range(len(searchspace)), Num_Networks)
    score_array = np.zeros(Num_Networks)
    acc_array = np.zeros(Num_Networks)
  else:
    batch_space = range(0,len(searchspace))
    score_array = np.zeros(len(searchspace))
    acc_array = np.zeros(len(searchspace))
  
  done_network = 1
  
  
  ########################
  # Compute Distance and Kernel Matrix
  ########################
  def counting_forward_hook(module, inp, out):
      arr = out.view(-1)
      network.K = torch.concatenate([network.K, arr])
      
  def counting_forward_hook_FC(module, inp, out):
      arr = inp[0].view(-1)
      network.Q = torch.concatenate([network.Q, arr])
  
  #######################################
  # Self-detecting Hyperparameter
  #######################################
  GAMMA_K_list = []
  GAMMA_Q_list = []
  for id in range(N_GAMMA):
    uid = batch_space[id]
    network = searchspace.get_network(uid)
    network = network.to(device)
    
    net_counter = list(network.named_modules())
    net_counter = len(net_counter)
    NC = 0
    for name, module in network.named_modules():
      NC += 1
      if 'ReLU' in str(type(module)):
        module.register_forward_hook(counting_forward_hook)
      if NC == net_counter:
        module.register_forward_hook(counting_forward_hook_FC)
        
    # Check LA
    x2 = torch.clone(x[0:1,:,:,:])
    x2 = x2.to(device)
    network.K = torch.tensor([], device=device)
    network.Q = torch.tensor([], device=device)
    network(x2)
    LA = len(network.K)
    LAQ = len(network.Q)
    
    Output_matrix = np.zeros([batch_size_NE, LA])
    Last_matrix = np.zeros([batch_size_NE, LAQ])
    for i in range(batch_size_NE):
      x2 = torch.clone(x[i:i+1,:,:,:])
      x2 = x2.to(device)
      network.K = torch.tensor([], device=device)
      network.Q = torch.tensor([], device=device)
      network(x2)
      Output_matrix[i,:] = network.K.cpu().detach().clone().numpy()
      Last_matrix[i,:] = network.Q.cpu().detach().clone().numpy()
      
    for i in range(batch_size_NE-1):
      for j in range(i+1,batch_size_NE):
          z1 = Output_matrix[i,:]
          z2 = Output_matrix[j,:]
          m1 = np.mean(z1)
          m2 = np.mean(z2)
          M = (m1-m2)**2
          z1 = z1-m1
          z2 = z2-m2
          s1 = np.mean(z1**2)
          s2 = np.mean(z2**2)
          if s1+s2 != 0:
            candi_gamma_K = M/((s1+s2)*2)
            GAMMA_K_list.append(candi_gamma_K)
            
    for i in range(batch_size_NE-1):
      for j in range(i+1,batch_size_NE):
          z1 = Last_matrix[i,:]
          z2 = Last_matrix[j,:]
          m1 = np.mean(z1)
          m2 = np.mean(z2)
          M = (m1-m2)**2
          z1 = z1-m1
          z2 = z2-m2
          s1 = np.mean(z1**2)
          s2 = np.mean(z2**2)
          if s1+s2 != 0:
            candi_gamma_Q = M/((s1+s2)*2)
            GAMMA_Q_list.append(candi_gamma_Q)
          
  GAMMA_K = np.min(np.array(GAMMA_K_list))
  GAMMA_Q = np.min(np.array(GAMMA_Q_list))
  print('gamma_k:',GAMMA_K)
  print('gamma_q:',GAMMA_Q)
  
  #######################################
  # Evaluate Networks in design space
  #######################################
  for uid in batch_space:
    network = searchspace.get_network(uid)
    
    network = network.to(device)

        
    net_counter = list(network.named_modules())
    net_counter = len(net_counter)
    NC = 0
    for name, module in network.named_modules():
      NC += 1
      if 'ReLU' in str(type(module)):
        module.register_forward_hook(counting_forward_hook)
      if NC == net_counter:
        module.register_forward_hook(counting_forward_hook_FC)
    
    #ls
    #print('NFC:',NFC)
    # Check LA
    x2 = torch.clone(x[0:1,:,:,:])
    x2 = x2.to(device)
    network.K = torch.tensor([], device=device)
    network.Q = torch.tensor([], device=device)
    network(x2)
    LA = len(network.K)
    LAQ = len(network.Q)
    
    Output_matrix = np.zeros([batch_size_NE, LA])
    Last_matrix = np.zeros([batch_size_NE, LAQ])
    for i in range(batch_size_NE):
      x2 = torch.clone(x[i:i+1,:,:,:])
      x2 = x2.to(device)
      network.K = torch.tensor([], device=device)
      network.Q = torch.tensor([], device=device)
      network(x2)
      Output_matrix[i,:] = network.K.cpu().detach().clone().numpy()
      Last_matrix[i,:] = network.Q.cpu().detach().clone().numpy()
      
    # Normalization
    Output_matrix = normalize(Output_matrix, axis=0)
    Last_matrix = normalize(Last_matrix, axis=0)
    
    # RBF kernel
    X_norm = np.sum(Output_matrix ** 2, axis = -1)
    K_Matrix = ne.evaluate('exp(-g * (A + B - 2 * C))', {
            'A' : X_norm[:,None],
            'B' : X_norm[None,:],
            'C' : np.dot(Output_matrix, Output_matrix.T),
            'g' : GAMMA_K
    })
    Y_norm = np.sum(Last_matrix ** 2, axis = -1)
    Q_Matrix = ne.evaluate('exp(-g * (A + B - 2 * C))', {
            'A' : Y_norm[:,None],
            'B' : Y_norm[None,:],
            'C' : np.dot(Last_matrix, Last_matrix.T),
            'g' : GAMMA_Q
    })
    
    # Compute score
    _, score = np.linalg.slogdet(np.kron(K_Matrix, Q_Matrix)) # idea18
    
    # Get accuracy of networks in design spaces
    accuracy = searchspace.get_final_accuracy(uid)
    
    if np.isinf(score_id18):
      score_id18 = -1e10
      
    score_array[done_network-1] = score
    acc_array[done_network-1] = accuracy
    
    done_network += 1
    
  CC = np.corrcoef(acc_array, score_array)
  tau, p = stats.kendalltau(acc_array,score_array)
  print("======================================")
  print('Trial: ', r)
  print()
  print('Pearson Correlation: ',CC[0,1])
  print('Kendall Correlation: ', tau)
  print("======================================")
  print()






