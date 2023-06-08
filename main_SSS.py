import time
from models import get_cell_based_tiny_net
import numpy as np
import torchvision
import torchvision.transforms as transforms
import numexpr as ne
import torch
from scipy import stats
import random
from nats_bench import create
import time
import config
from torchvision.datasets import ImageFolder

intro = ' ______  ______  _______          __                             __ \n'
intro +='|   __ \|   __ \|    ___| ______ |  |--..---.-..-----..-----..--|  |\n'
intro +='|      <|   __ <|    ___||______||  _  ||  _  ||__ --||  -__||  _  |\n'
intro +='|___|__||______/|___|            |_____||___._||_____||_____||_____|\n'
intro +='                     _______   _______   _______\n'
intro +='                    |    |  | |   _   | |     __|\n'
intro +='                    |       | |       | |__     |\n'
intro +='                    |__|____| |___|___| |_______|\n'

print(intro)

#######################################
# Hyperparameter
# - batch_size_NE: batch size for this NAS
# - Num_Networks: the number of networks in one trial
# - maxtrials: the number of trials
# - N_GAMMA: the number of network sampled randomly for self-detecting hyperparameter
# - dataset_path: dataset path
# - api_loc: NAS benchmark file path
#######################################
print('==> Preparing parameters..')
batch_size_NE = config.N
Num_Networks = config.Num_Networks
maxtrials = config.max_trials
N_GAMMA = config.M
dataset = config.dataset
# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print('Mini-batch size: {}'.format(batch_size_NE))
print('Number of candidate networks as the design space:{}'.format(Num_Networks))
print('Number of trials:{}'.format(maxtrials))
print('Number of candidate networks to detect Gamma for RBF:{}'.format(N_GAMMA))
print('Compuataion device:{}'.format(device))
print('Dataset:{}'.format(dataset))
print()

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
# Design Space
# - Load design space
# - networks are defined by pytorch
#######################################
s = time.time()
searchspace = create('./designspace/NATS-Bench-SSS/NATS-sss-v1_0-50262-simple', 'sss', fast_mode=True, verbose=False)
e = time.time()
print('DONE')
print('time:{}'.format(e-s))

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
          root='./dataset/CIFAR10', train=True, download=True, transform=transform_train)
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
  cifar100_training = torchvision.datasets.CIFAR100(root='./dataset/CIFAR100', train=True, download=True, transform=transform_train)
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


# Model
print('==> Building model..')
for r in range(maxtrials):
  batch_space = random.sample(range(len(searchspace)), Num_Networks*3)
  
  done_network = 1

  score_array = np.zeros(Num_Networks)
  acc_array = np.zeros(Num_Networks)
  
  ########################
  # Compute Distance and Kernel Matrix
  ########################
  def counting_forward_hook(module, inp, out):
    with torch.no_grad():
      arr = out.view(batch_size_NE, -1)
      network.K = torch.cat((network.K, arr),1)
      
  def counting_forward_hook_FC(module, inp, out):
      with torch.no_grad():
        if isinstance(inp, tuple):
            inp = inp[0]
        network.Q = inp
  
  #######################################
  # Self-detecting Hyperparameter
  #######################################
  GAMMA_K_list = []
  GAMMA_Q_list = []
  for id in range(N_GAMMA):
    uid = batch_space[id]
    config = searchspace.get_net_config(uid, dataset)
    network = get_cell_based_tiny_net(config)
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
        
    with torch.no_grad():
      network.K = torch.empty(0, device=device)
      network.Q = torch.empty(0, device=device)
      network(x[0:batch_size_NE,:,:,:].to(device))
      
      Output_matrix = network.K
      Last_matrix = network.Q
    
    with torch.no_grad():
      Output_matrix = Output_matrix.cpu().numpy()
      Last_matrix = Last_matrix.cpu().numpy()
      
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
  print('==> Detected Hyperparameter Gamma ..')
  print('gamma_k:',GAMMA_K)
  print('gamma_q:',GAMMA_Q)
  
  #######################################
  # Evaluate Networks in design space
  #######################################
  best_uid = 0
  best_accuracy = 0
  best_score = -1e10
  count = 1
  print('==> Evaluate networks in design space ..')
  for uid in batch_space:
    
    print('[{}/{}]Evalaute network id: {}'.format(count,Num_Networks,uid))
    
    config = searchspace.get_net_config(uid, dataset)
    network = get_cell_based_tiny_net(config)
    
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
    

    with torch.no_grad():
      network.K = torch.empty(0, device=device)
      network.Q = torch.empty(0, device=device)
      network(x[0:batch_size_NE,:,:,:].to(device))
      
      Output_matrix = network.K
      Last_matrix = network.Q
    
    # Normalization
    with torch.no_grad():
      Output_matrix = normalize(Output_matrix.cpu().numpy(), axis=0)
      Last_matrix = normalize(Last_matrix.cpu().numpy(), axis=0)


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
    _, K = np.linalg.slogdet(K_Matrix)
    _, Q = np.linalg.slogdet(Q_Matrix)
    score = batch_size_NE*(K+Q)
    
    # Get accuracy of networks in design spaces
    accuracy, latency, time_cost, current_total_time_cost = searchspace.simulate_train_eval(uid, dataset=dataset, hp='90')
     
    
    if np.isinf(score):
      score = -1e10
      
    print('\t Score: {}'.format(score))
    print('\t Accuracy: {}'.format(accuracy))
    score_array[done_network-1] = score
    acc_array[done_network-1] = accuracy
    
    if score > best_score:
      best_score = score
      best_accuracy = accuracy
      best_uid = uid
    
    count += 1
    done_network += 1
    
    if done_network > Num_Networks:
      break
    
  CC = np.corrcoef(acc_array, score_array)
  tau, p = stats.kendalltau(acc_array,score_array)
  print("======================================")
  print('Trial: ', r)
  print('\tPearson Correlation: ',CC[0,1])
  print('\tKendall Correlation: ', tau)
  print('In this design space')
  print('\tBest network id: {}'.format(best_uid))
  print('\tThe network accuracy: {}'.format(best_accuracy))
  print('\tThe network score: {}'.format(best_score))
  print("======================================")
  print()






