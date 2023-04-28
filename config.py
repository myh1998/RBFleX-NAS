#######################################
# Hyperparameter
# - batch_size_NE: batch size for this NAS
# - Num_Networks: the number of networks in one trial
# - maxtrials: the number of trials
# - N_GAMMA: the number of network sampled randomly for self-detecting hyperparameter
# - dataset_path: dataset path
#######################################
batch_size_NE = 16
Num_Networks = 1000
maxtrials = 10
N_GAMMA = 10
dataset_path = './'
dataset = 'cifar10' # select dataset from ['cifar10', 'cifar100', 'ImageNet16-120']
# if you use imageNet, please set train_root
train_root = './'

# NAS-Bench-201 and NATS-Bench-SSS
api_loc = './NATS-sss-v1_0-50262-simple' #NATS-Bench-SSS
#api_loc = './NATS-tss-v1_0-3ffb9-simple' #NAS-Bench-201

# Network Design Spaces (NDS)
NDS_SPACE = 'DARTS' # select one design space from ['Amoeba', 'DARTS', 'ENAS', 'PNAS', 'ResNet', 'NASNet']