#######################################
# Hyperparameter
# - batch_size_NE: batch size for this NAS
# - Num_Networks: the number of networks in one trial
# - maxtrials: the number of trials
# - N_GAMMA: the number of network sampled randomly for self-detecting hyperparameter
# - dataset_path: dataset path
#######################################
N = 16
Num_Networks = 1000
max_trials = 1
M = 10
dataset = 'cifar10' # select dataset from ['cifar10', 'cifar100', 'ImageNet16-120']

# if you use imageNet, please set train_root
train_root = './dataset/ImageNet/ILSVRC2012_img_train'

# Network Design Spaces (NDS)
NDS_SPACE = 'DARTS' # select one design space from ['Amoeba', 'DARTS', 'ENAS', 'PNAS', 'ResNet', 'NASNet','Amoeba_in','DARTS_in','ENAS_in', 'PNAS_in', 'NASNet_in']