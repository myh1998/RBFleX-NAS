

# RBFleX-NAS: Training-free Network Architecture Search by Radial Basis Function Kernel with Hyperparameter Detection

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![](https://img.shields.io/github/downloads/tomomasayamasaki/RBF-kernel-based-NAS/total)
![](https://img.shields.io/github/repo-size/tomomasayamasaki/RBF-kernel-based-NAS)
![](https://img.shields.io/github/commit-activity/y/tomomasayamasaki/RBF-kernel-based-NAS)
![](https://img.shields.io/github/last-commit/tomomasayamasaki/RBF-kernel-based-NAS)
![](https://img.shields.io/github/languages/count/tomomasayamasaki/RBF-kernel-based-NAS)


## 🟨 What is RBF-kernel-based NAS
<table>
<tr>
<td><img src="https://github.com/tomomasayamasaki/RBF-kernel-based-NAS/blob/main/README/Picture 1.png"></td>
<td><img src="https://github.com/tomomasayamasaki/RBF-kernel-based-NAS/blob/main/README/Picture 2.png"></td>
<td><img src="https://github.com/tomomasayamasaki/RBF-kernel-based-NAS/blob/main/README/Picture 3.png"></td>
</tr>
</table>
Neural Architecture Search (NAS) is a technique to automatically discover optimal neural network architectures from candidate networks. In this work, we propose a novel NAS algorithm without training the networks to improve the speed and lower the computational cost. Compared to state-of-the-art NAS without training algorithms, our kernel-based approach possesses the capability to capture and analyze the distinguishing characteristics of networks. This enables our algorithm to effectively differentiate between suboptimal and superior networks, while also allowing for the evaluation of networks that incorporate a diverse range of cutting-edge activation layers. The algorithm scores the candidates by checking the similarity of the outputs of the activation layers and the input feature maps of the last layer. Specifically, the framework collects an output vector and a feature map vector from each candidate network with respect to one individual image. The similarity of the two vectors is evaluated among images in the same batch by using a Radial Basis Function (RBF) kernel to derive the score and predict the accuracy of the untrained network. Each hyperparameter of the RBF kernel is fine-tuned with a hyperparameter detection algorithm. We verified the efficacy of the proposed algorithm on NATS-Bench-SSS and Network Design Space (NDS) with CIFAR-10, CIFAR-100, and ImageNet datasets. Our NAS algorithm outperforms state-of-the-art NAS algorithms such as NASWOT by 1.12-3.79x in terms of the Kendall correlation between the score and the network accuracy. Our NAS achieved 89.76% on CIFAR-10, 69.84% on CIFAR-100, and 43.16% on imageNet, respectively on NAS-Bench-201. Furthermore, our method works with only ReLU but also various types of activation layers.

## 🟨 Environmental Requirements
- Python version3
- pytorch
- torchvision
- numpy
- scipy
- numexpr


For NAS benchmark
- nats_bench
[note] you may run "pip install nats_bench"

## 🟨 Download NAS Benchmarks
Our program works for NAS-Bench-201(NATS-Bench-TSS), NATS-Bench-SSS, and Network Design Space. If you want to apply our NAS algorithm, edit our program to meet other NAS benchmarks.
### NATS-Bench
To download the latest version, please visit the github page of [NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size](https://github.com/D-X-Y/NATS-Bench). You can download the benchmark tar file of `NATS-tss-v1_0-3ffb9-simple.tar` and `NATS-sss-v1_0-50262-simple.tar` from [Google drive](https://drive.google.com/drive/folders/1zjB6wMANiKwB2A1yil2hQ8H_qyeSe2yt). After downloading these benchamark tar files, please store "NATS-tss-v1_0-3ffb9-simple.tar" on `./designspace/NAS-Bench-201/` and "NATS-sss-v1_0-50262-simple.tar" on `./designspace/NATS-Bench-SSS/`, respectively. And then, you may uncompress them by running `tar xvf NATS-tss-v1_0-3ffb9-simple.tar` and `tar xvf NATS-sss-v1_0-50262-simple.tar`.

### Network Design Space (NDS)
To download NDS benchmark json files, please visit the github page of [On Network Design Spaces for Visual Recognition](https://github.com/facebookresearch/nds). You can download the benchmark json files from [this link](https://dl.fbaipublicfiles.com/nds/data.zip). After downloading these json files, please store all of json files on `./designsoace/NDS/`

#### Noted
Please store each benchmark on each folder such as `./designspace/NAS-Bench-201/`, `./designspace/NATS-Bench-SSS/`, and `./designsoace/NDS/`.

## 🟨 Download image dataset
Our program works with three major datasets: CIFAR-10, CIFAR-100, and ImageNet.
### CIFAR-10 and CIFAR-100
These datasets are automatically download on `./dataset/CIFAR10` or `./dataset/CIFAR100` respectively by pytorch after running our program. If you already have CIFAR-10 and CIFAR-100 dataset on your environment, you may set your dataset path on root on each main program as follow.
```python
trainset = torchvision.datasets.CIFAR10(root='YOUR DATASET PATH', train=True, download=True, transform=transform_train)

cifar100_training = torchvision.datasets.CIFAR100(root='YOUR DATASET PATH', train=True, download=True, transform=transform_train)
```

### ImageNet
You may download training set of ImageNet from [here](https://www.image-net.org). Before downloading, you should create a account on the website. After downloading it, please store the dataset on `./dataset/ImageNet`. If you already have ImageNet dataset on your environment, you may set your dataset path on `config.py` as follow.
```python
train_root = 'YOUR DATASET PATH/ILSVRC2012_img_train'
```

## 🟨 How to Run
#### 1. Download NAS Benchmarks
put them on `./designspace/NAS-Bench-201/`, `./designspace/NATS-Bench-SSS/`, and `./designsoace/NDS/`, respectively.


#### 2. Set parameters on `config.py`
where `N` is mini-batch size of images, `Num_Network` is the number of networks from a design space in one trial, `max_trials` is the number of trials, `M` is the number of networks to detect hyperparameter gamma for RBF kernel, and `dataset` is the dataset image to evaluate networks. Noted that set a NDS space on `NDS_SPACE` if you want to run the NDS benchmark. For example, put 'DARTS' if you explore NDS(DARTS) benchmark. Design space with `in` such as `Amoeba_in, DARTS_in, ENAS_in, PNAS_in, NASNet_in` indicates a desige space for ImageNet.
```python
N = 16
Num_Networks = 1000
max_trials = 1
M = 10
dataset = 'cifar10' # select dataset from ['cifar10', 'cifar100', 'ImageNet16-120']
NDS_SPACE = 'DARTS' # select one design space from ['Amoeba', 'DARTS', 'ENAS', 'PNAS', 'ResNet', 'NASNet','Amoeba_in','DARTS_in','ENAS_in', 'PNAS_in', 'NASNet_in']
```

#### 3. Run our program
If you want to run it with NAS-Bench-201,
```python
python main_NAS201.py
```
If you want to run it with NATS-Bench-SSS,
```python
python main_SSS.py
```
If you want to run it with NDS,
```python
python main_NDS.py
```
Noted that define `NDS_SPACE` on `config.py` before running our program with NDS.

