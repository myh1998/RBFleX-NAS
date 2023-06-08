<a href="https://istd.sutd.edu.sg/people/phd-students/tomomasa-yamasaki">
    <img src="https://github.com/tomomasayamasaki/LAXOR/blob/main/README/logo.png" alt="Tomo logo" title="Tomo" align="right" height="110" />
</a>

# Training-free Network Architecture Search by Radial Basis Function Kernel with Hyperparameter Detection

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![](https://img.shields.io/github/downloads/tomomasayamasaki/RBF-kernel-based-NAS/total)
![](https://img.shields.io/github/repo-size/tomomasayamasaki/RBF-kernel-based-NAS)
![](https://img.shields.io/github/commit-activity/y/tomomasayamasaki/RBF-kernel-based-NAS)
![](https://img.shields.io/github/last-commit/tomomasayamasaki/RBF-kernel-based-NAS)
![](https://img.shields.io/github/languages/count/tomomasayamasaki/RBF-kernel-based-NAS)

## 🟨 Contents


## 🟨 What is RBF-kernel-based NAS
Neural Architecture Search (NAS) is a technique to automatically discover optimal neural network architectures from candidate networks. In this work, we propose a novel NAS algorithm without training the networks to improve the speed and lower the computational cost. Compared to state-of-the-art NAS without training algorithms, our kernel-based approach possesses the capability to capture and analyze the distinguishing characteristics of networks. This enables our algorithm to effectively differentiate between suboptimal and superior networks, while also allowing for the evaluation of networks that incorporate a diverse range of cutting-edge activation layers. The algorithm scores the candidates by checking the similarity of the outputs of the activation layers and the input feature maps of the last layer. Specifically, the framework collects an output vector and a feature map vector from each candidate network with respect to one individual image. The similarity of the two vectors is evaluated among images in the same batch by using a Radial Basis Function (RBF) kernel to derive the score and predict the accuracy of the untrained network. Each hyperparameter of the RBF kernel is fine-tuned with a hyperparameter detection algorithm. We verified the efficacy of the proposed algorithm on NATS-Bench-SSS and Network Design Space (NDS) with CIFAR-10, CIFAR-100, and ImageNet datasets. Our NAS algorithm outperforms state-of-the-art NAS algorithms such as NASWOT by 1.12-3.79x in terms of the Kendall correlation between the score and the network accuracy. Our NAS achieved 89.76\% on CIFAR-10, 69.84\% on CIFAR-100, and 43.16\% on imageNet, respectively on NAS-Bench-201. Furthermore, our method works with only ReLU but also various types of activation layers.

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

## 🟨 Download NAS Benchmark
Our program works for NAS-Bench-201(NATS-Bench-TSS), NATS-Bench-SSS, and Network Design Space. If you want to apply our NAS algorithm, edit our program to meet other NAS benchmarks.
### NATS-Bench

### Network Design Space (NDS)

## 🟨 How to Run


