# NAFBee: Neural Network Activation Function Benchmark
This is a benchmark for networks with a variety of activation functions. NAFBee provides network information and accuracy. User can obtain the accuracy without training. NAFBee is used for RBFleX-NAS.

# Requirement
- python 3.x
- PyTorch

# How to use
## 0. Import packages
```python
import json
from models import *
```

## 1. Load NAFBee.json
```python
file_path = "NAFBee.json"
with open(file_path, "r") as json_file:
    nafbee = json.load(json_file)
```

## 2. Get the network information
```python
info = nafbee["1"] #you can input numbers from 1 to 11.
print(info)

#{'network': 'VGG19', 'activation': 'ReLU', 'accuracy': 91.06}
```
## 3. All information of the No.1 network
```python
info_network = info["network"]
info_activation = info["activation"]
info_accuracy = info["accuracy"]
```

## 4. Define the model on PyTorch
```python
if "VGG" in info_network:
    model = VGG(info_network, info_activation)
```

# Demo
You can see a program to get the model. You can add any program using the model on Pytorch such as training or scoring.
```python
python NAFBee.py
```
