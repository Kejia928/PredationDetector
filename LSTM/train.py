from __future__ import division
from __future__ import print_function

from SequenceDataset import SequenceDataset
from torch.utils.data import DataLoader
from model import train_model
from LSTM import LSTM
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

num_class = 2
batch_size = 4
epoch = 500
weight_decay = 0.0001
model_name = "LSTM"
print(model_name)

input_size = 101248
hidden_size = 512
num_layers = 2  
output_size = num_class  
print("Loading model ==========")
model = LSTM(input_size, hidden_size, num_layers, output_size)
print(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

params_to_update = model.parameters()
print("Params to learn:")
for name, param in model.named_parameters():
        if param.requires_grad:
            print("\t", name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.0001, momentum=0.9, weight_decay=weight_decay)

# Set up the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and Evaluate
model, hist = train_model(model, criterion, batch_size, optimizer_ft, num_epochs=epoch,
                             is_inception=(model_name == "inception"))