import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from SequenceDataset import SequenceDataset
from LSTM import LSTM

input_size = 101248
hidden_size = 128
num_layers = 2  
output_size = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LSTM(input_size, hidden_size, num_layers, output_size)
state_dict = torch.load('runs/exp4/best.pt', map_location='cpu')
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
test_dataset = SequenceDataset(path='../video2img', json_path='../feeding_dataset/videoAnnotation_test.jsonl', drop=True, random=False)
test_dataset.getAnnotation()
test_dataset.initDataset()
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

running_corrects = 0
feeding_corrects = 0
no_corrects = 0

for input, label in tqdm.tqdm(test_loader, desc="Run on dataset: "):
    input = input.unsqueeze(2)
    input = input.view(input.shape[0], input.shape[1], -1)
    input = input.to(device)
    label = label.to(device)
    outputs = model(input)
    _, preds = torch.max(outputs, 1)
    running_corrects += torch.sum(preds == label.data)
    feeding_corrects += torch.sum((preds == label.data) * (label.data == 1))
    no_corrects += torch.sum((preds == label.data) * (label.data == 0))

acc = running_corrects.double() / len(test_loader.dataset)
feeding_acc = feeding_corrects.double() / len(test_dataset.pos)
no_acc = no_corrects.double() / len(test_dataset.pos)
print("Average Acc:", acc)
print("Feeding Acc:", feeding_acc)
print("No Acc:", no_acc)
