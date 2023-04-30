import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms import transforms
import json
import random
import numpy as np

def readYOLOTxt(path):
        if os.path.exists(path):
            with open(path) as f:
                data = f.read().split()
            data_array = np.array([float(x) for x in data])
            return data_array
        else:
            return [0]
    
def readResNetTxt(path):
    with open(path) as f:
        data = f.read().split()
    data_array = np.array([int(x == 'True') for x in data])
    return data_array

def expend(data, length=428):
    data_tensor = torch.tensor(data)
    num_repeats = (length // len(data)) + 1
    repeated_tensor = torch.cat([data_tensor] * num_repeats)
    expanded_tensor = torch.unsqueeze(repeated_tensor[:length], 0)
    expanded_tensor = torch.unsqueeze(expanded_tensor, 0)
    return expanded_tensor

class VideoLoader(Dataset):
    def __init__(self, path, videoName):
        self.data = []
        self.labels = []
        self.frame_group = []
        self.path = path
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])
        self.videoName = videoName

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.data[index]
        image = []
        for image_path in path:
            horizontal = Image.open(image_path[0])
            vertical = Image.open(image_path[1])
            yolo_label = image_path[2]
            resnet_label = image_path[3]
            if self.transforms:
                horizontal = self.transforms(horizontal)
                vertical = self.transforms(vertical)
                yolo_data = expend(data=readYOLOTxt(yolo_label), length=horizontal.shape[2]*2)
                resnet_data = expend(data=readResNetTxt(resnet_label), length=horizontal.shape[2]*2)
                image.append(torch.cat((torch.cat((horizontal, vertical), dim=2), yolo_data, resnet_data),dim=1))
        return torch.cat(image, dim=0)

    def getDatasetPath(self):
        return self.path
    
    def getFrameGroup(self):
        return self.frame_group

    def getVideoFramePath(self):
        return os.path.join(self.path, self.videoName[:-4])

    def initDataset(self):
        frame_path = os.path.join(os.path.join(self.path, self.videoName[:-4]), 'ori_img')
        hori = os.path.join(os.path.join(self.path, self.videoName[:-4]), 'horizontal_flow')
        vert = os.path.join(os.path.join(self.path, self.videoName[:-4]), 'vertical_flow')
        yolo = os.path.join(os.path.join(self.path, self.videoName[:-4]), 'yolo_img/labels')
        resnet = os.path.join(os.path.join(self.path, self.videoName[:-4]), 'resnet_output/labels')
        frame = os.listdir(frame_path)
        frame.sort()
        optical_flow = []
        frame_group = []
        for i in range(0, len(frame)):
            single_window = []
            filename = f'frame_{i}.jpg'
            formatted_num =  "%04d" % i
            single_window.append(os.path.join(hori, filename))
            single_window.append(os.path.join(vert, filename))
            single_window.append(os.path.join(yolo, formatted_num + '.txt'))
            single_window.append(os.path.join(resnet, formatted_num + '.txt'))
            optical_flow.append(single_window)
            frame_group.append(i)
            if i == len(frame)-1:
                if len(optical_flow) != 11:
                    last_item = optical_flow[-1]
                    optical_flow = optical_flow + [last_item] * (11 - len(optical_flow))

            if len(optical_flow) == 11 :
                self.data.append(optical_flow)
                self.frame_group.append(frame_group)
                self.labels.append(1)
                optical_flow = []
                frame_group = []
        return

