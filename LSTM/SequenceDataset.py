import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms import transforms
import json
import random
import numpy as np

def split_list(lst):
    result = []
    temp = []
    for i in range(len(lst)):
        if i == 0:
            temp.append(lst[i])
            if len(temp) == 10:
                result.append(temp)
                temp = []
        elif i != 0 and lst[i] == lst[i-1] + 1:
            temp.append(lst[i])
            if len(temp) == 11:
                result.append(temp)
                temp = []
        elif len(temp) != 0:
            last = temp[-1]
            for j in range(0, 11):
                temp.append(last)
                if len(temp) == 11:
                    result.append(temp)
                    temp = []
                    break
            temp.append(lst[i])
    return result

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


class SequenceDataset(Dataset):
    def __init__(self, path, json_path, drop=False, random=False):
        self.data = []
        self.labels = []
        self.path = path
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])
        self.videoName = []
        self.annotations = []
        self.json_path = json_path
        self.drop = drop
        self.pos = []
        self.neg = []
        self.random = random

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
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
        return torch.cat(image, dim=0), label

    def getInputDim(self):
        image, label = self.__getitem__(0)
        return image.shape

    def getDatasetPath(self):
        return self.path

    def getAnnotation(self):
        self.videoName = ''
        self.annotations = []
        with open(self.json_path, 'r') as f:
            for line in f:
                disc = {}
                line = json.loads(line)
                for key, value in line.items():
                    if key == 'videoName':
                        disc[key] = value
                    elif key == 'timeSegmentAnnotations':
                        annotations = [d['startTime'] for d in value]
                        fps = 30
                        frame_numbers = []
                        for ts in annotations:
                            seconds = float(ts.replace('s', ''))
                            frame_number = int(seconds * fps)
                            frame_numbers.append(frame_number)
                        disc[key] = frame_numbers
                self.annotations.append(disc)
            f.close()
        return

    def initDataset(self):
        for a in self.annotations:
            videoName = a['videoName']
            # print(videoName)
            feeding = a['timeSegmentAnnotations']
            frame_path = os.path.join(os.path.join(self.path, videoName[:-4]), 'ori_img')
            hori = os.path.join(os.path.join(self.path, videoName[:-4]), 'horizontal_flow')
            vert = os.path.join(os.path.join(self.path, videoName[:-4]), 'vertical_flow')
            yolo = os.path.join(os.path.join(self.path, videoName[:-4]), 'yolo_img/labels')
            resnet = os.path.join(os.path.join(self.path, videoName[:-4]), 'resnet_output/labels')
            frame = os.listdir(frame_path)
            frame.sort()
            frameNumber = list(range(0, len(frame)))
            # Positive sample
            already = set()
            for i in feeding:
                optical_flow = []
                for j in range(i-5, i+6):
                    single_frame = []
                    if j < 0: j = 0
                    if j > len(frame)-1: j = len(frame)-1
                    filename = f'frame_{j}.jpg'
                    formatted_num =  "%04d" % j
                    single_frame.append(os.path.join(hori, filename))
                    single_frame.append(os.path.join(vert, filename))
                    single_frame.append(os.path.join(yolo, formatted_num + '.txt'))
                    single_frame.append(os.path.join(resnet, formatted_num + '.txt'))
                    optical_flow.append(single_frame)
                    already.add(j)
                self.pos.append(optical_flow)
            frameNumber = [x for x in frameNumber if x not in already]
            # print(len(self.data))
            # print(feeding)
            # print(self.data)
            # print(already)
            # print(frameNumber)
            # exit()
            # Negative sample
            negative_frame = split_list(frameNumber)
            # print(negative_frame)
            # exit()
            for i in negative_frame:
                optical_flow = []
                for j in i:
                    single_frame = []
                    filename = f'frame_{j}.jpg'
                    formatted_num =  "%04d" % j
                    single_frame.append(os.path.join(hori, filename))
                    single_frame.append(os.path.join(vert, filename))
                    single_frame.append(os.path.join(yolo, formatted_num + '.txt'))
                    single_frame.append(os.path.join(resnet, formatted_num + '.txt'))
                    optical_flow.append(single_frame)
                self.neg.append(optical_flow)
                # print(optical_flow)
                # print(self.labels)
                # exit()
        for i in self.pos:
            self.data.append(i)
            self.labels.append(1)
        for i in self.neg:
            self.data.append(i)
            self.labels.append(0)

        if self.drop:
            if self.random:
                random.seed()
            else:
                random.seed(42)
            indices_to_delete = [i for i, x in enumerate(self.labels) if x == 0]
            indices_to_delete = random.sample(indices_to_delete, k=len(indices_to_delete)-self.labels.count(1))
            indices_to_delete.sort(reverse=True)
            for idx in indices_to_delete:
                del self.labels[idx]
                del self.data[idx]
        return

# train_dataset = SequenceDataset('../video2img', '../feeding_dataset/videoAnnotation_train.jsonl')
# test_dataset = SequenceDataset('../video2img', '../feeding_dataset/videoAnnotation_test.jsonl')
# train_dataset.getAnnotation()
# train_dataset.initDataset()
# print(len(train_dataset.data))
# print(train_dataset.labels.count(1))
# test_dataset.getAnnotation()
# test_dataset.initDataset()
# print(len(test_dataset.data))
# print(test_dataset.labels.count(1))
# print(train_dataset.data[6])
# image, _ = train_dataset.__getitem__(4)
# print(image.shape)