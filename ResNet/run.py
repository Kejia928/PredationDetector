import os

import cv2
import numpy as np
import torch
import tqdm
from torchvision import models
from model import initialize_model
from torchvision.transforms import transforms
from PIL import Image
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset

model = initialize_model(model_name="resnet18", num_classes=2, feature_extract=False, use_pretrained=False)
state_dict = torch.load('runs/exp15/best.pt', map_location='cpu')

subfolders = [f.path for f in os.scandir("../video2img/") if f.is_dir()]
subfolders.sort()
subfolders = subfolders[:10]

for subfolder in subfolders:
    dir_path = subfolder
    path = dir_path + "/ori_img"
    save_path = dir_path + "/resnet_output"
    label_path = save_path + '/labels'
    if os.path.exists(save_path):
        continue
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)
    images = CustomDataset(path)
    images.getImage()
    loader = DataLoader(images, batch_size=1, shuffle=False)

    model.load_state_dict(state_dict)
    model.eval()

    all_file = os.listdir(path)
    all_file.sort()
    for file in tqdm.tqdm(all_file):
        image_path = path + '/' + file
        txt_filename = file[:-4] + ".txt"
        test_dataset = CustomDataset(image_path)
        test_dataset.getSingleImage()
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        result = False
        for image, label in test_loader:
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            result = (preds == label.data)

        # 读取图片
        img = cv2.imread(image_path)

        # 标签信息
        have = "Have Fish"
        notHave = "No Fish"
        types = notHave
        if result:
            types = have

        font = cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 4
        font_color = (0, 0, 0)
        line_color = (0, 0, 255)
        text_size = cv2.getTextSize(types, font, font_scale, 2)[0]
        text_width, text_height = text_size[0], text_size[1]
        cv2.rectangle(img, (0, 0), (text_width + 10, text_height + 10), line_color, -1)
        cv2.putText(img, types, (5, text_height + 5), font, font_scale, font_color, thickness=10)
        cv2.imwrite(save_path+'/'+file, img)

        file = open(os.path.join(label_path, txt_filename), "w")
        file.write(str(result[0].item()))
        file.close