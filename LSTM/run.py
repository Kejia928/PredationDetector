from VideoLoader import VideoLoader
import numpy as np
import torch
import tqdm
from LSTM import LSTM
import os
import cv2

input_size = 101248
hidden_size = 256
num_layers = 2  
output_size = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LSTM(input_size, hidden_size, num_layers, output_size)
state_dict = torch.load('runs/exp5/best.pt', map_location='cpu')
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

video = VideoLoader(path='../video2img', videoName="SP8_20171019_F_4_00-05-15_00-05-42.mp4")
video.initDataset()
frame_group = video.frame_group
print("video length is : ", video.__len__())
print("One frame dimensiton: ", video.__getitem__(0).shape)

frame_label = []
for i in tqdm.tqdm(range(0, video.__len__())):
    input = video.__getitem__(i)
    input = input.unsqueeze(2)
    input = input.view(1, input.shape[0], -1)
    output = model(input)
    _, preds = torch.max(output, 1)
    frame_label.append(preds)

ori_img_path = os.path.join(video.getVideoFramePath(), 'ori_img')
result_path = os.path.join(video.getVideoFramePath(), 'LSTM_result')
os.makedirs(result_path, exist_ok=True)

events = ""
for i in tqdm.tqdm(range(len(frame_group)), desc="Label Image"):
    label = frame_label[i]
    if label == 1:
        event = "Feeding"
    else :
        event = "Swimming"
    for j in set(frame_group[i]):
        formatted_num =  "%04d" % j
        image_path = os.path.join(ori_img_path, formatted_num + ".jpg")
        save_path = os.path.join(result_path, formatted_num + ".jpg")
        img = cv2.imread(image_path)

        font = cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 4
        font_color = (0, 0, 0)
        line_color = (0, 0, 255)
        text_size = cv2.getTextSize(event, font, font_scale, 2)[0]
        text_width, text_height = text_size[0], text_size[1]
        cv2.rectangle(img, (0, 0), (text_width + 10, text_height + 10), line_color, -1)
        cv2.putText(img, event, (5, text_height + 5), font, font_scale, font_color, thickness=10)
        cv2.imwrite(save_path, img)


# set video parameter
size = (1920, 1080)
fps = 30
print("each picture's size is ({},{})".format(size[0], size[1]))
print("video fps is: " + str(fps))
os.makedirs(video.getVideoFramePath() + '/LSTM_video')
video_path = video.getVideoFramePath() + '/LSTM_video/' + 'result.mp4'
print("save path: " + video_path)

# get the all file name from the input path
all_files = os.listdir(result_path)
# sort the all files
all_files.sort()
index = len(all_files)
print("total image:" + str(index))

# create a video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videowrite = cv2.VideoWriter(video_path, fourcc, fps, size)
img_array = []

# loop each input image file
for filename in all_files:
    img = cv2.imread(os.path.join(result_path, filename))
    # if the input image can not be read
    if img is None:
        print(filename + " is error!")
        continue
    # put the image in an array
    img_array.append(img)

# loop the image array
desc = "make mp4"
for i in tqdm.tqdm(range(index), desc=desc):
    # reset the image size for 1080p video
    img_array[i] = cv2.resize(img_array[i], size)
    # write the image into the video
    videowrite.write(img_array[i])
print('make video completed')