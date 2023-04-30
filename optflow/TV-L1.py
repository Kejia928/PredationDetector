import cv2
import numpy as np
import tqdm
import os
import sys
import shutil

print(cv2.__version__)

print("RGB to TV-L1 optical flow converter")

path = "video2img/20170929_A_4_00-03-26_00-03-31"
frames_path = path + "/ori_img"

frames = os.listdir(frames_path + '/')
frames.sort()
# print(frames)
# exit()

for f in tqdm.tqdm(range(0, len(frames)-1)):
    frame_1_path = frames_path + "/" + frames[f]
    frame_2_path = frames_path + "/" + frames[f+1]

    # print(frame_1_path)
    # print(frame_2_path)

    frame_1 = cv2.imread(frame_1_path)
    frame_2 = cv2.imread(frame_2_path)

    prev_frame = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    next_frame = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

    # Compute and normalise TV-L1 optical flow
    dtvl1 = cv2.optflow.createOptFlow_DualTVL1()
    flow = dtvl1.calc(prev_frame, next_frame, None)

    horz = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
    vert = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
    horz = horz.astype("uint8")
    vert = vert.astype("uint8")

    horizontal_frame_path = path + '/horizontal_flow'
    if not os.path.exists(horizontal_frame_path):
        os.mkdir(horizontal_frame_path)

    vertical_frame_path = path + '/vertical_flow'
    if not os.path.exists(vertical_frame_path):
        os.mkdir(vertical_frame_path)

    # Save optical flow images
    cv2.imwrite(f"{horizontal_frame_path}/frame_{f}.jpg", horz)
    cv2.imwrite(f"{vertical_frame_path}/_frame_{f}.jpg", vert)

    # Pad final image
    frame_3_path = frames_path + "/" + frames[f+2]
    if not os.path.exists(frame_3_path):
        cv2.imwrite(f"{horizontal_frame_path}/frame_{f + 1}.jpg", horz)
        cv2.imwrite(f"{vertical_frame_path}/frame_{f + 1}.jpg", vert)


