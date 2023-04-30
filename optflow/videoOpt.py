import os
import cv2
import tqdm
import numpy

video_folder = 'feeding_dataset/training_feeding_v1'
img_folder = 'video2img'

if not os.path.exists(img_folder):
    os.mkdir(img_folder)

work = os.listdir(video_folder)
work.sort()
# work = work[0:4]

for filename in tqdm.tqdm(work):
    if filename.endswith('.mp4'):
        video_name = os.path.splitext(filename)[0]
        video_folder_path = os.path.join(img_folder, video_name)
        if not os.path.exists(video_folder_path):
            print(video_folder_path)
            os.mkdir(video_folder_path)
        else:
            continue
        ori_img = os.path.join(video_folder_path, "ori_img")
        if not os.path.exists(ori_img):
            os.mkdir(ori_img)
        video_path = os.path.join(video_folder, filename)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if ret:
                img_path = os.path.join(ori_img, f'{frame_count:04d}.jpg')
                cv2.imwrite(img_path, frame)
                frame_count += 1
            else:
                break
        
        cap.release()

        frames_path = ori_img

        frames = os.listdir(frames_path + '/')
        frames.sort()

        for f in range(0, len(frames)):
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

            horizontal_frame_path = video_folder_path + '/horizontal_flow'
            if not os.path.exists(horizontal_frame_path):
                os.mkdir(horizontal_frame_path)


            vertical_frame_path = video_folder_path + '/vertical_flow'
            if not os.path.exists(vertical_frame_path):
                os.mkdir(vertical_frame_path)

            # Save optical flow images
            cv2.imwrite(f"{horizontal_frame_path}/frame_{f}.jpg", horz)
            cv2.imwrite(f"{vertical_frame_path}/frame_{f}.jpg", vert)

            # Pad final image
            if(f+1 >= len(frames)-1):
                cv2.imwrite(f"{horizontal_frame_path}/frame_{f + 1}.jpg", horz)
                cv2.imwrite(f"{vertical_frame_path}/frame_{f + 1}.jpg", vert)
                break
