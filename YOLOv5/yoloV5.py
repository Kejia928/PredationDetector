import os
import shutil
import cv2
import tqdm


def make_dir(path):
    """
    Create directory
        :param path: directory path
    """

    # if the path exist, delete it
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return


def video2img(fp_in):
    """
    Cut video into picture
        :param fp_in: input video path
        :return: image path
    """

    # get video file name
    file_name = fp_in.replace("\\", "/").split("/")[-1].split(".")[0]
    # get input video file absolut path
    fp_in = os.path.abspath(fp_in)

    # create output path
    fp_out = 'video2pic/' + file_name + '/ori_pic'
    fp_out_abs = os.path.abspath(fp_out)
    make_dir(fp_out_abs)

    # open video
    vc = cv2.VideoCapture(fp_in)
    # get frame number
    frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    # loop each frame
    desc = "cut " + file_name
    for i in tqdm.tqdm(range(frames), desc=desc):
        # rval is a boolean value that indicates whether the next frame was successfully read from the video.
        # If rval is True, it means that the frame was successfully read, and the frame data is stored.
        # If rval is False, it means that there are no more frames to read and the loop should break.
        rval, frame = vc.read()
        if rval:
            # save the current frame of the video as an image file
            cv2.imwrite(f"""{fp_out_abs}/{str(i).rjust(6, "0")}__{file_name}.png""", frame)
            cv2.waitKey(1)
        else:
            break

    # close the video file
    vc.release()
    print("cut video completed")
    return fp_out


def make_video(img_path):
    """
    Transfer the image to video
        :param img_path: the image folder path
    """

    # set video parameter
    size = (1920, 1080)
    fps = 30
    print("each picture's size is ({},{})".format(size[0], size[1]))
    print("video fps is: " + str(fps))
    file_name = img_path.split("/")[-3]
    make_dir('video2pic/' + file_name + '/yolo_video')
    save_path = 'video2pic/' + file_name + '/yolo_video/' + file_name + '.mp4'
    print("save path: " + save_path)
    img_path = img_path+"/"

    # get the all file name from the input path
    all_files = os.listdir(img_path)
    # sort the all files
    all_files.sort()
    index = len(all_files)
    print("total image:" + str(index))

    # create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowrite = cv2.VideoWriter(save_path, fourcc, fps, size)
    img_array = []

    # loop each input image file
    for filename in all_files:
        img = cv2.imread(img_path + filename)
        # if the input image can not be read
        if img is None:
            print(filename + " is error!")
            continue
        # put the image in an array
        img_array.append(img)

    # loop the image array
    desc = "make " + file_name + ".mp4"
    for i in tqdm.tqdm(range(index), desc=desc):
        # reset the image size for 1080p video
        img_array[i] = cv2.resize(img_array[i], size)
        # write the image into the video
        videowrite.write(img_array[i])
    print('make video completed')
    return


def yolo_detection(fp_in, pt_path):
    """
    Run YoloV5 detector and get detection image file
        :param fp_in: input image path
        :param pt_path: weight file path
        :return: output detection image path
    """

    filename = fp_in.split("/")[-2]
    fp_in = os.path.abspath(fp_in)
    pt_path = os.path.abspath(pt_path)
    fp_out = 'video2pic/' + filename + '/yolo_pic'
    fp_out_abs = os.path.abspath(fp_out)
    # run yolo detector
    # detect_commond = "python yolov5_old/detect.py --weights " + pt_path + ' --source ' + fp_in + ' --project ' + fp_out_abs
    detect_commond = "python yolov5/detect.py --weights " + pt_path + ' --source ' + fp_in + ' --project ' + fp_out_abs + ' --save-txt'
    os.system(detect_commond)
    return fp_out+"/exp"


def ResnetDetector():
    # 读取图片
    img = cv2.imread("video-dataset/coco128/images/train/000000__20170929_F_5_5_png.rf.4072bb8da4dc77cffbc4ddcb5b6a1906.jpg")

    # 标签信息
    label = "your_label"

    # 设置字体、字体大小、字体颜色和线条颜色
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    line_color = (0, 0, 255)

    # 获取文本框大小
    text_size = cv2.getTextSize(label, font, font_scale, 1)[0]
    text_width, text_height = text_size[0], text_size[1]

    # 绘制文本框和文本
    cv2.rectangle(img, (0, 0), (text_width + 10, text_height + 10), line_color, -1)
    cv2.putText(img, label, (5, text_height + 5), font, font_scale, font_color, 1)

    # 显示图片
    cv2.imshow("Image with label", img)
    cv2.waitKey(0)
