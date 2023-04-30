import json
import os
import shutil
import cv2
import numpy as np
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
    file_name = img_path.split("/")[-2]
    make_dir('video2pic/' + file_name + '/mega_video')
    save_path = 'video2pic/' + file_name + '/mega_video/' + file_name + '.mp4'
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


def mega_detection(fp_in, pt_path):
    """
    Run Mega detector and get json file
        :param fp_in: input image path
        :param pt_path: weight file path
        :return: output json path
    """

    filename = fp_in.split("/")[-2]
    fp_out = 'video2pic/' + filename + '/' + filename + '.json'
    # set python path for mega detector
    # os.system('export PYTHONPATH="$PYTHONPATH:$HOME/Workspace/detector/predationDetector/cameratraps:$HOME/Workspace/detector/predationDetector/ai4eutils:$HOME/Workspace/detector/predationDetector/yolov5"')
    # run mega detector
    detect_commond = "python CameraTraps/detection/run_detector_batch.py " + pt_path + ' ' + fp_in + ' ' + fp_out + " --output_relative_filenames --recursive --checkpoint_frequency 10000"
    os.system(detect_commond)
    return fp_out


def splitJson(json_path):
    """
    Split whole output json file into json file for each picture
        :param json_path: detector output json file
        :return: output json path
    """

    # create folder to keep json file
    video_name = json_path.split("/")[1]
    fp_out = "video2pic/{}/mega_json".format(video_name)
    if not os.path.exists(fp_out):
        os.mkdir(fp_out)

    # split the json file
    with open(json_path, "r") as source_json_fp:
        source_json = json.load(source_json_fp)
        # match each json file with image
        for image in source_json['images']:
            new_json = {'images': [image]}
            for key in source_json:
                if key == "images":
                    continue
                new_json[key] = source_json[key]
            # write new json file
            filename = str(image['file']).split('.')[0]
            print("Writing to {}.json.".format(filename))
            with open("video2pic/{}/mega_json/{}.json".format(video_name, filename), "w") as new_json_fp:
                json.dump(new_json, new_json_fp, indent=1)
    return fp_out


def mega_box(box):
    """
    Calculate the mega box coordinate
        :param box: box[0]=xmin box[1]=ymin box[2]=width box[3]=height
        :return: [xmin, ymin, xmax, ymax]
    """

    xmin, ymin, width, height = box[0], box[1], box[2], box[3]
    xmax = xmin + width
    ymax = ymin + height
    return [xmin, ymin, xmax, ymax]


def trans_mega_json(mega_json, threshold=0.1):
    """
    Draw box based on the mega json file
        :param mega_json: the path to the json file
        :param threshold: the minimum confidence required to consider a detection
    """

    # define the output file path
    fp_out = 'video2pic/' + mega_json.split('/')[1] + '/mega_stats'
    # get the current working directory
    current_path = os.path.abspath('')
    # get the absolute path of the output file
    fp_out = os.path.abspath(fp_out)
    # create the output directory
    make_dir(fp_out)
    # get absolute path of the maga json file
    mega_json = os.path.abspath(mega_json)
    # change to the directory of the json file
    os.chdir(mega_json)
    # get a list of all json files in the directory
    json_files = os.listdir()
    # sort the list of json files
    json_files.sort()

    # load the contents of each json file into a list
    json_data = []
    for i in json_files:
        with open(i, "r") as read_file:
            data = json.load(read_file)
        json_data.append(dict(data))

    # initialise lists to store detection information
    detection_counts = []
    detection_locations = []
    detection_confidences = []

    # loop through the detections and extract only the ones that meet the confidence threshold
    for i in json_data:
        detections = []
        confidences = []

        # get the detection from the current json file data
        temp = i['images'][0]['detections']

        # loop through the detections and extract only the ones that meet the confidence threshold
        for j in temp:
            if (j['category'] == '1') and (j['conf'] > threshold):
                detections.append(mega_box(j['bbox']))
                confidences.append(j['conf'])

        # append the detection information to the corresponding lists
        detection_counts.append(len(detections))
        detection_locations.append(detections)
        detection_confidences.append(confidences)

    # change to the output directory
    os.chdir(fp_out)

    # save to the detection information to text files
    for i in tqdm.tqdm(range(len(detection_locations)), desc="output txt file"):
        np.savetxt(json_files[i].replace('json', 'txt'), np.array(detection_locations[i]))

    # change back to the started directory
    os.chdir(current_path)
    return [json_files, detection_counts, detection_locations, detection_confidences]


def draw_rectangle_by_point(img_file_path, new_img_file_path, points):
    """
    Draw box for detection on the picture
        :param img_file_path: original image path
        :param new_img_file_path: output image path
        :param points: draw points
    """

    # read image
    image = cv2.imread(img_file_path)
    # loop each points
    for item in points:
        point = item[1]
        first_point = (int(point[0]), int(point[1]))
        last_point = (int(point[2]), int(point[3]))
        text_loc = (int((point[0] + (point[2] - point[0]) / 20)), int((point[1] + (point[3] - point[1]) / 10)))
        # draw box
        cv2.rectangle(image, first_point, last_point, (0, 255, 0), 1)
        # add text for the box
        cv2.putText(image, item[0], text_loc, cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 0), thickness=1)
    # write new image
    cv2.imwrite(new_img_file_path, image)
    return


def mega_image(ori_pic_path, mega_json_path, threshold=0.1):
    """
    Output the detection result image
        :param ori_pic_path: original image path
        :param mega_json_path: mega json path
        :param threshold: confidence threshold
        :return: output detection image path
    """

    # get the detection information
    tmp = trans_mega_json(mega_json_path, threshold=threshold)
    detection_locations = tmp[2]
    detection_confidences = tmp[3]

    # create the output path
    fp_out = 'video2pic/' + ori_pic_path.split('/')[1] + '/mega_pic'
    fp_out = os.path.abspath(fp_out)
    make_dir(fp_out)

    # get the current diractory path
    current_path = os.path.abspath('')
    ori_pic_path = os.path.abspath(ori_pic_path)

    # load the original image
    os.chdir(ori_pic_path)
    pics = os.listdir()
    pics.sort()
    for i in tqdm.tqdm(range(len(pics)), desc="Draw detection"):
        img_file_path = os.path.abspath(pics[i])
        # change diractory to the output path
        os.chdir(fp_out)
        new_img_file_path = os.path.abspath(pics[i])
        # change back to the original image path
        os.chdir(ori_pic_path)
        points = []
        image = cv2.imread(img_file_path)
        long = image.shape[1]
        short = image.shape[0]
        for j in range(len(detection_locations[i])):
            tmp_locc = detection_locations[i][j]
            locc = [tmp_locc[0] * long, tmp_locc[1] * short, tmp_locc[2] * long, tmp_locc[3] * short]
            points.append((str(detection_confidences[i][j]), locc))
            # draw box
        draw_rectangle_by_point(img_file_path, new_img_file_path, points)
    os.chdir(current_path)
    return fp_out
