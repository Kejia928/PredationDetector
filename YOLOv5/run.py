import os

import megaDetector as Mega
import yoloV5 as Yolo


def run_mega_detector(video_path, model_path):
    """
    Before use this function, set the python path to the yolov5
    conda activate cameratraps-detector
    export PYTHONPATH="$PYTHONPATH:$HOME/Workspace/detector/predationDetector/cameratraps:$HOME/Workspace/detector/predationDetector/ai4eutils:$HOME/Workspace/detector/predationDetector/yolov5"
    """

    # get each frame image from video
    ori_path = Mega.video2img(video_path)
    # run mega detector
    json_path = Mega.mega_detection(ori_path, model_path)
    # split the mega json
    mega_json_path = Mega.splitJson(json_path)
    # get the detection in the image
    mega_pic = Mega.mega_image(ori_path, mega_json_path, 0.1)
    # transfer the result image to the video
    Mega.make_video(mega_pic)


def run_yolo_detector(video_path, model_path):
    """
    Before use this function, set the conda encironment to the yoloV5
    conda activate yoloV5
    """

    # get each frame image from video
    ori_path = Yolo.video2img(video_path)
    # # run yolo detector
    yolo_pic = Yolo.yolo_detection(ori_path, model_path)
    # # transfer the result image to the video
    Yolo.make_video(yolo_pic)


if __name__ == '__main__':
    # run_mega_detector("video/SP8_20171019_F_4_6.mp4", "model/md_v5a.0.0_rebuild_pt-1.12_zerolr.pt")
    run_yolo_detector("video/20170929_F_5_3.mp4", "model/yolo/yolov5n_best.pt")
    # all_file = os.listdir("video/")
    # for file in all_file:
    #     if file.split(".")[-1] == "mp4":
    #         path = "video/" + file
    #         des_path = "video2pic/"+file.split(".")[0]
    #         if os.path.exists(des_path):
    #             print("exits")
    #         else:
    #             run_yolo_detector(path, 'model/newbest.pt')
