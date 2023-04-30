import re
import os
import shutil
import tqdm

# # remove some file from dataset
# dir_path = "resnet-dataset/train/notHasFish"
#
# for filename in os.listdir(dir_path):
#     if re.search(r"Augmentation", filename):
#         file_path = os.path.join(dir_path, filename)
#         os.remove(file_path)

# # 指定源文件夹路径
# source_folder = 'video2pic'
#
# # 指定目标文件夹路径
# target_folder = 'image'
#
# # 遍历源文件夹中所有文件夹
# for root, dirs, files in tqdm.tqdm(os.walk(source_folder)):
#     # 遍历当前文件夹中所有文件
#     for filename in files:
#         # 如果文件是图像文件（这里只考虑了png和jpg格式，可以根据需要修改）
#         if filename.endswith('.png') or filename.endswith('.jpg'):
#             # 拼接源文件路径
#             source_file = os.path.join(root, filename)
#             # 拼接目标文件路径
#             target_file = os.path.join(target_folder, filename)
#             # 移动文件到目标文件夹
#             shutil.move(source_file, target_file)

already_file = os.listdir("image_dataset/coco128/images/train") + os.listdir("image_dataset/coco128/images/test")
already_file = [item.split('.')[0] for item in already_file]
already_file = [item.replace("_png", "") for item in already_file]
print(len(already_file))

all_file = os.listdir("image")
print(len(all_file))

# # Remove the already assigned image
# num = 0
# for i in all_file:
#     if i.split('.')[0] in already_file:
#         os.remove("image/" + i)
#         num = num + 1
#
# print(num)
