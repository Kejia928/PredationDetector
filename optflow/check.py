import os

def check_folders(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    print("Subfolder number:", len(subfolders))
    
    for subfolder in subfolders:
        ori_path = os.path.join(subfolder, "ori_img")
        horizontal_path = os.path.join(subfolder, "horizontal_flow")
        vertical_path = os.path.join(subfolder, "vertical_flow")
        ori_count = len(os.listdir(ori_path))
        horizontal_count = len(os.listdir(horizontal_path))
        vertical_count = len(os.listdir(vertical_path))
        if ori_count != horizontal_count or ori_count != vertical_count or vertical_count != horizontal_count:
            print(f"{subfolder}: ori文件夹数量为{ori_count}, horizontal文件夹数量为{horizontal_count}")

def check_resnet_folders(folder_path):
    count = 0
    for root, dirs, files in os.walk(folder_path):
        if "resnet_output" in dirs:
            count += 1
        else:
            print(root)
    print(f"在文件夹 {folder_path} 下共有 {count} 个文件夹包含名为 'resnet' 的文件夹。")

check_folders('video2img')
check_resnet_folders('video2img')