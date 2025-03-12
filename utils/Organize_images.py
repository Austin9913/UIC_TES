import os
import shutil


def unique_name(base_name, ext, output_dir):
    """生成唯一的文件名"""
    counter = 1
    new_name = f"{base_name}_{counter}{ext}"
    while os.path.exists(os.path.join(output_dir, new_name)):
        counter += 1
        new_name = f"{base_name}_{counter}{ext}"
    return new_name


def extract_images(root_dir, output_dir):
    """提取图片并处理文件名冲突"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(subdir, file)
                base_name, ext = os.path.splitext(file)
                output_file_name = file
                output_file_path = os.path.join(output_dir, output_file_name)

                if os.path.exists(output_file_path):
                    # 如果文件名已存在，进行重命名
                    output_file_name = unique_name(base_name, ext, output_dir)
                    output_file_path = os.path.join(output_dir, output_file_name)
                    print("有重复")

                # 复制图片到输出目录
                shutil.copy(file_path, output_file_path)
                print(f"文件 {file_path} 已复制到 {output_file_path}")

    print("图片提取完成。")


# 定义根目录和输出目录
root_dir = '/home/test/Uterus_Dis_Cl/dataset_thyroid/train/malignant/Tiroides4/'  # 替换为你的图片文件夹路径
output_dir = '/home/test/Uterus_Dis_Cl/dataset_thyroid/train/malignant/'  # 替换为你想要输出图片的文件夹路径

extract_images(root_dir, output_dir)