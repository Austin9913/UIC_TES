import shutil

import albumentations as A
import torch
import math


import cv2

import numpy as np

import pandas as pd

import os

from PIL import Image
from torchvision import transforms

import random

def make_odd(num):
    num = math.ceil(num)
    if num % 2 == 0:
        num += 1
    return num


def med_augment(image_path, name, level=6, number_branch=8,shield=True,seed=8,num=0):

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    #image_path = "/home/lvxinpeng/Uterus_Dis_Ci_new/10all_jpg/images/test/"

    #name 暂定几折就用几

    data_path = "/home/test/Uterus_Dis_Cl/Dataset_BUSI_with_GT/"
    output_path = f"{data_path}/medaugment/train/{name}/"

    transform = A.Compose([
        A.ColorJitter(brightness=0.04 * level, contrast=0, saturation=0, hue=0, p=0.2 * level),
        A.ColorJitter(brightness=0, contrast=0.04 * level, saturation=0, hue=0, p=0.2 * level),
        A.Posterize(num_bits=math.floor(8 - 0.8 * level), p=0.2 * level),
        A.Sharpen(alpha=(0.04 * level, 0.1 * level), lightness=(1, 1), p=0.2 * level),
        A.GaussianBlur(blur_limit=(3, make_odd(3 + 0.8 * level)), p=0.2 * level),
        A.GaussNoise(var_limit=(2 * level, 10 * level), mean=0, per_channel=True, p=0.2 * level),
        A.Rotate(limit=4 * level, interpolation=1, border_mode=0, value=0, mask_value=None, rotate_method='largest_box',
                 crop_border=False, p=0.2 * level),
        A.HorizontalFlip(p=0.2 * level),
        A.VerticalFlip(p=0.2 * level),
        A.Affine(scale=(1 - 0.04 * level, 1 + 0.04 * level), translate_percent=None, translate_px=None, rotate=None,
                 shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                 shear={'x': (0, 2 * level), 'y': (0, 0)}
                 , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),  # x
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                 shear={'x': (0, 0), 'y': (0, 2 * level)}
                 , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0.02 * level), 'y': (0, 0)}, translate_px=None, rotate=None,
                 shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0), 'y': (0, 0.02 * level)}, translate_px=None, rotate=None,
                 shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level)
    ])

    print("-----------")
    image_name = os.path.basename(image_path)

    print("Image name:", image_name)

    file_n, file_s = image_name.rsplit(".", 1)
    # print("文件名部分:", file_n)
    # print("文件扩展名部分:", file_s)
    image = cv2.imread(image_path)

    strategy = [(1, 2), (0, 3), (0, 2), (1, 1)]

    saved_images_path = []
    last_num=0
    for i in range(number_branch):
        if number_branch != 4:
            employ = random.choice(strategy)
        else:
            index = random.randrange(len(strategy))
            employ = strategy.pop(index)
        level, shape = random.sample(transform[:6], employ[0]), random.sample(transform[6:], employ[1])
        img_transform = A.Compose([*level, *shape])
        random.shuffle(img_transform.transforms)
        if not os.path.exists(output_path): os.makedirs(output_path)

        transformed = img_transform(image=image)
        transformed_image = transformed['image']
        cv2.imwrite(f"{output_path}/{file_n}_{i + 1}_{num}.{file_s}", transformed_image)
        saved_images_path.append(f"{output_path}{file_n}_{i + 1}_{num}.{file_s}")
        last_num=i
    if shield:
        cv2.imwrite(f"{output_path}/{file_n}_{number_branch + 1}_{num}.{file_s}", image)
        saved_images_path.append(f"{output_path}{file_n}_{number_branch + 1}_{num}.{file_s}")
    return saved_images_path





def augmenty(csv_path,name):
    #读取CSV文件

    df = pd.read_csv(csv_path, encoding='gbk')

    # 新的DataFrame用于存储增强后的图像信息
    augmented_data = []
    num=1
    # 遍历每行数据
    for index, row in df.iterrows():
        image_path = row["image_path"]
        label = row["label"]
        patient_id = row["patient_id"]
        print(image_path)
        # print("====")
        # 加载图像


        # 对图像进行数据增强

        saved_images_path = med_augment(image_path,name,4,3,True,8,num)


        #更新CSV信息并保存增强后的图像
        for j,temp_path in enumerate(saved_images_path):

            # 更新DataFrame
            augmented_data.append({"image_path": temp_path, "label": label, "patient_id": patient_id})
        num=num+1
    # 将增强后的图像信息保存到新的CSV文件中
    augmented_df = pd.DataFrame(augmented_data)
    dynamic_csv_name = f"kflod_augmented{name}_train.csv"
    augmented_csv_path = "/home/test/Uterus_Dis_Cl/Dataset_BUSI_with_GT/v7_v10/"
    augmented_csv_path = os.path.join(augmented_csv_path, dynamic_csv_name)
    augmented_df.to_csv(augmented_csv_path, index=False)

#augmenty("/home/lvxinpeng/Uterus_Dis_Ci_new/10all_jpg/v7_v10/kflod_5_train.csv","5")
for i in range(1, 6):#甲状腺
    # 调用函数，将i作为第二个参数传递
    augmenty("/home/test/Uterus_Dis_Cl/Dataset_BUSI_with_GT/v7_v10/kflod_{}_train.csv".format(i), i)
# 循环从1到5 #子宫内膜
# for i in range(1, 6):
#     # 调用函数，将i作为第二个参数传递
#     augmenty("/home/lvxinpeng/uterus/Uterus_Dis_Cl/10all_jpg/v7_v10/kflod_{}_train.csv".format(i), i)
# # 循环从1到5 getCsv_BUSI
# for i in range(1, 6):
#     # 调用函数，将i作为第二个参数传递
#     augmenty("/home/lvxinpeng/uterus/Uterus_Dis_Cl/Dataset_BUSI_with_GT/v7_v10/kflod_{}_train.csv".format(i), i)

#测试
# augmenty("/home/lvxinpeng/uterus/Uterus_Dis_Cl/10all_jpg/v7_v10/kflod_test_train.csv".format(1), 1)