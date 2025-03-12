
"""
    将原始数据集进行划分成训练集、验证集和测试集
"""

import os
import random
import shutil
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from torchvision.datasets import ImageFolder
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from csv import reader

train_per = 0.7
test_per = 0.3
# test_per = 0.1
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', 
    '.bmp', '.BMP', '.tif', '.TIF',
    '.tiff', '.TIFF',
]

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def deldir(exit_dir):
    if os.path.exists(exit_dir):
        shutil.rmtree(exit_dir)

def delfile(exit_file):
    if os.path.exists(exit_file):
        os.remove(exit_file)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# 划分训练集、测试集、验证集
def split_dataset(dataset_dir):
    deldir(os.path.join(dataset_dir, "v7_v7"))
    print(dataset_dir)
    path_list = []
    for root, dirs, files in os.walk(os.path.join(dataset_dir, "1quzhou_people_hospital")):  # 依次遍历文件夹，root为文件夹路径，dirs为此文件夹下的文件夹列表，files为此文件夹下的文件列表
        # 判断数据集有几类
        for file in files:
            if is_image_file(file):  # 判断文件是否为图片
                path = os.path.join(root, file)  # 读取当前文件路径
                father_path = os.path.abspath(os.path.dirname(path) + os.path.sep + "..")  # 父父文件夹
                father_path_name = os.path.basename(father_path)  # 文件路径中直接获取最后一个文件夹名，病例ID
                grader_father = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")  # 父父父文件夹的父文件夹
                grader_grader_name = os.path.basename(grader_father)  # 文件路径中直接获取最后一个文件夹名，疾病分类
                path_list.append(grader_father)
    my_set = set(path_list)  # 转化为集合，去重
    path_list = list(my_set)  # 三类，列表包含三个文件夹的路径
    for sDir in path_list:
        sDir_list = os.listdir(sDir)  # 获取该目录下的所有文件或文件夹目录
        random.seed(342)
        random.shuffle(sDir_list)  # 打乱列表顺序
        sDir_num = len(sDir_list)  # 判断病人个数
        print(sDir, sDir_num)
        train_point = int(sDir_num * train_per)

        sDir_name = os.path.basename(sDir)  # 文件路径中直接获取最后一个文件夹名
        print(sDir, sDir_num,train_point)
        # 建立训练、测试、验证的文件夹并复制病人数据
        for i in range(sDir_num):
            if i < train_point:
                out_dir = os.path.join(dataset_dir, "v7_v7", "train", sDir_name)
            else:
                out_dir = os.path.join(dataset_dir, "v7_v7", "test", sDir_name)

            makedir(out_dir)
            out_path = os.path.join(out_dir, sDir_list[i])
            patient_path = os.path.join(sDir, sDir_list[i])
            shutil.copytree(patient_path, out_path)

# 划分k折验证
def split_kflod(dataset_path, kflod_num):
    df_all_patients = pd.read_csv(os.path.join(dataset_path, "all_patients.csv"))

    floder = KFold(n_splits=kflod_num, random_state=999, shuffle=True)

    df_patients = dict()
    train_files = []  # 存放k折的训练集划分
    test_files = []  # # 存放k折的测试集集划分
    for i in range(3):
        df_patients[i] = df_all_patients.loc[df_all_patients["label"] == i]
        a=floder.split(df_patients[i])
        for k, (Trindex, Tsindex) in enumerate(floder.split(df_patients[i])):
            #修改
            #train_files.append(np.array(df_patients[i])[Trindex].tolist())
            #test_files.append(np.array(df_patients[i])[Tsindex].tolist())
            train_files.append(np.array(df_patients[i])[Trindex, -1].tolist())
            test_files.append(np.array(df_patients[i])[Tsindex, -1].tolist())

    train_list = []
    test_list = []
    for i in range(kflod_num):
        train_patients = []
        test_patients = []
        for j in range(3):
            train_patients.extend(train_files[i + j * kflod_num])
            test_patients.extend(test_files[i + j * kflod_num])
        train_list.append(train_patients)
        test_list.append(test_patients)

    df = pd.DataFrame(data=train_list, index=['0', '1', '2', '4', '5'])
    df.to_csv(os.path.join(dataset_path, "train_patch.csv"))
    df1 = pd.DataFrame(data=test_list, index=['0', '1', '2', '4', '5'])
    df1.to_csv(os.path.join(dataset_path, "test_patch.csv"))

# 得到划分好的k折验证csv
def gen_kflod_csv(dataset_path):

    df_images = pd.read_csv(os.path.join(dataset_path, "all_images.csv"))
    # 修改
    with open(os.path.join(dataset_path, "train_patch.csv"), 'r') as csv_file:
    #with open(os.path.join(dataset_path, "train_patch.csv"), 'r',encoding='utf-8') as csv_file:
        csv_reader = reader(csv_file)
        list_of_rows = list(csv_reader)
        for i in range(1,len(list_of_rows)):
            patient_ids = []
            for j in range(1,len(list_of_rows[i])):
                if list_of_rows[i][j]:
                    #修改
                    #patient_ids.append(list_of_rows[i][j].split("'")[1])
                    patient_ids.append(list_of_rows[i][j])
                else:
                    print("empty！")

            df_image = df_images.loc[df_images["patient_id"].isin(patient_ids)]
            df_image.to_csv(os.path.join(dataset_path, 'kflod_{}_train.csv').format(i), encoding='gbk', index=False)

    with open(os.path.join(dataset_path, "test_patch.csv"), 'r') as csv_file:
        #修改
    #with open(os.path.join(dataset_path, "test_patch.csv"), 'r',encoding='utf-8') as csv_file:
        csv_reader = reader(csv_file)
        list_of_rows = list(csv_reader)
        for i in range(1,len(list_of_rows)):
            patient_ids = []
            for j in range(1,len(list_of_rows[i])):
                if list_of_rows[i][j]:
                    #patient_ids.append(list_of_rows[i][j].split("'")[1])
                    patient_ids.append(list_of_rows[i][j])
                else:
                    print("empty！")
            print(len(patient_ids))
            df_image = df_images.loc[df_images["patient_id"].isin(patient_ids)]
            df_image.to_csv(os.path.join(dataset_path, 'kflod_{}_test.csv').format(i), encoding='gbk', index=False)

# 不分中心的划分训练集、测试集
def split_mix(dataset_path):
    df_all_patients = pd.read_csv(os.path.join(dataset_path, "all_patients.csv"))
    df_patients = dict()
    train_list = []
    test_list =[]
    for i in range(3):
        df_patients[i] = df_all_patients.loc[df_all_patients["label"] == i]
        patients_list = list(df_patients[i]["patient_id"])

        random.seed(999)
        random.shuffle(patients_list)  # 打乱列表顺序
        patients_nums = len(patients_list)
        train_point = int(patients_nums * train_per)
        train_list.extend(patients_list[0:train_point])
        test_list.extend(patients_list[train_point:])

    df_all_images = pd.read_csv(os.path.join(dataset_path, "all_images.csv"))
    df_image_train = df_all_images.loc[df_all_images["patient_id"].isin(train_list)]
    df_image_test = df_all_images.loc[df_all_images["patient_id"].isin(test_list)]

    df_image_train.to_csv(os.path.join(dataset_path, "v7_v5", "train.csv"), encoding='gbk', index=False)
    df_image_test.to_csv(os.path.join(dataset_path, "v7_v5", "test.csv"), encoding='gbk', index=False)

def get_label(grader_father_name):
    if grader_father_name == 'cancer':
        label = '0'
    elif grader_father_name == 'hyper':
        label = '1'
    elif grader_father_name == 'polyp':
        label = '2'
    else:
        print("Error Directory!")
        exit()
    return label
def gen_patients_csv(csv_path, img_dir):
    delfile(csv_path)
    f = open(csv_path, 'w', newline='')  # 创建文件对象
    csv_writer = csv.writer(f)  # 2. 基于文件对象构建 csv写入对象
    csv_writer.writerow(["image_path", "label", "patient_id"])  # 3. 构建列表头
    processed_ids = set()
    for root, dirs, files in os.walk(img_dir, topdown=True):  # 获取train文件下各文件夹名称
        for file in files:
            if is_image_file(file):  # 判断文件是否为图片

                # 读取图片地址
                image_path = os.path.join(root, file)  # 读取当前文件路径
                print(image_path)
                # 读取病例id
                father_path = os.path.abspath(os.path.dirname(image_path) + os.path.sep + "..")  # 父父文件夹
                father_path_name = os.path.basename(father_path)  # 文件路径中直接获取最后一个文件夹名，病例ID
                if father_path_name in processed_ids:
                    continue  # 如果已处理过，则跳过当前循环，继续下一个文件
                else:
                    processed_ids.add(father_path_name)
                    grader_father = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")  # 父父父文件夹的父文件夹
                    grader_father_name = os.path.basename(grader_father)  # 文件路径中直接获取最后一个文件夹名，疾病分类

                    patient_id = father_path_name
                    label = get_label(grader_father_name)

                    line = [image_path, str(label), str(patient_id)]
                    csv_writer.writerow(line)
    f.close()



# 生成对应csv文件包括：图片地址、标签、病人id
def gen_csv(csv_path, img_dir):

    delfile(csv_path)
    f = open(csv_path, 'w', newline='') # 创建文件对象
    csv_writer = csv.writer(f) # 2. 基于文件对象构建 csv写入对象
    csv_writer.writerow(["image_path","label","patient_id"]) # 3. 构建列表头

    for root, dirs, files in os.walk(img_dir, topdown=True):  # 获取train文件下各文件夹名称
        for file in files:
            if is_image_file(file): # 判断文件是否为图片

                # 读取图片地址
                image_path = os.path.join(root, file)  # 读取当前文件路径
                print(image_path)
                # 读取病例id
                father_path = os.path.abspath(os.path.dirname(image_path) + os.path.sep + "..")  # 父父文件夹
                father_path_name = os.path.basename(father_path)  # 文件路径中直接获取最后一个文件夹名，病例ID
                grader_father = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")  # 父父父文件夹的父文件夹
                grader_father_name = os.path.basename(grader_father)  # 文件路径中直接获取最后一个文件夹名，疾病分类

                patient_id = father_path_name
                label = get_label(grader_father_name)

                line = [image_path, str(label), str(patient_id)]
                csv_writer.writerow(line)
    f.close()

# 读取csv得到图片地址、标签、病人id
def get_images_and_labels_and_id(csv_path,state):
    f = open(csv_path, "r", encoding="gbk", errors='ignore')
    imgs_list = []
    label_list = []
    patient_id_list = []
    for line in f: # line为str字符串
        line = line.rstrip() # 删除字符串后指定字符，默认为空格，这里删空行

        # 判断是否是表头，是表头就跳出本次循环
        if line == 'image_path,label,patient_id':
            continue

        line = line.split(',') # 根据分隔符（本次为，）将字符串划分为列表
        imgs_list.append(line[0])
        label_list.append(int(line[1]))
        patient_id_list.append(line[2])
    if state == "train":
        pass
        # df_csv = pd.read_csv(csv_path)
        # df_csv = df_csv.loc[df_csv["label"] == 0]
        #
        # sample_copy = max(label_list.count(1),label_list.count(2)) - label_list.count(0)
        # df_temp = df_csv.sample(n=sample_copy)
        #
        # father_path = os.path.abspath(os.path.dirname(csv_path) + os.path.sep + ".")
        # df_temp.to_csv(os.path.join(father_path, 'temp.csv'), encoding='gbk', index=False)
        # f = open(os.path.join(father_path, 'temp.csv'), "r", encoding="gbk", errors='ignore')
        #
        # for line in f:  # line为str字符串
        #     line = line.rstrip()  # 删除字符串后指定字符，默认为空格，这里删空行
        #     # 判断是否是表头，是表头就跳出本次循环
        #     if line == 'image_path,label,patient_id':
        #         continue
        #     line = line.split(',')  # 根据分隔符（本次为，）将字符串划分为列表
        #     imgs_list.append(line[0])
        #     label_list.append(int(line[1]))
        #     patient_id_list.append(line[2])
        # delfile(os.path.join(father_path, 'temp.csv'))

    return imgs_list, label_list, patient_id_list

#椒盐噪声
class AddPepperNoise(object):
    """"
    Args:
        snr (float): Signal Noise Rate
        p (float): 概率值， 依概率执行
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p: # 按概率进行
            # 把img转化成ndarry的形式
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            # 原始图像的概率（这里为0.9）
            signal_pct = self.snr
            # 噪声概率共0.1
            noise_pct = (1 - self.snr)
            # 按一定概率对（h,w,1）的矩阵使用0，1，2这三个数字进行掩码：掩码为0（原始图像）的概率signal_pct，掩码为1（盐噪声）的概率noise_pct/2.，掩码为2（椒噪声）的概率noise_pct/2.
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            # 将mask按列复制c遍
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255 # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB') # 转化为PIL的形式
        else:
            return img

def get_transform(state):
    if state == "train":
        transform = transforms.Compose([
            # AddPepperNoise(0.9, 0.5),
            transforms.RandomHorizontalFlip(p=0.1),
            # transforms.RandomVerticalFlip(p=0.1),
            # transforms.RandomRotation(45, expand=False, center=None, fill=0, resample=None),
            transforms.Resize([224, 224]),  #
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # transforms.Normalize([0.5],[0.5])
        ])
    elif state == "test":
        transform = transforms.Compose([
            transforms.Resize([224, 224]),  #
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    elif state == "valid":
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return "状态错误"
    return transform

class MyDataset(Dataset):
    def __init__(self, csv_path, state):
        self.csv_path = csv_path
        self.transform = get_transform(state)
        self.images, self.labels, self.patient_id = get_images_and_labels_and_id(self.csv_path, state)
        self.size = len(self.images)        

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]
        patient_id = self.patient_id[index]

        img = Image.open(img_path)  
        if len(img.split())!=3: 
            img =Image.open(img_path).convert('RGB')  # 单通道转为三通道
            
        sample = {'image': img, 'label': label, 'patient_id': patient_id,'image_path': img_path}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample

    def __len__(self):
        return len(self.images)










# 计算归一化的均值和方差
def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0,pin_memory=True)
    mean = torch.zeros(1)
    std = torch.zeros(1)
    for batch_idx, batch_data in enumerate(train_loader):
        X = batch_data['image']
        for d in range(1):
            mean[d] += X[:, :, :].mean()
            std[d] += X[:, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


# if __name__ == '__main__':
#     train_dataset = ImageFolder(root=r'./data/food/', transform=None)
#     print(getStat(train_dataset))


# transform_train = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # 转为三通道
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# if __name__ == '__main__':
#     train_dataset = MyDataset(os.path.join("COVID19\data\COVID19", "train.csv"), transform = transform_train)
#     dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
#     for index, batch_data in enumerate(dataloader):
#         print("index", index)
#         print("image", batch_data['image'])
#         print("label", batch_data['label'])
#         print("patient_id", batch_data['patient_id'])