import csv
import os
from _csv import reader

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM',
    '.bmp', '.BMP', '.tif', '.TIF',
    '.tiff', '.TIFF',
]

def is_image_file(filename):
    # 检查文件是否为PNG图片且不包含'_mask.png'
    return filename.lower().endswith('.png') and '_mask.png' not in filename
def delfile(exit_file):
    if os.path.exists(exit_file):
        os.remove(exit_file)
def get_label(grader_father_name):
    if grader_father_name == 'normal':
        label = '0'
    elif grader_father_name == 'malignant':
        label = '1'
    elif grader_father_name == 'benign':
        label = '2'
    else:
        print("Error Directory!")
        exit()
    return label
# 生成对应csv文件包括：图片地址、标签、病人id
def gen_csv(csv_path, img_dir):

    delfile(csv_path)
    f = open(csv_path, 'w', newline='') # 创建文件对象
    csv_writer = csv.writer(f) # 2. 基于文件对象构建 csv写入对象
    csv_writer.writerow(["image_path","label","patient_id"]) # 3. 构建列表头
    temp_id=0
    for root, dirs, files in os.walk(img_dir, topdown=True):  # 获取train文件下各文件夹名称
        for file in files:
            if is_image_file(file): # 判断文件是否为图片

                # 读取图片地址
                image_path = os.path.join(root, file)  # 读取当前文件路径
                print(image_path)
                # 读取病例id
                current_dir_path = os.path.dirname(image_path)

                # 如果你想要目录名作为标签，可以使用 os.path.basename 来获取
                label = os.path.basename(current_dir_path)

                label = get_label(label)

                patient_id = temp_id

                temp_id = temp_id+1
                line = [image_path, str(label),"id"+ str(patient_id)]
                csv_writer.writerow(line)
    f.close()



def gen_patients_csv(csv_path, img_dir):
    delfile(csv_path)
    f = open(csv_path, 'w', newline='')  # 创建文件对象
    csv_writer = csv.writer(f)  # 2. 基于文件对象构建 csv写入对象
    csv_writer.writerow(["image_path", "label", "patient_id"])  # 3. 构建列表头
    temp_id = 0
    for root, dirs, files in os.walk(img_dir, topdown=True):  # 获取train文件下各文件夹名称
        for file in files:
            if is_image_file(file):  # 判断文件是否为图片

                # 读取图片地址
                image_path = os.path.join(root, file)  # 读取当前文件路径
                print(image_path)
                # 读取病例id
                current_dir_path = os.path.dirname(image_path)

                # 如果你想要目录名作为标签，可以使用 os.path.basename 来获取
                label = os.path.basename(current_dir_path)
                label = get_label(label)
                patient_id = temp_id

                temp_id = temp_id + 1
                line = [image_path, str(label), "id"+str(patient_id)]
                csv_writer.writerow(line)
    f.close()

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

def test_gen_csv():
    # 定义 CSV 文件的路径

    image_csv_file_path = "/home/test/Uterus_Dis_Cl/Dataset_BUSI_with_GT/v7_v10/all_images.csv"
    patients_csv_file_path = "/home/test/Uterus_Dis_Cl/Dataset_BUSI_with_GT/v7_v10/all_patients.csv"

    # 定义图片文件夹的路径
    images_directory = "/home/test/Uterus_Dis_Cl/Dataset_BUSI_with_GT/"
    split_kflod_file_path= "/home/test/Uterus_Dis_Cl/Dataset_BUSI_with_GT/v7_v10"
    # 调用 gen_csv 函数生成 CSV 文件
    gen_csv(image_csv_file_path, images_directory)
    gen_patients_csv(patients_csv_file_path, images_directory)
    split_kflod(split_kflod_file_path, 5)
    gen_kflod_csv(split_kflod_file_path)
    print(f"CSV 文件已生成在：{split_kflod_file_path}")


#gen_csv(os.path.join(args.dataset_path, "v7_v10", "all_images.csv"), os.path.join(args.dataset_path))

# 执行测试主函数
if __name__ == "__main__":
    test_gen_csv()