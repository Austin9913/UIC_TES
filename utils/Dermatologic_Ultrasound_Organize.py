

# CSV 文件路径
csv_file = '/home/test/Uterus_Dis_Cl/Dermatologic_Ultrasound/201database.csv'

import csv
import os
import shutil



# 基础路径
base_path = '/home/test/Uterus_Dis_Cl/Dermatologic_Ultrasound/'  # 请替换为您的实际路径

# 读取CSV文件
with open(csv_file, mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # 跳过标题行
    for row in csv_reader:
        if len(row) < 5 or not row[3]:  # 确保行有足够的数据并且dx列不为空
            continue
        dx_value = row[3].strip()  # 获取dx值并去除可能的空格
        if not dx_value:  # 如果dx列是空的，跳过这一行
            continue

        # 构建源文件和目标文件夹路径
        source_bw = os.path.join(base_path, row[0])
        source_doppler = os.path.join(base_path, row[1])
        target_folder = f'{base_path}/dx/{dx_value}'

        # 如果目标文件夹不存在，则创建它
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # 复制文件
        shutil.copy(source_bw, os.path.join(target_folder, os.path.basename(source_bw)))
        shutil.copy(source_doppler, os.path.join(target_folder, os.path.basename(source_doppler)))

print("文件复制完成。")
#处理文件夹在Dermatologic_Ultrasound_Organize2.py