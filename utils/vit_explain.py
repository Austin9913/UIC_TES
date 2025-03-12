import argparse
import sys
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import csv
from PIL import Image
import torch
import numpy as np
import cv2
import shutil
import matplotlib.pyplot as plt
from .vit_rollout import VITAttentionRollout
from .vit_grad_rollout import VITAttentionGradRollout

from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
# def ge_
def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def deldir(exit_dir):
    if os.path.exists(exit_dir):
        shutil.rmtree(exit_dir)

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def cls_path(file):
    if file.find("癌")!= -1 or file.find("cancer")!= -1:
        path="cancer"
    elif file.find("增生")!= -1 or file.find("hyper")!= -1:
        path="hyper"
    elif file.find("息肉")!= -1 or file.find("polyp")!= -1:
        path="polyp"
    else:
        print("路径错误！",file)
        pass
    return path

def draw_CAM(arch, model,image_path,save_path,head_fusion,discard_ratio,category_index):

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    model = model.cuda()

    if category_index is None:
        cate_name = "cam_attention_rollout_{:.3f}_{}".format(discard_ratio, head_fusion)
    else:
        cate_name = "cam_grad_rollout_{}_{:.3f}_{}".format(category_index, discard_ratio, head_fusion)
    deldir(os.path.join(save_path, arch, cate_name))

    with open(image_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == "image_path":
                pass
            else:
                patient_id = row[2]
                image_path = row[0]
                cls = cls_path(image_path)  # 疾病分类

                path = image_path
                path = path.split('/')[-1]
                id_name = path.split('.')[0]
                save = os.path.join(save_path, arch, cate_name, cls)
                print(save)
                makedir(save)

                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
                img = Image.open(image_path)
                # img = img.resize((224, 224))
                input_tensor = transform(img).unsqueeze(0).cuda()

                if arch.startswith('convnext'):
                    target_layers = [model.stages[-1]]
                    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
                    grayscale_cam = cam(input_tensor=input_tensor, targets =category_index)  # [batch, 224,224]
                    mask = grayscale_cam[0] #  (224, 224)
                    print("mask.shape", mask.shape)

                elif arch.startswith('vit'):
                    attention_rollout = VITAttentionRollout(model, head_fusion=head_fusion, discard_ratio=discard_ratio)
                    mask = attention_rollout(input_tensor)
                    print("mask.shape",mask.shape)

                else:
                    print("模型输入错误")
                    exit()

                name = os.path.join(save, "{}-{}.jpg".format(patient_id,id_name))
                np_img = np.array(img)[:, :, ::-1]
                mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
                mask = show_mask_on_image(np_img, mask)
                cv2.imwrite(name, mask)
