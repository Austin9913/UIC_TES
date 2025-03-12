import argparse
import cv2
import numpy as np
import torch
from torchvision import models

# from convnext import *

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
import matplotlib.pyplot as plt
from .vit_rollout import VITAttentionRollout
from .vit_grad_rollout import VITAttentionGradRollout
# 如果出现 OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



def draw_CAM1(model, image_path, save_path, aug_smooth, eigen_smooth, method):
    """ python cam.py -image-path <path_to_image>
        Example usage of loading an image, and computing:
            1. CAM
            2. Guided Back Propagation
            3. Combining both
    """
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    # model = model.cuda()

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    target_layers = [model.stages[-1]]

    # model = torch.nn.DataParallel(model).cuda()
    cam = methods[method](model=model, target_layers=target_layers, use_cuda=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # img = Image.open(image_path)
    # img = img.resize((224, 224))
    # input_tensor = transform(img).unsqueeze(0)
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category, aug_smooth=aug_smooth, eigen_smooth=eigen_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite(f'/home/temp51/wym/{method}_cam.jpg', cam_image)
    cv2.imwrite(f'/home/temp51/wym/{method}_gb.jpg', gb)
    cv2.imwrite(f'/home/temp51/wym/{method}_cam_gb.jpg', cam_gb)

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}
if __name__ == '__main__':
    # model = models.__dict__["convnext_base"](
    #     pretrained=True,
    #     drop_path_rate=0.1,
    #     layer_scale_init_value=1e-6,
    #     head_init_scale=0.001,
    # )
    # in_22k = False
    # model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.1, layer_scale_init_value=1e-6, head_init_scale=0.001,)
    # url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
    # checkpoint = torch.load(r"C:\Users\18502_kimr\.cache\torch\hub\checkpoints\convnext_base-6075fbad.pth")
    # model.load_state_dict(checkpoint)
    # in_features = model.head.in_features
    # model.head = nn.Linear(in_features, 3)
    # print("-----------")
    #
    # image_path = r"F:\20230305\11both_jpg\1衢州人民医院\hyper\58851\2021-08-03 102807\h0.jpg"
    # save_path = r"F:\cam"
    draw_CAM(model, image_path, save_path, True, True, "gradcam")

'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam', 'eigengradcam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam')
'''