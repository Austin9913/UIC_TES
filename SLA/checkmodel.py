from models.FASD_main.models.cmt_fasd import cmt_ti_fasd
from models.FASD_main.models.convnext import convnext_tiny
from models.FASD_main.models.davit_fasd import davit_fasd
from models.FASD_main.models.hifuse_fasd import HiFuse_Tiny_fasd
from models.FASD_main.models.swin_t_fasd import ExtendedSwinV2
from models.FASD_main.models.twins_fasd import twins_fasd
from .modules import *

import torch
from torch import nn

import os, sys
sys.path.append("..")
import models.Uterus as models
from Uterus_Dis_Cl.models.Uterus.davit_main.timm.models.factory import create_model

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2))

def check_model(name, *output_types):
    if len(output_types) == 2:
        if name.startswith('resnet18_pre'):
            model = models.__dict__[name](
                num_classes=128,
            )
            model.num_features = 128
        elif name.startswith('resnet50_pre'):
            model = models.__dict__[name](
                num_classes=512,
            )
            model.num_features = 512
        elif name.startswith('vit_pre'):
            model = models.__dict__[name](
                num_classes=output_types[0],
            )
            model.num_features = output_types[0]
        elif name.startswith('densenet_pre'):
            model = models.__dict__[name](
                num_classes=256,
            )
            model.num_features = 256
        elif name.startswith('vgg_pre'):
            model = models.__dict__[name](
                num_classes=512,
            )
            model.num_features = 512
        elif name.startswith('hifuse'):
            model = models.__dict__[name](
                num_classes=128,
            )
            model.num_features = 128
        elif name.startswith('convnext'):
            model = models.__dict__[name](
                pretrained=True,
                drop_path_rate=0.1,
                layer_scale_init_value=1e-6,
                head_init_scale=0.001,
            )
            model.num_features = 1000
        elif name.startswith('davit'):#双注意力
            model = models.__dict__[name](
                #在Uterus_Dis_Cl/models/Uterus/davit.py控制参数
                #pretrained=True,
                # 这两个参数不会影响到模型在数据集上的准确率或其他
                # 性能指标，而是用于控制模型的部署和集成方式。
                scriptable=None,
                exportable=None,
                #drop_path_rate=args.drop
               # no_jit=None,
            )
            model.num_features = 3
        elif name.startswith('unet'):
            model = models.__dict__[name](
                num_classes=512,
            )
            model.num_features = 512
        elif name.startswith('cmt'):
            model = models.__dict__[name](
                num_classes=512,
                pretrained=True,
                drop_rate=0.0,
                drop_path_rate=0.1,
                # drop_block_rate=args.drop_block,
                img_size=224,
            )
            model.num_features = 1000
        elif name.startswith('swin_v2'):  # swin 未修改
            model = models.__dict__[name](

                pretrained=False,
                scriptable=None,
                exportable=None,

            )
            # in_features = model.head.in_features
            # model.head = nn.Linear(in_features, 100)
            model.num_features = 1000
        elif name.startswith('twins'):
            model = models.__dict__[name](
            )
            model.num_features = 100
        else:
            print("模型输入错误！")
            exit()
        modules = []
        for out in output_types:
            if type(out) is int:
                print("out",out)
                modules.append(nn.Linear(model.num_features, out))
            else:
                raise Exception('out should be integer')
        model = PredictionModel(model, modules)

    elif len(output_types) == 1:
        if name.startswith('resnet50_pre'):
            model = models.__dict__[name](
                num_classes=output_types[0],
            )
        elif name.startswith('densenet_pre'):
            model = models.__dict__[name](
                num_classes=output_types[0],
            )
        elif name.startswith('vgg_pre'):
            model = models.__dict__[name](
                num_classes=output_types[0],
            )
        elif name.startswith('twins'):
            model = models.__dict__[name](
                num_classes=output_types[0],
            )

        else:
            print("模型输入错误！")
            exit()
        # print(model)

    else:
        print("参数个数错误！")
        exit()

    return model
from models.FASD_main.models.resnet import resnet18
from models.FASD_main.models import densenet121
from models.FASD_main.models.vit import vit_pre
def check_model_FASD(name, *output_types):
    if len(output_types) != None:
        if name.startswith('resnet18_pre'):
            model = resnet18(
                num_classes=3,
            )
        elif name.startswith('densenet_pre'):
            model = densenet121(
                num_classes=3,
            )
        elif name.startswith('vit_pre'):
            model = vit_pre(
                num_classes=3,
            )
        elif name.startswith('convnext_tiny'):
            model = convnext_tiny(
                num_classes=3,
            )
        elif name.startswith('davit'):
            model = davit_fasd(
                num_classes=3,
            )
        elif name.startswith('cmt_ti'):
            model = cmt_ti_fasd(
                num_classes=3,
            )
        elif name.startswith('hifuse'):
            model = HiFuse_Tiny_fasd(
                num_classes=3,
            )
        elif name.startswith('twins'):
            model = twins_fasd(
                num_classes=3,
            )
        elif name.startswith('swin_v2'):
            model = ExtendedSwinV2(
                num_classes=3,
            )
        else:
            print("模型输入错误！")
            exit()
    else:
        print("参数个数错误！")
        exit()

    return model