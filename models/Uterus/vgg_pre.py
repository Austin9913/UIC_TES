'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei 
'''

import torch
import torch.nn as nn
from torchvision import models

__all__ = [
    'VGG',  'vgg_pre',
]


class VGG(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        self.model = models.vgg16(pretrained=True)

        # for param in self.model.parameters():
        #     param.requires_grad = False

        '''重写model的classifier属性，重新设计分类器的结构'''
        fc_inputs = self.model.classifier[6].in_features
        self.model.classifier[6] = torch.nn.Linear(fc_inputs, num_classes)


    def forward(self, x):
        x = self.model(x)
        return x
    


def vgg_pre(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(**kwargs)
    return model

