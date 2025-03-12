import torch
import torch.nn as nn
from torchvision import models




class ResNet(nn.Module):
    def __init__(self, num_classes=1000):

        super(ResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        fc_inputs = self.model.fc.in_features
        for param in self.model.parameters():
            param.requires_grad = False

        '''重写model的classifier属性，重新设计分类器的结构'''
        self.model.fc = torch.nn.Linear(fc_inputs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


def resnet18_pre(**kwargs):
    # 构建一个ResNet-18模型

    model = ResNet(**kwargs)
    return model

