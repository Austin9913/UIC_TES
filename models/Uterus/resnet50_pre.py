import torch
import torch.nn as nn
from torchvision import models


class VariationalDropout(nn.Module):
    def __init__(self, p=0.5):
        super(VariationalDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x

        # 在训练时，生成随机的dropout掩码
        mask = torch.bernoulli(torch.ones_like(x) * (1 - self.p))

        # 对输入进行dropout
        return x * mask / (1 - self.p)


class ResNet(nn.Module):
    def __init__(self, num_classes=1000):

        super(ResNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        #self.model = models.resnet50(pretrained=True)



        # for param in self.model.parameters():
        #     param.requires_grad = False

        '''重写model的classifier属性，重新设计分类器的结构'''

        fc_inputs = self.model.fc.in_features
        self.model.fc=nn.Sequential(
            VariationalDropout(p=0.5),
            nn.Linear(fc_inputs, num_classes)
        )
        #self.model.fc = torch.nn.Linear(fc_inputs, num_classes)


    def forward(self, x):
        x = self.model(x)
        return x


def resnet50_pre(**kwargs):
    # 构建一个ResNet-50模型

    model = ResNet(**kwargs)
    return model

