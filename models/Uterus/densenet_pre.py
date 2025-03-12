import torch
import torch.nn as nn
from torchvision import models
from models.Uterus.densent_new.densent_new import DenseNet_new
#from Uterus_Dis_Cl.models.Uterus.DLB_main.models.densenet import densenetd40k12 as DLB_densenet

__all__ = ['densenet_pre']


class DenseNet(nn.Module):

    def __init__(self, num_classes=10,KD=False):
        super(DenseNet, self).__init__()
        # self.model = DenseNet_new(num_classes=num_classes,KD=KD)
        self.KD=KD
        self.model = models.densenet121(pretrained=True)
        #
        #
        fc_inputs = self.model.classifier.in_features
        #self.model.classifier = torch.nn.Linear(fc_inputs, num_classes)
        self.model.classifier = nn.Sequential(
            #VariationalDropout(p=0.3),
            nn.Linear(fc_inputs, num_classes)
        )
    def forward(self, x):
        if self.KD == True:
            x_f,out = self.model(x)
            return x_f, out
        else:
            out = self.model(x)
            return out

def densenet_pre(**kwargs):
    return DenseNet(**kwargs)

if __name__ == '__main__':
    imgs=torch.randn(4,3,224,224).cuda()

    model=DenseNet(num_classes=3).cuda()


    out=model.forward(   imgs)

    print(out.shape)
