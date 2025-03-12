from models.Uterus.davit_main.timm.models.factory import create_model
import torch.nn as nn


class davit(nn.Module):
    def __init__(self,scriptable,exportable, model_name='DaViT_small',):
        super(davit, self).__init__()

        self.model_name = model_name
        #self.pretrained = pretrained
        self.scriptable = scriptable
        self.exportable=exportable

        # 调用 create_model 函数创建模型
        self.model = create_model(
            model_name,
            pretrained=False,
            num_classes=3,
            drop_rate=0.7,  # 没用，函数内没使用
            # drop_connect_rate=0.1,  # DEPRECATED, use drop_path
            drop_path_rate=0.5,  # 这个有用
            bn_tf=False,
            bn_momentum=0.5,
            bn_eps=1e-5,
            scriptable=False,
            checkpoint_path='/home/test/Uterus_Dis_Cl/models/Uterus/davit_main/pre/davit-small_3rdparty_in1k_20221116-51a849a6.pth'

        )

    def forward(self, x):
        return self.model(x)
