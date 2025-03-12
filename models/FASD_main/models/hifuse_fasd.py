import torch
import torch.nn as nn

from models.Uterus.HiFuse_main.main_model import main_model


class ExtendedMainModel(main_model):
    def __init__(self, num_classes, new_param=None, **kwargs):
        # 调用父类的构造函数初始化
        super().__init__(num_classes, **kwargs)
        self.fc=self.conv_head
    def forward(self, imgs):
        ######  Global Branch ######
        x_s, H, W = self.patch_embed(imgs)
        x_s = self.pos_drop(x_s)
        x_s_1, H, W = self.layers1(x_s, H, W)
        x_s_2, H, W = self.layers2(x_s_1, H, W)
        x_s_3, H, W = self.layers3(x_s_2, H, W)
        x_s_4, H, W = self.layers4(x_s_3, H, W)

        # [B,L,C] ---> [B,C,H,W]
        x_s_1 = torch.transpose(x_s_1, 1, 2)
        x_s_1 = x_s_1.view(x_s_1.shape[0], -1, 56, 56)
        x_s_2 = torch.transpose(x_s_2, 1, 2)
        x_s_2 = x_s_2.view(x_s_2.shape[0], -1, 28, 28)
        x_s_3 = torch.transpose(x_s_3, 1, 2)
        x_s_3 = x_s_3.view(x_s_3.shape[0], -1, 14, 14)
        x_s_4 = torch.transpose(x_s_4, 1, 2)
        x_s_4 = x_s_4.view(x_s_4.shape[0], -1, 7, 7)

        ######  Local Branch ######
        x_c = self.downsample_layers[0](imgs)
        x_c_1 = self.stages[0](x_c)
        x_c = self.downsample_layers[1](x_c_1)
        x_c_2 = self.stages[1](x_c)
        x_c = self.downsample_layers[2](x_c_2)
        x_c_3 = self.stages[2](x_c)
        x_c = self.downsample_layers[3](x_c_3)
        x_c_4 = self.stages[3](x_c)

        ###### Hierachical Feature Fusion Path ######
        x_f_1 = self.fu1(x_c_1, x_s_1, None)
        x_f_2 = self.fu2(x_c_2, x_s_2, x_f_1)
        x_f_3 = self.fu3(x_c_3, x_s_3, x_f_2)
        x_f_4 = self.fu4(x_c_4, x_s_4, x_f_3)
        x_fu = self.conv_norm(x_f_4.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
        fea_vec=x_fu
        x_fu = self.conv_head(x_fu)

        return x_fu,fea_vec
def HiFuse_Tiny_fasd(num_classes: int):
    model = ExtendedMainModel(depths=(2, 2, 2, 2),
                     conv_depths=(2, 2, 2, 2),
                     num_classes=num_classes)
    return model