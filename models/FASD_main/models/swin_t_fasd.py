from torch import nn

from models.Uterus import swin_v2


class ExtendedSwinV2(swin_v2):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 ape=False, patch_norm=True, use_checkpoint=False,
                 pretrained_window_sizes=[0, 0, 0, 0], additional_param=None, **kwargs):
        # 调用父类的初始化方法
        super().__init__(img_size, patch_size, in_chans, num_classes,
                         embed_dim, depths, num_heads, window_size, mlp_ratio,
                         qkv_bias, drop_rate, attn_drop_rate, drop_path_rate,
                         norm_layer, ape, patch_norm, use_checkpoint,
                         pretrained_window_sizes, **kwargs)





    def forward(self, x):
        # 调用修改后的 forward_features
        x = self.forward_features(x)
        fea_vec=x
        x = self.head(x)  # 使用父类的分类头
        return x,fea_vec
