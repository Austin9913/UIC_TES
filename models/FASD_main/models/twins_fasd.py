
from models.Uterus.Twins_main.gvt import ALTGVT, GroupBlock


import torch.nn as nn

from functools import partial


from timm.models.vision_transformer import _cfg

class ExtendedALTGVT(ALTGVT):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4],
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[4, 4, 4],
                 sr_ratios=[4, 2, 1], block_cls=GroupBlock, wss=[7, 7, 7],
                 additional_param=None):
        # 调用父类的初始化方法
        super(ExtendedALTGVT, self).__init__(
            img_size, patch_size, in_chans, num_classes, embed_dims, num_heads,
            mlp_ratios, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
            norm_layer, depths, sr_ratios, block_cls, wss
        )


    def forward(self, x):
        x = self.forward_features(x)
        fea_vec=x
        x = self.head(x)

        return x,fea_vec

def alt_gvt_small(pretrained=False, **kwargs):
    model = ExtendedALTGVT(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 10, 4], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model
def twins_fasd(num_classes=3):

    model = alt_gvt_small(pretrained=True, img_size=256,num_classes=num_classes, drop_path_rate=0.1)
    return  model