import torch
from pytorch_pretrained_vit import ViT

def vit_pre(num_classes):
    model = CustomViT(num_classes=num_classes)
    return model


class CustomViT(ViT):
    def __init__(self, num_classes):
        super().__init__('B_16_imagenet1k', pretrained=True, image_size=224, num_classes=num_classes)

    def forward(self, x):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)  # b,gh*gw+1,d
        x = self.transformer(x)  # b,gh*gw+1,d
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc'):
            x = self.norm(x)[:, 0]  # b,d
            fea_vec = x

            x = self.fc(x)

        return x,fea_vec
if __name__ == "__main__":
    # 创建模型实例
    num_classes = 10  # 例如设置分类数为10
    model = vit_pre(num_classes)

    # 创建随机输入数据，假设 batch_size = 1, channels = 3, image_size = 224
    input_data = torch.randn(1, 3, 224, 224)

    # 前向传播并获取输出
    outputs,fea_vec = model(input_data)
    print(outputs)
    print(fea_vec)

