import torch

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
        x, size = self.patch_embeds[0](x, (x.size(2), x.size(3)))
        features = [x]
        sizes = [size]
        branches = [0]

        for block_index, block_param in enumerate(self.architecture):
            branch_ids = sorted(set(block_param))
            for branch_id in branch_ids:
                if branch_id not in branches:
                    x, size = self.patch_embeds[branch_id](features[-1], sizes[-1])
                    features.append(x)
                    sizes.append(size)
                    branches.append(branch_id)
            for layer_index, branch_id in enumerate(block_param):
                features[branch_id], _ = self.main_blocks[block_index][layer_index](features[branch_id],
                                                                                    sizes[branch_id])

        features[-1] = self.avgpool(features[-1].transpose(1, 2))
        features[-1] = torch.flatten(features[-1], 1)
        x = self.norms(features[-1])
        fea_vec=x
        x = self.head(x)
        return x,fea_vec
def davit_fasd():
    model = davit(scriptable=None,
                exportable=None,)
    return model
if __name__ == "__main__":
    # 创建模型实例
    num_classes = 10  # 例如设置分类数为10
    model = davit_fasd(num_classes)

    # 创建随机输入数据，假设 batch_size = 1, channels = 3, image_size = 224
    input_data = torch.randn(1, 3, 224, 224)

    # 前向传播并获取输出
    outputs,fea_vec = model(input_data)
    print(outputs)
    print(fea_vec)