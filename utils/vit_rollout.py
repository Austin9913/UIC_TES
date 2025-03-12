import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2


def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1)) # torch.Size([197, 197]),生成行数为197，列数为197的数组，且对角线全1，其余部分全0
    n= 0
    with torch.no_grad():
        print("-------------")
        for attention in attentions:

            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)  # torch.Size([1, 197, 197])
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1) # 数组拉平为一行n列
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False) # 得到前0.9*197*197个数
            indices = indices[indices != 0]
            flat[0, indices] = 0 # 取0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0, 1:]

    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask




class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn.drop', head_fusion="max",
                 discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.hook_list = []
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                # module.register_forward_hook(self.get_attention)
                self.hook_list.append(module.register_forward_hook(self.get_attention))
        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())


    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)
        print("self.attentions", len(self.attentions))
        for hook in self.hook_list:
            hook.remove()


        return rollout(self.attentions, self.discard_ratio, self.head_fusion)