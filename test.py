import torch

# 输出带CPU，表示torch是CPU版本的，否则会是+cuxxx
print(f'torch的版本是：{torch.__version__}')
#
print(f'torch是否能使用cuda：{torch.cuda.is_available()}')
