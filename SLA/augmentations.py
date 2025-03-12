import torch
import numpy as np

def rotation():
    def _transform(model, images, labels=None):
        size = images.shape[1:]
        return torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1).view(-1, *size)

    return _transform, 4
def mixup_new():

    def _transform(model, images, epoch,labels=None, alpha=1.0, use_cuda=True):

            #mixup

            lam = 0.7 + 0.3 * np.random.beta(2, 2)
            batch_size = images.size(0)
            if use_cuda:
                index = torch.randperm(batch_size).cuda()
            else:
                index = torch.randperm(batch_size)

            mixed_images = lam * images + (1 - lam) * images[index, :]


            # 对混合后的图片进行旋转操作
            size = images.shape[1:]
            mixed_images_rotated = torch.stack([torch.rot90(mixed_images, k, (2, 3)) for k in range(4)], 1).view(-1, *size)

            return mixed_images_rotated

    return _transform, 4

# #=============================================
# #测试
# def _transform(model, images, epoch,labels=None, alpha=1.0, use_cuda=True):
#     # mixup
#     '''Returns mixed inputs, pairs of targets, and lambda'''
#     lam = 0.7 + 0.3 * np.random.beta(2, 2)
#     batch_size = images.size(0)
#     indices = torch.arange(batch_size).long()
#     max_attempts=10
#     if use_cuda:
#         indices = indices.cuda()
#
#     mixed_images = torch.zeros_like(images)
#     mixed_labels = torch.zeros_like(labels) if labels is not None else None
#         # 确保选择相同标签的样本进行混合
#     for i in range(batch_size):
#         for _ in range(max_attempts):  # 尝试指定次数
#             indices[i] = torch.randint(0, batch_size, (1,)).long()  # 重新随机选择索引
#             if labels[i] == labels[indices[i]] and i != indices[i]:  # 检查标签是否相同且不是自己
#                 break  # 找到了符合条件的样本，退出循环
#
#         # 如果找不到符合条件的样本，就使用自己的样本
#         mixed_images[i] = lam * images[i] + (1 - lam) * images[indices[i]] if i != indices[i] else images[i]
#         if mixed_labels is not None:
#             mixed_labels[i] = lam * labels[i] + (1 - lam) * labels[indices[i]] if i != indices[i] else labels[i]
#
#     return mixed_images, mixed_labels
#
# # 测试主函数
# def test_main():
#     # 设置使用CUDA（如果可用）
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#
#     # 生成随机图像数据，例如：10个图像，每个图像大小为3x64x64（假设3个颜色通道）
#     batch_size, num_channels, height, width = 4, 3, 2, 2
#     images = torch.randn(batch_size, num_channels, height, width, device=device)
#
#     # 生成随机标签，例如：10个标签
#     num_classes = 3
#     labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long, device=device)
#
#     # 创建mixup变换函数
#
#
#     # 假设当前是训练的第1个epoch
#     epoch = 1
#     print(images)
#     # 应用mixup变换
#     images,mixed_labels = _transform(None,images, epoch, labels)
#
#     # 打印结果
#     print("Mixed Images Shape:", images.shape)
#     print(images)
#     print("Mixed Labels:", mixed_labels)
#
#
#
# # 运行测试主函数
# test_main()