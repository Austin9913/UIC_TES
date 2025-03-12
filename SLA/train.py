from __future__ import print_function

import time
import pandas as pd
import torch.nn.functional as F
import torch
import torch.nn as nn
from progress.bar import Bar as Bar
from .eval import AUC_Statis, AUC_Single, ACC_Statis, ACC_Single, AUC,\
    Confusion_Matrix_Single, Confusion_Matrix_Statis
from torch.autograd import Variable
__all__ = ['train_sla_sd', 'train_sla']
def mixup_criterion(y_a, y_b, lam):
    '''Compute the mixup loss'''
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def replace_nan_with_value(tensor, value):
    # 确保输入是一个二维张量
    assert tensor.dim() == 2, "The input tensor must be 2-dimensional"

    # 遍历张量的每个元素
    for i in range(tensor.size(0)):  # 遍历行
        for j in range(tensor.size(1)):  # 遍历列
            if torch.isnan(tensor[i, j]):
                tensor[i, j] = value

    return tensor
#标签平滑处理
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        # 将目标标签转换为独热编码
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        # 平滑处理
        smoothed_target = target_one_hot * self.confidence + (1 - target_one_hot) * self.smoothing / (self.num_classes - 1)

        # 计算交叉熵损失
        loss = nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(pred, dim=1), smoothed_target)
        return loss
def train_sla(trainloader, model, transform, optimizer, T, use_cuda):
    # switch to train mode
    model.train()

    patient_ids = []
    joint_patient_ids = []
    neg_preds = []
    joint_labels_orig = []
    agg_labels_orig = []
    agg_preds_orig = []

    bar = Bar('Processing', max=len(trainloader))

    for batch_idx, batch_data in enumerate(trainloader):

        if use_cuda:
            # torch.Size([64, 3, 224, 224]), torch.Size([64])
            images, labels = batch_data['image'].cuda(), batch_data['label'].cuda()

        batch_size = images.shape[0]  # 64
        images = transform(model, images, labels)  # 256 = 64 * 4
        n = images.shape[0] // batch_size  # rotation（4）、rotation2（2）

        # 3*4种标签，torch.Size([256, 12])
        joint_preds = model(images)
        joint_labels = torch.stack([labels * n + i for i in range(n)], 1).view(-1)  # torch.Size([256])
        joint_patient_id = []
        for i in range(len(batch_data['patient_id']) * n):
            joint_patient_id.append(batch_data['patient_id'][int(i / n)])

        # 计算损失
        joint_loss = F.cross_entropy(joint_preds, joint_labels)
        loss = joint_loss

        agg_preds = 0
        for i in range(n):
            # SLA+AG(聚合推理)：，torch.Size([256, 12]) -> torch.Size([64, 3]),
            # 每次从第i行元素取，步长为n, 取kn+i行（每种旋转的预测结果）
            agg_preds = agg_preds + joint_preds[i::n, i::n] / n

        # compute gradient and do SGD step
        optimizer.zero_grad()  # 先将梯度归零
        loss.backward()  # 然后反向传播计算得到每个参数的梯度值
        optimizer.step()  # 最后通过梯度下降执行一步参数更新

        # 概率进行归一化，用于计算auc
        joint_preds = F.softmax(joint_preds, dim=1)  # 根据不同的dim规则来做归一化操作
        agg_preds = F.softmax(agg_preds, dim=1)  # 根据不同的dim规则来做归一化操作

        patient_ids.extend(batch_data['patient_id'])
        joint_patient_ids.extend(joint_patient_id)
        neg_preds.extend(joint_preds.detach().cpu().numpy().tolist())  # 样本的概率
        joint_labels_orig.extend(joint_labels.detach().cpu().numpy().tolist())  # 对应标签
        agg_labels_orig.extend(labels.detach().cpu().numpy().tolist())  # 对应标签
        agg_preds_orig.extend(agg_preds.detach().cpu().numpy().tolist())  # 对应标签

        # plot progress
        bar.suffix = '({batch}/{size}) joint_loss: {joint_loss:.4f}s | sum_loss: {sum_loss:.4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            joint_loss=joint_loss,
            sum_loss=loss,
        )
        bar.next()
    bar.finish()
    # torch.Size([256, 12])
    df_aug = pd.DataFrame({'patient_ids': joint_patient_ids,
                           'neg_preds': neg_preds, 'labels': joint_labels_orig})

    # torch.Size([64, 3])
    df_agg = pd.DataFrame({'patient_ids': patient_ids,
                           'neg_preds': agg_preds_orig, 'labels': agg_labels_orig})

    agg_auc_single = AUC_Single(df_agg)  # 图像的auc
    agg_auc_statis, agg_class0_auc_statis, agg_class1_auc_statis, agg_class2_auc_statis = AUC(df_agg)  # 病例的auc
    agg_acc_single, _ = ACC_Single(df_agg)  # 图像的acc
    agg_acc_statis, _ = ACC_Statis(df_agg)  # 病例的auc
    agg_cnf_matrix_statis = Confusion_Matrix_Statis(df_agg)  # 病例的混淆矩阵
    agg_cnf_matrix_single = Confusion_Matrix_Single(df_agg)  # 图像的混淆矩阵

    return loss, \
        agg_auc_single, agg_auc_statis, agg_class0_auc_statis, agg_class1_auc_statis, agg_class2_auc_statis, \
        agg_acc_single, agg_acc_statis, agg_cnf_matrix_statis, agg_cnf_matrix_single

def train_sla_sd(trainloader, model, transform, optimizer, T, epoch, use_cuda,args,criterion,pre_data,  pre_joint_out, pre_single_out):
    # switch to train mode
    model.train()
    alpha=0.5
    patient_ids = []
    joint_patient_ids = []
    neg_preds = []
    labels = []
    single_preds_orig = []
    single_labels_orig = []
    agg_preds_s = []
    Tem=1
    bar = Bar('Processing', max=len(trainloader))
    #print("1")
    for batch_idx, batch_data in enumerate(trainloader):
        if use_cuda:
            #print(use_cuda)
            # torch.Size([64, 3, 224, 224]), torch.Size([64])
            images, single_labels = batch_data['image'].cuda(), batch_data['label'].cuda()

        batch_size = images.shape[0]   # 64


        images = transform(model, images,single_labels)  # 256 = 64 * 4


        n = images.shape[0] // batch_size  # rotation（4）、rotation2（2）
        #print("2")
        # 3*4种标签，torch.Size([256, 12])  3种标签，torch.Size([256, 3])
        joint_preds, single_preds = model(images, None)
        #print("3")
        print("joint_preds 的形状:", joint_preds.shape)
        print("single_preds 的形状:", single_preds.shape)






        single_preds = single_preds[::n]   # SLA+SD(加了自蒸馏的单一推理SI)：从聚合知识Pagregate（·|x）到另一个分类器σ（f（x；θ）进行自蒸馏。仅使用非增广或原始样本 torch.Size([64, 3]),从第0行元素取，步长为n,取kn行（原始图像的预测结果）
        joint_labels = torch.stack([single_labels * n + i for i in range(n)], 1).view(-1) # torch.Size([256])
        joint_patient_id = []

        print("single_preds 的形状:", single_preds.shape)

        for i in range(len(batch_data['patient_id'])*n):
            joint_patient_id.append(batch_data['patient_id'][int(i / n)])

        # 计算损失

        single_loss =  F.cross_entropy(single_preds, single_labels)
        joint_loss =  F.cross_entropy(joint_preds, joint_labels)


        agg_preds = 0
        for i in range(n):
            # SLA+AG(聚合推理)：，torch.Size([256, 12]) -> torch.Size([64, 3]),
            # 每次从第i行元素取，步长为n（类）, 取kn+i行（每种旋转的预测结果）
            agg_preds = agg_preds + joint_preds[i::n, i::n] / n

        # distillation_loss = F.kl_div(F.log_softmax(single_preds / T, 1),
        #                              F.softmax(agg_preds.detach() / T, 1),
        #                              reduction='batchmean')
        distillation_loss = F.kl_div(F.log_softmax(single_preds, dim=1), F.softmax(agg_preds.detach(), dim=1),
                                     reduction='batchmean')
        #DBL#################################
        temp =1
        if temp==0:
        #if pre_data != None :

            pre_images, pre_single_labels = pre_data['image'].cuda(), pre_data['label'].cuda()
            pre_joint_labels = torch.stack([pre_single_labels * n + i for i in range(n)], 1).view(
                -1)  # torch.Size([256])
            pre_images = transform(model, pre_images, pre_single_labels)
            joint_preds_pre, single_preds_pre = model(pre_images, None)
            single_preds_pre = single_preds_pre[::n]
            dml_loss_single = (
                    F.kl_div(
                        F.log_softmax(single_preds_pre / Tem, dim=1),
                        F.softmax(pre_single_out.detach() / Tem, dim=1),  # detach
                        reduction="batchmean",
                    )
                    * T
                    * T
            )
            dml_loss_joint = (
                    F.kl_div(
                        F.log_softmax(joint_preds_pre / Tem, dim=1),
                        F.softmax(pre_joint_out.detach() / Tem, dim=1),  # detach
                        reduction="batchmean",
                    )
                    * T
                    * T
            )
            loss_dbl = (alpha * dml_loss_joint + alpha * dml_loss_single)*args.ahadbl
        else:
            loss_dbl = 0  # 留个位置 后续更改
        # DBL#################################

        loss_sla = (joint_loss + single_loss + distillation_loss)*args.ahasla
        #loss_sla = (joint_loss + single_loss + distillation_loss + loss_dbl)*0.4
        # loss = joint_loss + single_loss + distillation_loss.mul(T ** 2)
        loss = loss_sla + loss_dbl

        pre_data=batch_data
        pre_joint_out=joint_preds
        pre_single_out=single_preds
        # compute gradient and do SGD step
        optimizer.zero_grad()  # 先将梯度归零
        loss.backward()  # 然后反向传播计算得到每个参数的梯度值
        optimizer.step()  # 最后通过梯度下降执行一步参数更新

        # 概率进行归一化，用于计算auc
        joint_preds = F.softmax(joint_preds, dim=1)  # 根据不同的dim规则来做归一化操作
        single_preds = F.softmax(single_preds, dim=1)  # 根据不同的dim规则来做归一化操作
        agg_preds = F.softmax(agg_preds, dim=1)  # 根据不同的dim规则来做归一化操作
        # 检查是否包含 NaN 值并处理
        if torch.isnan(joint_preds).any():
            joint_preds = torch.full_like(joint_preds, 20)

        if torch.isnan(single_preds).any():
            single_preds = torch.full_like(single_preds, 20)

        if torch.isnan(agg_preds).any():
            agg_preds = torch.full_like(agg_preds, 20)
        # measure elapsed time
        end = time.time()

        patient_ids.extend(batch_data['patient_id'])
        joint_patient_ids.extend(joint_patient_id)
        neg_preds.extend(joint_preds.detach().cpu().numpy().tolist())  # 样本的概率
        labels.extend(joint_labels.detach().cpu().numpy().tolist())  # 对应标签
        single_preds_orig.extend(single_preds.detach().cpu().numpy().tolist())  # 单一推理
        single_labels_orig.extend(single_labels.detach().cpu().numpy().tolist())  # 对应标签
        agg_preds_s.extend(agg_preds.detach().cpu().numpy().tolist())  # 对应标签


        # plot progress
        bar.suffix = '({batch}/{size}) joint_loss: {joint_loss:.4f}s | single_loss: {single_loss:.4f} ' \
                     '| distillation_loss: {distillation_loss:.4f} | sum_loss: {sum_loss:.4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            joint_loss=joint_loss,
            single_loss=single_loss,
            distillation_loss=distillation_loss.mul(T ** 2),
            sum_loss=loss,
        )
        bar.next()
    bar.finish()

    # torch.Size([256, 12])

    df_aug = pd.DataFrame({'patient_ids': joint_patient_ids,
                            'neg_preds': neg_preds, 'labels': labels})

    # torch.Size([256, 3]) 单一推理
    df_sing = pd.DataFrame({'patient_ids': patient_ids,
                            'neg_preds': single_preds_orig, 'labels': single_labels_orig})
    # torch.Size([64, 3]) 自蒸馏
    df_agg = pd.DataFrame({'patient_ids': patient_ids,
                            'neg_preds': agg_preds_s, 'labels': single_labels_orig})


    sing_auc_single = AUC_Single(df_sing)  # 图像的auc
    sing_auc_statis, sing_class_auc_statis, sing_class1_auc_statis, sing_class2_auc_statis = AUC(df_sing)  # 病例的auc
    sing_acc_single, _ = ACC_Single(df_sing)  # 图像的acc
    sing_acc_statis, _ = ACC_Statis(df_sing)  # 病例的auc
    sing_cnf_matrix_statis = Confusion_Matrix_Statis(df_sing)  # 病例的混淆矩阵
    sing_cnf_matrix_single = Confusion_Matrix_Single(df_sing)  # 图像的混淆矩阵


    agg_auc_single = AUC_Single(df_agg)  # 图像的auc
    agg_auc_statis, agg_class0_auc_statis, agg_class1_auc_statis, agg_class2_auc_statis = AUC(df_agg)  # 病例的auc
    agg_acc_single, _ = ACC_Single(df_agg)  # 图像的acc
    agg_acc_statis, _ = ACC_Statis(df_agg)  # 病例的auc
    agg_cnf_matrix_statis = Confusion_Matrix_Statis(df_agg)  # 病例的混淆矩阵
    agg_cnf_matrix_single = Confusion_Matrix_Single(df_agg)  # 图像的混淆矩阵

    return loss, \
            sing_auc_single,sing_auc_statis,sing_class_auc_statis, sing_class1_auc_statis, sing_class2_auc_statis,\
            sing_acc_single,sing_acc_statis,sing_cnf_matrix_statis,sing_cnf_matrix_single, \
            agg_auc_single,agg_auc_statis,agg_class0_auc_statis, agg_class1_auc_statis, agg_class2_auc_statis, \
            agg_acc_single, agg_acc_statis,agg_cnf_matrix_statis,agg_cnf_matrix_single, pre_data, pre_joint_out, pre_single_out
