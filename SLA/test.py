from __future__ import print_function

import time
import pandas as pd
import torch.nn.functional as F
import torch

from progress.bar import Bar as Bar
from .eval import AUC_Statis, AUC_Single, ACC_Statis, ACC_Single, AUC,\
    Confusion_Matrix_Single, Confusion_Matrix_Statis
from torch.autograd import Variable
__all__ = ['test_sla_sd', 'test_sla']


def replace_nan_with_value(tensor, value):
    # 确保输入是一个二维张量
    assert tensor.dim() == 2, "The input tensor must be 2-dimensional"

    # 遍历张量的每个元素
    for i in range(tensor.size(0)):  # 遍历行
        for j in range(tensor.size(1)):  # 遍历列
            if torch.isnan(tensor[i, j]):
                tensor[i, j] = value

    return tensor
def test_sla(testloader, model, transform, epoch, use_cuda):
    global best_acc
    # switch to train mode
    model.eval()

    end = time.time()

    patient_ids = []
    joint_patient_ids = []
    neg_preds = []
    joint_labels_orig = []
    agg_labels_orig = []
    agg_preds_orig = []

    bar = Bar('Processing', max=len(testloader))

    for batch_idx, batch_data in enumerate(testloader):
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

        agg_preds = 0
        for i in range(n):
            # SLA+AG(聚合推理)：，torch.Size([256, 12]) -> torch.Size([64, 3]),
            # 每次从第i行元素取，步长为n, 取kn+i行（每种旋转的预测结果）
            agg_preds = agg_preds + joint_preds[i::n, i::n] / n

        loss = joint_loss


        # 概率进行归一化，用于计算auc
        joint_preds = F.softmax(joint_preds, dim=1)  # 根据不同的dim规则来做归一化操作
        agg_preds = F.softmax(agg_preds, dim=1)  # 根据不同的dim规则来做归一化操作

        # measure elapsed time
        end = time.time()

        patient_ids.extend(batch_data['patient_id'])
        joint_patient_ids.extend(joint_patient_id)
        neg_preds.extend(joint_preds.detach().cpu().numpy().tolist())  # 样本的概率
        joint_labels_orig.extend(joint_labels.detach().cpu().numpy().tolist())  # 对应标签
        agg_labels_orig.extend(labels.detach().cpu().numpy().tolist())  # 对应标签
        agg_preds_orig.extend(agg_preds.detach().cpu().numpy().tolist())  # 对应标签

        # plot progress
        bar.suffix = '({batch}/{size}) joint_loss: {joint_loss:.4f}s | sum_loss: {sum_loss:.4f}'.format(
            batch=batch_idx + 1,
            size=len(testloader),
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
    agg_acc_single, df = ACC_Single(df_agg)  # 图像的acc
    # df.to_csv('/home/temp55/wym/Uterus_Dis_Cl/data/Uterus_v6/v6_v6/Single_valid_{}_df.csv'.format(
    #     "agg_sla+sd_convnext_base_pre_20230329_1"), encoding='gbk')
    agg_acc_statis, df_Statis = ACC_Statis(df_agg)  # 病例的auc
    # df_Statis.to_csv('/home/temp55/wym/Uterus_Dis_Cl/data/Uterus_v6/v6_v6/Statis_valid_{}_df.csv'.format(
    #     "agg_sla+sd_convnext_base_pre_20230329_1"), encoding='gbk')
    agg_cnf_matrix_statis = Confusion_Matrix_Statis(df_agg)  # 病例的混淆矩阵
    agg_cnf_matrix_single = Confusion_Matrix_Single(df_agg)  # 图像的混淆矩阵

    return loss, \
        agg_auc_single, agg_auc_statis, agg_class0_auc_statis, agg_class1_auc_statis, agg_class2_auc_statis, \
        agg_acc_single, agg_acc_statis, agg_cnf_matrix_statis, agg_cnf_matrix_single,

def const_transform(model, images, labels=None):
    #对于自监督来说只能是旋转
        size = images.shape[1:]
        return torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1).view(-1, *size)

def test_sla_sd(testloader, model, transform, T, epoch, use_cuda,args,criterion):
    global best_acc
    # switch to train mode
    model.eval()

    patient_ids = []
    joint_patient_ids = []
    neg_preds = []
    labels = []
    single_preds_orig = []
    single_labels_orig = []
    agg_preds_s = []


    bar = Bar('Processing', max=len(testloader))
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(testloader):
            if use_cuda:
                # torch.Size([64, 3, 224, 224]), torch.Size([64])
                images, single_labels = batch_data['image'].cuda(), batch_data['label'].cuda()

            batch_size = images.shape[0]   # 64
            if args.aug == 'rotation':
                images = transform(model, images, single_labels)  # 256 = 64 * 4
            elif args.aug == 'mixup_new':

                    images = transform(model, images, epoch,single_labels)
                    images = Variable(images)

            n = images.shape[0] // batch_size  # rotation（4）、rotation2（2）

            # 3*4种标签，torch.Size([256, 12])  3种标签，torch.Size([256, 3])
            joint_preds, single_preds = model(images, None)

            single_preds = single_preds[::n]  # torch.Size([64, 3]),从第0行元素取，步长为n取元素行（原始图像的预测结果）
            joint_labels = torch.stack([single_labels * n + i for i in range(n)], 1).view(-1) # torch.Size([256])
            joint_patient_id = []
            for i in range(len(batch_data['patient_id'])*n):
                joint_patient_id.append(batch_data['patient_id'][int(i / n)])

            # 计算损失
            single_loss = F.cross_entropy(single_preds, single_labels)
            joint_loss = F.cross_entropy(joint_preds, joint_labels)


            agg_preds = 0
            for i in range(n):
                # torch.Size([64, 3])
                agg_preds = agg_preds + joint_preds[i::n, i::n] / n

            distillation_loss = F.kl_div(F.log_softmax(single_preds, dim=1), F.softmax(agg_preds.detach(), dim=1), reduction='batchmean')

            loss = joint_loss + single_loss + distillation_loss


            # 概率进行归一化，用于计算auc
            joint_preds = F.softmax(joint_preds, dim=1)  # 根据不同的dim规则来做归一化操作
            single_preds = F.softmax(single_preds, dim=1)  # 根据不同的dim规则来做归一化操作
            agg_preds = F.softmax(agg_preds, dim=1)  # 根据不同的dim规则来做归一化操作

            # measure elapsed time
            end = time.time()

            patient_ids.extend(batch_data['patient_id'])
            joint_patient_ids.extend(joint_patient_id)
            neg_preds.extend(joint_preds.detach().cpu().numpy().tolist())  # 样本的概率
            labels.extend(joint_labels.detach().cpu().numpy().tolist())  # 对应标签
            single_preds_orig.extend(single_preds.detach().cpu().numpy().tolist())  # 样本的概率
            single_labels_orig.extend(single_labels.detach().cpu().numpy().tolist())  # 对应标签
            agg_preds_s.extend(agg_preds.detach().cpu().numpy().tolist())  # 对应标签


            # plot progress
            bar.suffix = '({batch}/{size}) joint_loss: {joint_loss:.4f}s | single_loss: {single_loss:.4f} ' \
                         '| distillation_loss: {distillation_loss:.4f} | sum_loss: {sum_loss:.4f}'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                joint_loss=joint_loss,
                single_loss=single_loss,
                distillation_loss=distillation_loss.mul(T ** 2),
                sum_loss=loss,
            )
            bar.next()
    bar.finish()
    df_aug = pd.DataFrame({'patient_ids': joint_patient_ids,
                            'neg_preds': neg_preds, 'labels': labels})

    # torch.Size([256, 3])
    df_sing = pd.DataFrame({'patient_ids': patient_ids,
                            'neg_preds': single_preds_orig, 'labels': single_labels_orig})
    # torch.Size([64, 3])
    df_agg = pd.DataFrame({'patient_ids': patient_ids,
                            'neg_preds': agg_preds_s, 'labels': single_labels_orig})


    sing_auc_single = AUC_Single(df_sing)  # 图像的auc
    sing_auc_statis, sing_class_auc_statis, sing_class1_auc_statis, sing_class2_auc_statis = AUC(df_sing)  # 病例的auc
    sing_acc_single, df = ACC_Single(df_sing)  # 图像的acc
    # df.to_csv('/home/temp55/wym/Uterus_Dis_Cl/data/Uterus_v6/v6_v6/Single_valid_{}_df.csv'.format(
    #     "sing_sla+sd_convnext_base_pre_20230329_1"), encoding='gbk')
    sing_acc_statis, df_Statis = ACC_Statis(df_sing)  # 病例的auc
    # df_Statis.to_csv('/home/temp55/wym/Uterus_Dis_Cl/data/Uterus_v6/v6_v6/Statis_valid_{}_df.csv'.format(
    #     "sing_sla+sd_convnext_base_pre_20230329_1"), encoding='gbk')
    sing_cnf_matrix_statis = Confusion_Matrix_Statis(df_sing)  # 病例的混淆矩阵
    sing_cnf_matrix_single = Confusion_Matrix_Single(df_sing)  # 图像的混淆矩阵


    agg_auc_single = AUC_Single(df_agg)  # 图像的auc
    agg_auc_statis, agg_class0_auc_statis, agg_class1_auc_statis, agg_class2_auc_statis = AUC(df_agg)  # 病例的auc
    agg_acc_single, df = ACC_Single(df_agg)  # 图像的acc
    # df.to_csv('/home/temp55/wym/Uterus_Dis_Cl/data/Uterus_v6/v6_v6/Single_valid_{}_df.csv'.format(
    #     "agg_sla+sd_convnext_base_pre_20230329_1"), encoding='gbk')
    agg_acc_statis, df_Statis = ACC_Statis(df_agg)  # 病例的auc
    # df_Statis.to_csv('/home/temp55/wym/Uterus_Dis_Cl/data/Uterus_v6/v6_v6/Statis_valid_{}_df.csv'.format(
    #     "agg_sla+sd_convnext_base_pre_20230329_1"), encoding='gbk')
    agg_cnf_matrix_statis = Confusion_Matrix_Statis(df_agg)  # 病例的混淆矩阵
    agg_cnf_matrix_single = Confusion_Matrix_Single(df_agg)  # 图像的混淆矩阵

    return loss, \
            sing_auc_single,sing_auc_statis,sing_class_auc_statis, sing_class1_auc_statis, sing_class2_auc_statis, \
            sing_acc_single,sing_acc_statis,sing_cnf_matrix_statis,sing_cnf_matrix_single, \
            agg_auc_single,agg_auc_statis,agg_class0_auc_statis, agg_class1_auc_statis, agg_class2_auc_statis, \
            agg_acc_single, agg_acc_statis,agg_cnf_matrix_statis,agg_cnf_matrix_single, df_agg
