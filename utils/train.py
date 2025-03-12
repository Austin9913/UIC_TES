from __future__ import print_function

import numpy as np
import torch
import time
import pandas as pd
import torch.nn.functional as F

from progress.bar import Bar as Bar
from torch import nn

from . import AverageMeter, AUC_Statis, AUC_Single, ACC_Statis, ACC_Single, AUC,\
    Confusion_Matrix_Single, Confusion_Matrix_Statis

__all__ = ['train']


def train(trainloader, model, criterion, optimizer, epoch, use_cuda,pre_data, pre_single_out):
    # switch to train mode
    model.train()
    Tem = 1
    T=0.8
    losses = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    patient_ids = []
    neg_preds = []
    labels = []

    bar = Bar('Processing', max=len(trainloader))

    for batch_idx, batch_data in enumerate(trainloader):
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = batch_data['image'].cuda(), batch_data['label'].cuda()

        # compute model output and softmax output
        outputs = model(inputs)
        neg_pred = F.softmax(outputs, dim=1)  # 根据不同的dim规则来做归一化操作

        # measure record loss
        loss = criterion(outputs, targets)
        # 注意DLB是否放开
        DLB = True
        if DLB:
            if pre_data != None:

                inputs, pre_single_labels = pre_data['image'].cuda(), pre_data['label'].cuda()
                single_preds_pre = model(inputs)
                dml_loss_single = (
                        F.kl_div(
                            F.log_softmax(single_preds_pre / Tem, dim=1),
                            F.softmax(pre_single_out.detach() / Tem, dim=1),  # detach
                            reduction="batchmean",
                        )
                        * T
                        * T
                )


            else:
                dml_loss_single = 0  # 留个位置 后续更改
        losses.update(loss.item(), inputs.size(0))
        pre_data=batch_data
        pre_single_out=outputs


        # compute gradient and do SGD step
        optimizer.zero_grad()  # 先将梯度归零
        loss.backward()  # 然后反向传播计算得到每个参数的梯度值
        optimizer.step()  # 最后通过梯度下降执行一步参数更新


        # measure elapsed time
        end = time.time()

        # 生成list，为存入字典做准备
        # .detach()阻断反向传播，返回值：Tensor，且经过detach()方法后，变量仍然在GPU上
        # .cpu()将数据移至CPU中，返回值：Tensor
        # .numpy()将tensor变量转numpy，返回值：numpy.array()
        # .tolist()将tensor转换为list数据
        # patient_ids.extend(batch_data['patient_id'].detach().cpu().numpy().tolist())
        patient_ids.extend(batch_data['patient_id'])
        neg_preds.extend(neg_pred.detach().cpu().numpy().tolist())  # 正样本的概率
        labels.extend(targets.detach().cpu().numpy().tolist())  # 对应标签

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Last_bacth_Loss: {loss_0:.4f} | Loss_avg: {loss_1:.4f} '.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.avg,
            loss_0=loss,
            loss_1=losses.avg,
        )
        bar.next()
    bar.finish()
    df = pd.DataFrame({'patient_ids': patient_ids, 'neg_preds': neg_preds, 'labels': labels})

    auc_single = AUC_Single(df)  # 图像的auc
    # auc_statis = AUC_Statis(df)  # 病例的auc
    auc_statis, class0_auc_statis, class1_auc_statis, class2_auc_statis = AUC(df) # 病例的auc
    acc_single = ACC_Single(df)  # 图像的acc
    acc_statis = ACC_Statis(df)  # 病例的auc
    cnf_matrix_statis = Confusion_Matrix_Statis(df)  # 病例的混淆矩阵
    cnf_matrix_single = Confusion_Matrix_Single(df)  # 图像的混淆矩阵
    # print("auc_single",auc_single,"auc_statis",auc_statis,"acc_single",acc_single,"acc_statis",acc_statis,
    #       "cnf_matrix_statis",cnf_matrix_statis,"cnf_matrix_single",cnf_matrix_single)

    return losses.avg, auc_single, auc_statis, class0_auc_statis, class1_auc_statis, class2_auc_statis, \
        acc_single, acc_statis, cnf_matrix_statis, cnf_matrix_single,pre_data, pre_single_out
save_change = False
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x
def hard_select(fea_vec, target):

    fea_vec = normalize(fea_vec, axis=-1)

    fea_vec_t = fea_vec.permute(1, 0)


    cos_dis = torch.matmul(fea_vec, fea_vec_t)
    cos_dis_np = cos_dis.detach().cpu().numpy()

    target_list = target.detach().cpu().numpy().tolist()
    for i in range(cos_dis.size(0)):
        for j in range(cos_dis.size(0)):
            if target_list[j] == target_list[i]:
                cos_dis_np[i][j] = 0

    index = []
    for i in range(cos_dis.size(0)):
        tmp = np.argmax(cos_dis_np[i])
        index.append(tmp)
    fea_vec_rank = fea_vec[index]
    exp_fea = fea_vec - fea_vec_rank
    return exp_fea


def intra_extract(fea_vec, target):
    global catagrey_centre
    global catagrey_num
    num_class=3
    batch_catagrey_centre = torch.zeros((num_class, fea_vec.size(1)), device='cuda')
    batch_catagrey_num = torch.zeros((num_class, 1), device='cuda')

    batch_catagrey_num_over_2 = torch.zeros((num_class, 1), device='cuda') ## batch内样本大于2才相减

    batch_fea_centre = []



    target_list = target.detach().cpu().numpy().tolist()

    for i in range(len(target_list)):

        batch_catagrey_centre[target_list[i], :] += fea_vec[i]
        batch_catagrey_num[target[i], :] += 1


    batch_catagrey_centre = torch.div(batch_catagrey_centre, batch_catagrey_num + 1e-10)

    batch_catagrey_num_list = batch_catagrey_num.detach().cpu().numpy().tolist()

    for i in range(len(batch_catagrey_num_list)):
        if batch_catagrey_num_list[i][0] >= 2:
            batch_catagrey_num_over_2[i, :] = 1

    for i in range(len(target_list)):
        tmp_intra = batch_catagrey_centre[target_list[i], :] * batch_catagrey_num_over_2[target_list[i], :]
        batch_fea_centre.append(tmp_intra.reshape(1, -1))

    batch_fea_centre = torch.cat(batch_fea_centre, dim=0)

    fea_sub_intra = fea_vec - batch_fea_centre

    return fea_sub_intra
def loss_fn_kd(outputs, labels, teacher_outputs, alpha, temperature):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """

    T = temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1.0 - alpha)

    return KD_loss
def forward_with_fallback(mode, x):
    # 检查是否存在 mode.fc 属性
    if hasattr(mode, 'fc') and mode.fc is not None:
        return mode.fc(x)
    elif hasattr(mode, 'head') :
        return mode.head(x)
    else:
        raise AttributeError("Neither 'fc' nor 'head' attributes exist in the mode object.")
def FASD_train(trainloader, model, criterion, optimizer, epoch, use_cuda,pre_data, pre_single_out):
    global save_change
    model.train()
    Tem = 1
    T=0.8
    losses = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    patient_ids = []
    neg_preds = []
    labels = []
    bar = Bar('Processing', max=len(trainloader))

    for batch_idx, batch_data in enumerate(trainloader):
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = batch_data['image'].cuda(), batch_data['label'].cuda()

        # compute model output and softmax output


        ##FASD#####################



        if pre_data != None and False:
            pre_inputs, pre_single_labels = pre_data['image'].cuda(), pre_data['label'].cuda()

            inputs = torch.cat([inputs[:, 0, ...], pre_inputs[:, 1, ...]])
            targets = torch.cat([targets, pre_single_labels])



        else:
            #inputs = inputs[:, 0, ...]

            targets = targets



        pre_data=batch_data
        outputs,exp_fea = model(inputs)

        neg_pred = F.softmax(outputs, dim=1)  # 根据不同的dim规则来做归一化操作

        pre_single_out=outputs
        out_all=pre_single_out
        exp_fea_sub = hard_select(exp_fea, targets)  ##减去类间共有特征

        out_sub = forward_with_fallback(model,exp_fea_sub)

        intra_fea_sub = intra_extract(exp_fea, targets)  ## 减去类内共有特征
        out_intra = forward_with_fallback(model,intra_fea_sub)

        loss_all = criterion(out_all, targets)

        loss_IB = loss_fn_kd(out_sub, targets, out_all.detach(), alpha=1, temperature=4)
        loss_intra = loss_fn_kd(out_intra, targets, out_all.detach(), alpha=1, temperature=4)

        loss = loss_all + loss_IB * 30 + loss_intra * 30  ## final loss: = 1.4 : 27 : 10
        loss = min((epoch + 1) / 0.8, 1.0) * loss
        losses.update(loss.item(), inputs.size(0))


        optimizer.zero_grad()  # 先将梯度归零
        loss.backward()  # 然后反向传播计算得到每个参数的梯度值
        optimizer.step()  # 最后通过梯度下降执行一步参数更新


        # measure elapsed time
        end = time.time()

        # 生成list，为存入字典做准备
        # .detach()阻断反向传播，返回值：Tensor，且经过detach()方法后，变量仍然在GPU上
        # .cpu()将数据移至CPU中，返回值：Tensor
        # .numpy()将tensor变量转numpy，返回值：numpy.array()
        # .tolist()将tensor转换为list数据
        # patient_ids.extend(batch_data['patient_id'].detach().cpu().numpy().tolist())
        patient_ids.extend(batch_data['patient_id'])
        neg_preds.extend(neg_pred.detach().cpu().numpy().tolist())  # 正样本的概率
        labels.extend(targets.detach().cpu().numpy().tolist())  # 对应标签

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Last_bacth_Loss: {loss_0:.4f} | Loss_avg: {loss_1:.4f} '.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.avg,
            loss_0=loss,
            loss_1=losses.avg,
        )
        bar.next()
    bar.finish()
    df = pd.DataFrame({'patient_ids': patient_ids, 'neg_preds': neg_preds, 'labels': labels})

    auc_single = AUC_Single(df)  # 图像的auc
    # auc_statis = AUC_Statis(df)  # 病例的auc
    auc_statis, class0_auc_statis, class1_auc_statis, class2_auc_statis = AUC(df) # 病例的auc
    acc_single = ACC_Single(df)  # 图像的acc
    acc_statis = ACC_Statis(df)  # 病例的auc
    cnf_matrix_statis = Confusion_Matrix_Statis(df)  # 病例的混淆矩阵
    cnf_matrix_single = Confusion_Matrix_Single(df)  # 图像的混淆矩阵
    # print("auc_single",auc_single,"auc_statis",auc_statis,"acc_single",acc_single,"acc_statis",acc_statis,
    #       "cnf_matrix_statis",cnf_matrix_statis,"cnf_matrix_single",cnf_matrix_single)

    return losses.avg, auc_single, auc_statis, class0_auc_statis, class1_auc_statis, class2_auc_statis, \
        acc_single, acc_statis, cnf_matrix_statis, cnf_matrix_single,pre_data, pre_single_out