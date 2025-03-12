from __future__ import print_function


import time
import pandas as pd
import torch
import torch.nn.parallel
import torch.nn.functional as F
import numpy as np
import csv

from progress.bar import Bar as Bar
from . import AverageMeter, AUC_Statis, AUC_Single, ACC_Statis, ACC_Single, AUC,\
    Confusion_Matrix_Single, Confusion_Matrix_Statis

__all__ = ['test']


def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc
    # switch to evaluate mode
    model.eval()


    losses = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    patient_ids = []
    neg_preds = []
    labels = []
    image_path =[]



    bar = Bar('Processing', max=len(testloader))
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(testloader):
            data_time.update(time.time() - end)
            if use_cuda:
                inputs, targets = batch_data['image'].cuda(), batch_data['label'].cuda()

            # compute output
            outputs,_ = model(inputs)
            neg_pred = F.softmax(outputs, dim=1)


            # measure record loss
            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))  # 更新已经计算过的所有样本的总损失

            # measure elapsed time
            end = time.time()

            # 生成list，为存入字典做准备
            patient_ids.extend(batch_data['patient_id'])
            neg_preds.extend(neg_pred.detach().cpu().numpy().tolist())  # 正样本的概率
            labels.extend(targets.detach().cpu().numpy().tolist())  # 对应标签
            image_path.extend(batch_data['image_path'])

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Last_bacth_Loss: {loss_0:.4f} | Loss_avg: {loss_1:.4f}'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.avg,
                loss_0=loss,
                loss_1=losses.avg,
            )
            bar.next()
        bar.finish()

    df = pd.DataFrame({'patient_ids': patient_ids, 'neg_preds': neg_preds, 'labels': labels,
                       'image_path': image_path})


    auc_single = AUC_Single(df)  # 图像的auc
    # auc_statis = AUC_Statis(df)  # 病例的auc
    auc_statis, class0_auc_statis, class1_auc_statis, class2_auc_statis = AUC(df)  # 病例的auc
    acc_single = ACC_Single(df)  # 图像的acc
    acc_statis = ACC_Statis(df)  # 病例的auc
    cnf_matrix_single = Confusion_Matrix_Single(df)  # 图像的混淆矩阵
    cnf_matrix_statis = Confusion_Matrix_Statis(df)  # 病例的混淆矩阵

    # print("auc_single", auc_single, "auc_statis", auc_statis, "acc_single", acc_single, "acc_statis", acc_statis,
    #       "cnf_matrix_statis", cnf_matrix_statis, "cnf_matrix_single", cnf_matrix_single)

    return losses.avg, auc_single, auc_statis, class0_auc_statis, class1_auc_statis, class2_auc_statis, \
        acc_single, acc_statis, cnf_matrix_statis, cnf_matrix_single,df