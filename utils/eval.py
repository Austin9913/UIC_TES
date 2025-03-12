
from __future__ import print_function, absolute_import
import os
import csv
import pandas as pd
import time

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, f1_score,classification_report
import matplotlib.pyplot as plt



__all__ = ['auc_calculate_gen_csv', 'accuracy','Save_Metrics','Save_C_M',
           'AUC_Statis', 'AUC_Single', 'ACC_Statis', 'ACC_Single','AUC',
           'Confusion_Matrix_Single', 'Confusion_Matrix_Statis']



def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def delfile(exit_file):
    if os.path.exists(exit_file):
        os.remove(exit_file)

# 将图像为主体转换为以病例为主体
def Single_To_Statis(df):
    patient_ids = df['patient_ids'].to_numpy().tolist()  # 转为列表
    patient_ids = list(set(patient_ids))  # 转为集合去重后，再转为列表

    neg_mean_preds = []
    patient_mean_ids = []
    labels_mean = []
    for i, patient_id in enumerate(patient_ids):
        df1 = df.loc[df["patient_ids"] == patient_id]  #

        # 得到病例的均值概率
        neg_pred = df1['neg_preds'].to_numpy()
        neg_preds = np.empty((len(df1), 3), dtype=float)

        for i in range(len(df1)):
            neg_preds[i] = np.array(neg_pred[i])
        neg_preds = np.mean(neg_preds, axis=0)

        label = df1['labels'].to_numpy().tolist()  # 得到病例标签
        patient_id_ = df1['patient_ids'].to_numpy().tolist()  # 得到病例ID

        neg_mean_preds.extend([neg_preds.tolist()])
        patient_mean_ids.extend([patient_id_[0]])
        labels_mean.extend([label[0]])

    df_Statis = pd.DataFrame({'patient_ids': patient_mean_ids,
                              'neg_preds': neg_mean_preds,
                              'labels': labels_mean})
    return df_Statis

# 病例的auc
def AUC_Statis(df):
    df_Statis = Single_To_Statis(df)
    ytrue = df_Statis['labels'].to_numpy().tolist()
    ypred = df_Statis['neg_preds'].to_numpy().tolist()
    statis = roc_auc_score(ytrue, ypred, multi_class='ovr')
    return statis

# 图像的auc
def AUC_Single(df):
    ytrue = df['labels'].to_numpy().tolist()
    ypred = df['neg_preds'].to_numpy().tolist()
    single = roc_auc_score(ytrue, ypred, multi_class='ovr')
    return single
#  convnext_base  vit_pre

def AUC(df):
    df_Statis = Single_To_Statis(df)

    person_preds = df_Statis['neg_preds'].to_numpy().tolist()
    person_label = df_Statis['labels'].to_numpy().tolist()

    roc_auc = dict()
    y_true = label_binarize(person_label, classes=[0,1,2]) # ,4,5,6,7
    y_true = np.array(y_true)
    person_preds = np.array(person_preds)
    for i in range(3):
        roc_auc[i] = roc_auc_score(y_true[:, i], person_preds[:, i])
        print("class {} ".format(i) + 'statis auc ' + '{:.4f}'.format(roc_auc[i]))
    macro_auc = roc_auc_score(y_true, person_preds)
    return macro_auc, roc_auc[0], roc_auc[1], roc_auc[2]

# 病例的acc
def ACC_Statis(df):
    df_Statis = Single_To_Statis(df)
    neg_pred = df_Statis['neg_preds'].to_numpy().tolist()
    ypred = np.argmax(neg_pred, axis=1)
    df_Statis['statis'] = ypred
    acc_statis = (df_Statis['labels'] == df_Statis['statis']).mean()  # 计算成功率
    # df_Statis.to_csv('/home/temp55/wym/Uterus_Dis_Cl/data/Uterus_v7/v7_v1/Statis_{}_df.csv'.format("baseline_convnext_base_pre_20230415_4"), encoding='gbk')
    return acc_statis

# 图像的acc
def ACC_Single(df):
    neg_pred = df['neg_preds'].to_numpy().tolist()
    ypred = np.argmax(neg_pred, axis=1)
    df['single'] = ypred
    acc_single = (df['labels'] == df['single']).mean()  # 计算成功率
    # df.to_csv('/home/temp55/wym/Uterus_Dis_Cl/data/Uterus_v7/v7_v1/Single_{}_df.csv'.format("baseline_convnext_base_pre_20230415_4"), encoding='gbk')
    return acc_single

# 图像计算混淆矩阵
def Confusion_Matrix_Single(df):
    ytrue = df['labels'].to_numpy().tolist()
    ypred = df['neg_preds'].to_numpy().tolist()
    ypred = np.argmax(ypred, axis=1)
    cnf_matrix_single = confusion_matrix(ytrue, ypred)
    return cnf_matrix_single

# 病例计算混淆矩阵
def Confusion_Matrix_Statis(df):
    df_Statis = Single_To_Statis(df)
    ytrue = df_Statis['labels'].to_numpy().tolist()
    ypred = df_Statis['neg_preds'].to_numpy().tolist()
    ypred = np.argmax(ypred, axis=1)
    cnf_matrix_statis = confusion_matrix(ytrue, ypred)
    return cnf_matrix_statis

def makefile(new_file):
    if not os.path.exists(new_file):
        f = open(new_file, 'w', newline='')  # 创建文件对象
        csv_writer = csv.writer(f)  # 2. 基于文件对象构建 csv写入对象
        csv_writer.writerow(["Time", "model", "state", "epoch", "learning_rate", "batch_size", "Avg_Loss",
                             "auc_single", "auc_statistic",
                             "class0_auc_statistic", "class1_auc_statistic", "class2_auc_statistic",
                             "acc_single", "acc_statistic",
                             "Weighted precision", "Weighted recall", "Weighted f1-score",
                             "Macro precision", "Macro recall", "Macro f1-score"])  # 3. 构建列表头
def Save_Metrics(time1, time2, arch, state, epoch, learning_rate, batch_size, avg_loss,
                 auc_single, auc_statis, class0_auc_statis, class1_auc_statis, class2_auc_statis,
                 acc_single, acc_statis,
                 weighted_precision, weighted_recall, weighted_f1_score,
                 macro_precision, macro_recall, macro_f1_score):
    makedir('/home/test/Uterus_Dis_Cl/image/{}'.format(time1))
    makefile('/home/test/Uterus_Dis_Cl/image/{}/metrics.csv'.format(time1))

    line = [time2, arch, state, epoch, learning_rate, batch_size, avg_loss,
            auc_single, auc_statis, class0_auc_statis, class1_auc_statis, class2_auc_statis,
            acc_single, acc_statis,
            weighted_precision, weighted_recall, weighted_f1_score,
            macro_precision, macro_recall, macro_f1_score]
    f = open('/home/test/Uterus_Dis_Cl/image/{}/metrics.csv'.format(time1), 'a+', newline='')  # 创建文件对象
    csv_writer = csv.writer(f)  # 2. 基于文件对象构建 csv写入对象
    csv_writer.writerow(line)
    f.close()

# 保存最好的混淆矩阵
def Save_C_M(time1, txt_path, epoch, train_cnf_matrix_statis, train_cnf_matrix_single,
             test_cnf_matrix_statis, test_cnf_matrix_single
             ):
    delfile(txt_path)
    f = open(txt_path, 'a+')
    f.write('\r\n' + "epoch：")
    f.write('\r\n' + str(epoch))
    f.write('\r\n'+"训练集混淆矩阵：")
    f.write('\r\n'+"病例为主体：")
    f.write('\r\n'+str(train_cnf_matrix_statis))
    f.write('\r\n'+"图像为主体：")
    f.write('\r\n'+str(train_cnf_matrix_single))
    f.write('\r\n'+"测试集混淆矩阵：")
    f.write('\r\n'+"病例为主体：")
    f.write('\r\n'+str(test_cnf_matrix_statis))
    f.write('\r\n'+"图像为主体：")
    f.write('\r\n'+str(test_cnf_matrix_single))
    f.close()
# 计算前k类的准确率
def accuracy(output, target, topk=(1, 1)):
    """Computes the precision@k for the specified values of k"""
    output = output.data
    target = target.data

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # 求output数组中的最大值（即概率最大的分类）,返回值和索引
    pred = pred.t()  # 转置
    correct = pred.eq(target.expand_as(
        pred))  # b.expand_as(a)就是将b进行扩充，扩充到a的维度，.eq()函数比较两向量是否,两个向量的维度必须一致，如果相等，对应维度上的数为1，若果不相等则对应位置上的元素为0.

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# 生成auc计算所需数据并写入csv，包括id，病例id，真实标签，预测标签，预测概率
def auc_calculate_gen_csv(output, target, batch_idx, patient_id):
    print("----------------")
    f = open("/home/test/Uterus_Dis_Cl/AUC_data.csv", 'a+', newline='') # 创建文件对象
    csv_writer = csv.writer(f) # 2. 基于文件对象构建 csv写入对象
    if batch_idx == 0:
        csv_writer.writerow(["id","patient_id","labels","pred_labels","pred_scores"]) # 3. 构建列表头

    pred_score = output.numpy() 
    _, pred_label = output.topk(1, 1, True, True) # 概率大的作为标签
    pred_label = pred_label.numpy()
    patient_ids = patient_id.numpy()
    target = target.numpy()

    for i in range(len(pred_label)):
        id = batch_idx * len(pred_label) + i
        patient_id = int(patient_ids[i])
        labels = int(target[i])
        pred_labels = int(pred_label[i][0])
        pred_scores = list(pred_score[i])
        line = [id, patient_id, labels, pred_labels, pred_scores]
        csv_writer.writerow(line) 
    f.close()

# 生成auc计算所需数据并写入字典，包括id，病例id，真实标签，预测标签，预测概率
ids = list() #存储序号
patient_ids = list() #存储病例id
labels = list() # 存储真实标签
pred_labels = list() #存储预测标签
pred_scores = list() #存储预测概率
def auc_calculate_gen_csv(output, target, batch_idx, patient_id,batch_size):
    print("----------------")

    # if batch_idx == 0:

    pred_score = output.numpy() 
    _, pred_label = output.topk(1, 1, True, True) # 概率大的作为标签
    pred_label = pred_label.numpy()
    patient_id_ = patient_id.numpy()  # 原数据无法修改，故名称需不同
    target = target.numpy()

    for i in range(len(pred_label)):
        id = batch_idx * len(pred_label) + i
        ids.append(id)
        patient_ids.append(int(patient_id_[i]))
        labels.append(int(target[i]))
        pred_labels.append(int(pred_label[i][0]))
        pred_scores.append(list(pred_score[i]))
    if (batch_idx+1) == batch_size:
        data ={
            'ids':ids,
            'patient_ids':patient_ids,
            'labels':labels,
            'pred_labels':pred_labels,
            'pred_scores':pred_scores
        }
        df = pd.DataFrame(data)
        return df
    else:
        return 0

'''
TP: 实际为阳性,同时被检测为阳性。
FP: 实际为阴性,但被检测为阳性。
FN: 实际为阳性,但被检测为阴性。
TN: 实际为阴性,同时也被检测为阴性。
TPR(True Positive Rate,真阳性率): TPR = TP/(TP+FN) 预测对的正样本 占 全体正样本 的比例
TNR(True Negative Rate,真阴性率): TNR = TN/(FP+TN) 预测对的负样本 占 全体负样本 的比例,                注：就是“负类别”的召回率。
FPR(False Positive Rate,假阳性率): FPR = FP/(FP+TN) 预测错的正样本 占 全体负样本 的比例,                   也叫误识别率、虚警率。
FNR(False Negative Rate,假阴性率): FNR = FN/(TP+FN) 预测错的负样本 占 全体正样本 的比例,                   也叫拒识率。
灵敏度(Sensitivity,真阳性率/召回率): sensitive = TP/(TP+FN) = TP/P = recall 预测对的正样本 占 全体正样本 的比例,衡量了分类器对正例的识别能力,防止漏诊
特异度(Specificity,真阴性率): specificity = TN/(TN+FP) = TN/N, 预测对的负样本 占 全体负样本 的比例,衡量了分类器对负例的识别能力,防止误诊
准确率(Accuracy): Accuracy =(TP+TN)/(TP+FP+TN+FN),准确率在不均衡的样本集上度量效果很差
精确率(Precision,查准率/阳性预测值): Precision =TP/(TP+FP)  预测为正且实际为正的样本 占 预测为正的样本的比例
F1-score: F1 = 2Precision*Recall/(Precision+Recall),是召回率和精确率的调和平均数。
误报率(False discovery rate, FDR):FDR = FP/(FP+TP) = 1- Precision预测为正的样本中,实际为负的样本所占比例。
阴性预测值(Negative Predictive Value,NPV ):NPV = TN/(TN+FN),预测为负的样本中负样本的比例。

# 多分类 
average : {‘micro’, ‘macro’, ‘samples’, ‘weighted’} or None, default=’macro’  
Macro-average方法: 对分类器的评估指标求平均。该方法受样本量小的类别影响大。
Weighted-average方法: 对分类器的评估指标求加权平均,权重为该类别在总样本中的占比。该方法受样本量大的类别影响大。
Micro-average方法: 把每个类别的TP, FP, FN先相加之后,在根据分类的公式进行计算。
Samples-average方法: 计算评估数据中的每个样本的真实和预测类别的 metric （指标），并返回 (sample_weight-weighted) 加权平均

'''

# 二分类
def AUC_2(df, state, time1, epoch):
    def threshold(ytrue, ypred):
        fpr, tpr, thresholds = roc_curve(ytrue, ypred, pos_label=1, drop_intermediate=False)  # 标签为1的是正样本，其余的都是负样本
        y = tpr - fpr
        youden_index = np.argmax(y)
        optimal_threshold = thresholds[youden_index]
        point = [fpr[youden_index], tpr[youden_index]]
        roc_auc = auc(fpr, tpr)
        return optimal_threshold, point, fpr, tpr, roc_auc

    single_threshold, single_point, single_fpr, single_tpr, single = threshold(df['labels'], df['neg_preds'])
    df['single'] = (df['neg_preds'] >= single_threshold).astype(int)  # single_threshold判断概率是否大于阈值,大于的话(1),否则为(0)
    acc_single = (df['labels'] == df['single']).mean()  # 计算成功率

    # 生成以病人为主体的字典
    df = df.groupby('patient_ids')[['labels', 'neg_preds']].mean()
    df['labels'] = df['labels'].astype('int')
    statistic_threshold, statistic_point, statistic_fpr, statistic_tpr, statis = threshold(df['labels'],
                                                                                           df['neg_preds'])
    df['outputs'] = (df['neg_preds'] >= statistic_threshold).astype(int)  # statistic_threshold
    acc_statistic = (df['labels'] == df['outputs']).mean()

    # sensitive = TP/(TP+FN) = TP/P = recall，表示的是所有正例中被分对的比例，衡量了分类器对正例的识别能力,防止漏诊
    df_sensitivity = df.loc[df["labels"] == 1]
    statistic_sensitivity = (df_sensitivity['labels'] == df_sensitivity['outputs']).mean()
    # specificity = TN/(TN+FP) = TN/N，表示的是所有负例中被分对的比例，衡量了分类器对负例的识别能力，防止误诊
    df_specificity = df.loc[df["labels"] == 0]
    statistic_specificity = (df_specificity['labels'] == df_specificity['outputs']).mean()

    df_P = df.loc[df["labels"] == 1]  # 标签为阳性的部分字典
    df_N = df.loc[df["labels"] == 0]  # 标签为阴性的部分字典

    TP = (df_P['outputs'] == 1).sum()  # 真阳性
    TN = (df_N['outputs'] == 0).sum()  # 真阴性
    FP = (df_N['outputs'] == 1).sum()  # 假阳性
    FN = (df_P['outputs'] == 0).sum()  # 假阴性
    statistic_TPR = TP / (TP + FN)
    statistic_TNR = TN / (FP + TN)
    statistic_FPR = FP / (FP + TN)
    statistic_FNR = FN / (TP + FN)
    statistic_precision = TP / (TP + FP)  # precision= TP/（TP+FP），表示被分为正例的示例中实际为正例的比例
    # precision, recall, thresholds = precision_recall_curve(df['labels'], df['neg_preds']) # PR曲线
    f1_score = (2 * statistic_sensitivity * statistic_precision) / (
                statistic_sensitivity + statistic_precision)  # 假设Recall 与 Presision 的权重一样大， 求得的两个值的加权平均值
    print("病人为主体的真阳性：", TP, "真阴性：", TN, "假阳性：", FP, "假阴性：", FN)
    print("病人为主体的真阳性率：", statistic_TPR, "真阴性率：", statistic_TNR, "假阳性率：", statistic_FPR,
          "假阴性率：", statistic_FNR, "精确率", statistic_precision, "f1_score", f1_score)

    cnf_matrix = confusion_matrix(df['labels'], df['outputs'])  # 计算混淆矩阵
    print("病人为主体的混淆矩阵:")
    print(cnf_matrix)

    colors = ['red', 'yellow', 'blue']
    # 单个样本
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax[0].plot(single_fpr, single_tpr, colors[0], label='Resnet50' + ' ACC=%0.2f AUC = %0.2f' % (acc_single, single))
    ax[0].legend(loc='lower right', fontsize=12)
    ax[0].plot([0, 1], [0, 1], color='black', linestyle='dashed')
    ax[0].set_xlabel('Flase Positive Rate', fontsize=15, color='black')
    ax[0].set_ylabel('True Positive Rate', fontsize=15, color='black')
    ax[0].set_title('single')
    # 病人为主体
    ax[1].plot(statistic_fpr, statistic_tpr, colors[0],
               label='Resnet50' + ' ACC=%0.2f AUC = %0.2f' % (acc_statistic, statis))
    ax[1].legend(loc='lower right', fontsize=12)
    ax[1].plot([0, 1], [0, 1], color='black', linestyle='dashed')
    ax[1].set_xlabel('Flase Positive Rate', fontsize=15, color='black')
    ax[1].set_ylabel('True Positive Rate', fontsize=15, color='black')
    ax[1].set_title('patient')

    makedir('/home/test/Uterus_Dis_Cl/image/{}'.format(time1))
    makefile('/home/test/Uterus_Dis_Cl/image/{}/metrics.csv'.format(time1))

    time3 = time.strftime("%Y%m%d-%H%M", time.localtime())  # epoch记录写入csv和画图的时间

    plt.savefig('/home/test/Uterus_Dis_Cl/image/{}/{}_epoch_{}_{}的roc曲线.png'.format(time1, state, epoch,
                                                                                             time.strftime(
                                                                                                 "%Y%m%d-%H%M",
                                                                                                 time.localtime())))

    line = [time3, state, epoch, statis, acc_statistic, single, acc_single]
    f = open('/home/test/Uterus_Dis_Cl/image/{}/metrics.csv'.format(time1), 'a+', newline='')  # 创建文件对象
    csv_writer = csv.writer(f)  # 2. 基于文件对象构建 csv写入对象
    csv_writer.writerow(line)
    f.close()

    # acc_single(单个样本的成功率), acc_statistic(病例为主体的成功率),
    # single(单个样本的auc面积), statis(病例为主体的auc面积),
    # single_threshold(单个样本的阈值), statistic_threshold(病例为主体的auc面积),
    # single_fpr, single_tpr, single_point, (单个样本)
    # statistic_fpr, statistic_tpr, statistic_point, (病例为主体)
    # statistic_sensitivity(正例的识别能力), statistic_specificity(负例的识别能力)
    return acc_single, acc_statistic, single, statis, single_threshold, statistic_threshold, \
        single_fpr, single_tpr, single_point, statistic_fpr, statistic_tpr, statistic_point, \
        statistic_sensitivity, statistic_specificity





