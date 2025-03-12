'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function


import os
import shutil
from random import random

import numpy as np
import torch
import torch.nn.parallel

from utils import train, test, Save_Metrics, Save_C_M
from SLA import train_sla_sd, test_sla_sd, train_sla, test_sla, Save_Metrics_Sla, Save_C_M_Sla

#全面取消证书认证
import ssl

from utils.train import FASD_train

ssl._create_default_https_context = ssl._create_unverified_context

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def save_checkpoint(state, is_best, i, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    if i != -1:
        checkpoint = os.path.join(checkpoint, "flod_{}".format(i))
        makedir(checkpoint)
    else:
        print("i", i)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


# baseline的train、test、valid
def base_train_test(i, trainloader, testloader, model, criterion, optimizer, epoch, use_cuda, time1, time2, lr, arch, train_batch, test_batch,
                    checkpoint, sum_epoch, best_acc,pre_data, pre_single_out):
    # 训练集
    train_avg_loss, train_auc_single, train_auc_statis, train_class0_auc_statis, train_class1_auc_statis, train_class2_auc_statis, \
        train_acc_single, train_acc_statis, train_cnf_matrix_statis, train_cnf_matrix_single,pre_data, pre_single_out \
        = train(trainloader, model, criterion, optimizer, epoch, use_cuda,pre_data, pre_single_out)
    Save_Metrics(time1, time2, arch, 'train_fold_{}'.format(i), epoch, lr, train_batch, train_avg_loss,
                 train_auc_single, train_auc_statis, train_class0_auc_statis, train_class1_auc_statis,
                 train_class2_auc_statis)


    # 测试集
    test_avg_loss, test_auc_single, test_auc_statis, test_class0_auc_statis, test_class1_auc_statis, test_class2_auc_statis, \
        test_acc_single, test_acc_statis, test_cnf_matrix_statis, test_cnf_matrix_single,df \
        = test(testloader, model, criterion, epoch, use_cuda)
    Save_Metrics(time1, time2, arch, 'test_fold_{}'.format(i), epoch, lr, test_batch, test_avg_loss,
                 test_auc_single, test_auc_statis, test_class0_auc_statis, test_class1_auc_statis,
                 test_class2_auc_statis, test_acc_single)



    if train_acc_statis >= 0.6:
        print("已收敛")
        test_acc = test_acc_statis
        is_best = test_acc > best_acc
        if is_best==True :
            # 路径
            txt_path = '/home/test/Uterus_Dis_Cl/image/{}/fold_{}_c_m_acc_best.txt'.format(time1, i)
            #txt_path = 'C:/Users/10433/Desktop/Uterus_Dis_Cl/image/{}/fold_{}_c_m_acc_best.txt'.format(time1, i)
            Save_C_M(time1, txt_path, epoch, train_cnf_matrix_statis, train_cnf_matrix_single, test_cnf_matrix_statis, test_cnf_matrix_single)
            best_acc = max(test_acc, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, i, checkpoint=checkpoint)
        print(test_acc)
        print(best_acc)
    else:
        print("未收敛")
        test_acc = test_acc_statis
        best_acc = max(test_acc, best_acc)
        print(test_acc)
        print(best_acc)
        best_acc = 0
        print(best_acc)
    if (epoch + 1) == sum_epoch:
        #路径
        txt_path = '/home/test/Uterus_Dis_Cl/image/{}/fold_{}_c_m_last_epoch.txt'.format(time1,i)
        #txt_path = 'C:/Users/10433/Desktop/Uterus_Dis_Cl/image/{}/fold_{}_c_m_last_epoch.txt'.format(time1,i)
        Save_C_M(time1, txt_path, epoch, train_cnf_matrix_statis, train_cnf_matrix_single, test_cnf_matrix_statis, test_cnf_matrix_single)
    return best_acc
def FASD_train_test(i, trainloader, testloader, model, criterion, optimizer, epoch, use_cuda, time1, time2, lr, arch, train_batch, test_batch,
                    checkpoint, sum_epoch, best_acc,pre_data, pre_single_out):
    # 训练集
    train_avg_loss, train_auc_single, train_auc_statis, train_class0_auc_statis, train_class1_auc_statis, train_class2_auc_statis, \
        train_acc_single, train_acc_statis, train_cnf_matrix_statis, train_cnf_matrix_single,pre_data, pre_single_out \
        = FASD_train(trainloader, model, criterion, optimizer, epoch, use_cuda,pre_data, pre_single_out)
    Save_Metrics(time1, time2, arch, 'train_fold_{}'.format(i), epoch, lr, train_batch, train_avg_loss,
                 train_auc_single, train_auc_statis, train_class0_auc_statis, train_class1_auc_statis,
                 train_class2_auc_statis)


    # 测试集
    test_avg_loss, test_auc_single, test_auc_statis, test_class0_auc_statis, test_class1_auc_statis, test_class2_auc_statis, \
        test_acc_single, test_acc_statis, test_cnf_matrix_statis, test_cnf_matrix_single,df \
        = test(testloader, model, criterion, epoch, use_cuda)
    Save_Metrics(time1, time2, arch, 'test_fold_{}'.format(i), epoch, lr, test_batch, test_avg_loss,
                 test_auc_single, test_auc_statis,
                 test_class2_auc_statis, test_acc_single, test_acc_statis,0,0,0,0,0,0)



    if train_acc_statis >= 0.6:
        print("已收敛")
        test_acc = test_acc_statis
        is_best = test_acc > best_acc
        if is_best==True :
            # 路径
            txt_path = '/home/test/Uterus_Dis_Cl/image/{}/fold_{}_c_m_acc_best.txt'.format(time1, i)
            #txt_path = 'C:/Users/10433/Desktop/Uterus_Dis_Cl/image/{}/fold_{}_c_m_acc_best.txt'.format(time1, i)
            Save_C_M(time1, txt_path, epoch, train_cnf_matrix_statis, train_cnf_matrix_single, test_cnf_matrix_statis, test_cnf_matrix_single)
            best_acc = max(test_acc, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, i, checkpoint=checkpoint)
        print(test_acc)
        print(best_acc)
    else:
        print("未收敛")
        test_acc = test_acc_statis
        best_acc = max(test_acc, best_acc)
        print(test_acc)
        print(best_acc)
        best_acc = 0
        print(best_acc)
    if (epoch + 1) == sum_epoch:
        #路径
        txt_path = '/home/test/Uterus_Dis_Cl/image/{}/fold_{}_c_m_last_epoch.txt'.format(time1,i)
        #txt_path = 'C:/Users/10433/Desktop/Uterus_Dis_Cl/image/{}/fold_{}_c_m_last_epoch.txt'.format(time1,i)
        Save_C_M(time1, txt_path, epoch, train_cnf_matrix_statis, train_cnf_matrix_single, test_cnf_matrix_statis, test_cnf_matrix_single)
    return best_acc
def sla_train_test(i, trainloader, testloader, model, optimizer, epoch, use_cuda, time1, time2, lr, arch, train_batch, test_batch,
                    checkpoint, sum_epoch, transform, best_acc):
    # 训练集

    train_loss, \
    train_agg_auc_single, train_agg_auc_statis, train_agg_class0_auc_statis, train_agg_class1_auc_statis, train_agg_class2_auc_statis, \
    train_agg_acc_single, train_agg_acc_statis, train_agg_cnf_matrix_statis, train_agg_cnf_matrix_single \
        = train_sla(trainloader, model, transform, optimizer, epoch, use_cuda)

    Save_Metrics(time1, time2, arch, 'train_fold_{}'.format(i), epoch, lr, train_batch, train_loss,
                 train_agg_auc_single, train_agg_auc_statis,  # agg torch.Size([64, 3])
                 train_agg_class0_auc_statis, train_agg_class1_auc_statis, train_agg_class2_auc_statis,
                 train_agg_acc_single, train_agg_acc_statis,0,0,0,0,0,0)

    # 测试集
    test_loss, \
    test_agg_auc_single, test_agg_auc_statis, test_agg_class0_auc_statis, test_agg_class1_auc_statis, test_agg_class2_auc_statis, \
    test_agg_acc_single, test_agg_acc_statis, test_agg_cnf_matrix_statis, test_agg_cnf_matrix_single \
        = test_sla(testloader, model, transform, epoch, use_cuda)

    Save_Metrics(time1, time2, arch, 'test_fold_{}'.format(i), epoch, lr, test_batch, train_loss,
                 test_agg_auc_single, test_agg_auc_statis,  # agg torch.Size([64, 3])
                 test_agg_class0_auc_statis, test_agg_class1_auc_statis, test_agg_class2_auc_statis,
                 test_agg_acc_single, test_agg_acc_statis,0,0,0,0,0,0)


    print("train_agg_acc_statis", train_agg_acc_statis)
    if train_agg_acc_statis >= 0.9:
        print("已收敛")
        test_acc = test_agg_acc_statis
        is_best = test_acc > best_acc
        if is_best==True :
            # 路径
            txt_path = '/home/test/Uterus_Dis_Cl/image/{}/fold_{}_c_m_acc_best.txt'.format(time1, i)
            #txt_path = 'C:/Users/10433/Desktop/Uterus_Dis_Cl/image/{}/fold_{}_c_m_acc_best.txt'.format(time1, i)
            Save_C_M(time1, txt_path, epoch,
                     train_agg_cnf_matrix_statis, train_agg_cnf_matrix_single,
                     test_agg_cnf_matrix_statis, test_agg_cnf_matrix_single)
            best_acc = max(test_acc, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, i, checkpoint=checkpoint)
        print(test_acc)
        print(best_acc)
    else:
        print("未收敛")
        test_acc = test_agg_acc_statis
        best_acc = max(test_acc, best_acc)
        print(test_acc)
        print(best_acc)
        best_acc = 0
        print(best_acc)
    if (epoch + 1) == sum_epoch:
        # 路径
        txt_path = '/home/test/Uterus_Dis_Cl/image/{}/fold_{}_c_m_last_epoch.txt'.format(time1,i)
        #txt_path = 'C:/Users/10433/Desktop/Uterus_Dis_Cl/Uterus_Dis_Cl/image/{}/fold_{}_c_m_last_epoch.txt'.format(time1,i)
        Save_C_M(time1, txt_path, epoch,
                 train_agg_cnf_matrix_statis, train_agg_cnf_matrix_single,
                 test_agg_cnf_matrix_statis, test_agg_cnf_matrix_single)

    return best_acc

def sla_sd_train_test(criterion,args,i, trainloader, testloader, model, optimizer, epoch, use_cuda, time1, time2, lr, arch,
                      train_batch, test_batch, checkpoint, sum_epoch, transform, T, best_agg_acc,best_sing_acc, pre_data, pre_joint_out, pre_single_out):
    # 训练集
    train_loss, \
    train_sing_auc_single, train_sing_auc_statis, train_sing_class0_auc_statis, train_sing_class1_auc_statis, train_sing_class2_auc_statis, \
    train_sing_acc_single, train_sing_acc_statis, train_sing_cnf_matrix_statis, train_sing_cnf_matrix_single, \
    train_agg_auc_single, train_agg_auc_statis, train_agg_class0_auc_statis, train_agg_class1_auc_statis, train_agg_class2_auc_statis, \
    train_agg_acc_single, train_agg_acc_statis, train_agg_cnf_matrix_statis, train_agg_cnf_matrix_single, pre_data, pre_joint_out, pre_single_out \
        = train_sla_sd(trainloader, model, transform, optimizer, T, epoch, use_cuda,args,criterion, pre_data, pre_joint_out, pre_single_out)


    Save_Metrics_Sla(time1, time2, arch, 'train_fold_{}'.format(i), epoch, lr, train_batch, train_loss,
                     train_sing_auc_single, train_sing_auc_statis, # singel  # torch.Size([256, 3])
                     train_sing_class0_auc_statis, train_sing_class1_auc_statis, train_sing_class2_auc_statis,
                     train_sing_acc_single, train_sing_acc_statis,
                     train_agg_auc_single, train_agg_auc_statis,  # agg torch.Size([64, 3])
                     train_agg_class0_auc_statis, train_agg_class1_auc_statis, train_agg_class2_auc_statis,
                     train_agg_acc_single, train_agg_acc_statis)

    # 测试集
    test_loss, \
    test_sing_auc_single, test_sing_auc_statis, test_sing_class0_auc_statis, test_sing_class1_auc_statis, test_sing_class2_auc_statis, \
    test_sing_acc_single, test_sing_acc_statis, test_sing_cnf_matrix_statis, test_sing_cnf_matrix_single, \
    test_agg_auc_single, test_agg_auc_statis, test_agg_class0_auc_statis, test_agg_class1_auc_statis, test_agg_class2_auc_statis, \
    test_agg_acc_single, test_agg_acc_statis, test_agg_cnf_matrix_statis, test_agg_cnf_matrix_single, df \
        = test_sla_sd(testloader, model, transform, T, epoch, use_cuda,args,criterion)

    Save_Metrics_Sla(time1, time2, arch, 'test_fold_{}'.format(i), epoch, lr, test_batch, test_loss,
                     test_sing_auc_single, test_sing_auc_statis,  # singel  # torch.Size([256, 3])
                     test_sing_class0_auc_statis, test_sing_class1_auc_statis, test_sing_class2_auc_statis,
                     test_sing_acc_single, test_sing_acc_statis,
                     test_agg_auc_single, test_agg_auc_statis,  # agg torch.Size([64, 3])
                     test_agg_class0_auc_statis, test_agg_class1_auc_statis, test_agg_class2_auc_statis,
                     test_agg_acc_single, test_agg_acc_statis)


    print("train_sing_acc_statis",train_sing_acc_statis)
    print("train_agg_acc_statis", train_agg_acc_statis)
    if train_agg_acc_statis >= 0.6 or test_sing_acc_statis>= 0.6:
        print("已收敛")
        test_agg_acc = test_agg_acc_statis
        test_sing_acc = test_agg_acc_statis
        is_agg_best = test_agg_acc > best_agg_acc
        is_sing_best = test_sing_acc > best_sing_acc
        if is_agg_best==True :
            print("is_agg_best")
            # 路径
            txt_path = '/home/test/Uterus_Dis_Cl/image/{}/fold_{}_c_m_acc_agg_best.txt'.format(time1, i)
            #txt_path = 'C:/Users/10433/Desktop/Uterus_Dis_Cl/image/{}/fold_{}_c_m_acc_best.txt'.format(time1, i)
            Save_C_M_Sla(time1, txt_path, epoch,
                         train_sing_cnf_matrix_statis, train_sing_cnf_matrix_single,
                         train_agg_cnf_matrix_statis, train_agg_cnf_matrix_single,
                         test_sing_cnf_matrix_statis, test_sing_cnf_matrix_single,
                         test_agg_cnf_matrix_statis, test_agg_cnf_matrix_single)
            best_agg_acc = max(test_agg_acc, best_agg_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_agg_acc,
                'best_acc': best_agg_acc,
                'optimizer': optimizer.state_dict(),
            }, is_agg_best, i, checkpoint=checkpoint)
        print("test_agg_acc:",test_agg_acc)
        print("best_agg_acc:",best_agg_acc)

        if is_sing_best == True:
            # 路径
            print("is_sing_best")
            txt_path = '/home/test/Uterus_Dis_Cl/image/{}/fold_{}_c_m_acc_sing_best.txt'.format(time1, i)
            # txt_path = 'C:/Users/10433/Desktop/Uterus_Dis_Cl/image/{}/fold_{}_c_m_acc_best.txt'.format(time1, i)
            Save_C_M_Sla(time1, txt_path, epoch,
                         train_sing_cnf_matrix_statis, train_sing_cnf_matrix_single,
                         train_agg_cnf_matrix_statis, train_agg_cnf_matrix_single,
                         test_sing_cnf_matrix_statis, test_sing_cnf_matrix_single,
                         test_agg_cnf_matrix_statis, test_agg_cnf_matrix_single)
            best_sing_acc = max(test_sing_acc, best_sing_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_sing_acc,
                'best_acc': best_sing_acc,
                'optimizer': optimizer.state_dict(),
            }, is_agg_best, i, checkpoint=checkpoint)

        print("test_sing_acc:",test_sing_acc)
        print("best_sing_acc:",best_sing_acc)

    else:
        print("未收敛")
        test_agg_acc = test_agg_acc_statis
        test_sing_acc = test_agg_acc_statis
        best_agg_acc = max(test_agg_acc, best_agg_acc)
        best_sing_acc = max(test_sing_acc, best_sing_acc)
        print("test_agg_acc:",test_agg_acc)
        print("best_agg_acc:",best_agg_acc)
        print("test_sing_acc:",test_sing_acc)
        print("best_sing_acc:",best_sing_acc)
        # best_acc = 0
        # print(best_acc)
    if (epoch + 1) == sum_epoch:
        # 路径
        txt_path = '/home/test/Uterus_Dis_Cl/image/{}/fold_{}_c_m_last_epoch.txt'.format(time1,i)
        #txt_path = '/C:/Users/10433/Desktop/Uterus_Dis_Cl/image/{}/fold_{}_c_m_last_epoch.txt'.format(time1,i)
        Save_C_M_Sla(time1, txt_path, epoch,
                     train_sing_cnf_matrix_statis, train_sing_cnf_matrix_single,
                     train_agg_cnf_matrix_statis, train_agg_cnf_matrix_single,
                     test_sing_cnf_matrix_statis, test_sing_cnf_matrix_single,
                     test_agg_cnf_matrix_statis, test_agg_cnf_matrix_single)

    return best_agg_acc,best_sing_acc, pre_data, pre_joint_out, pre_single_out

