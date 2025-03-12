'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function
import math
import argparse
import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import models.Uterus as models
import datetime

from utils import Logger, MyDataset, savefig, split_dataset, gen_csv, split_kflod, gen_kflod_csv, gen_patients_csv,split_mix, train, test,\
    Save_Metrics, Save_C_M
# draw_CAM, VITAttentionRollout, VITAttentionGradRollout
import SLA.augmentations as augmentations
from trainer import base_train_test, sla_sd_train_test, sla_train_test, FASD_train_test
from SLA import rotation, check_model, train_sla_sd, test_sla_sd, Save_Metrics_Sla, Save_C_M_Sla, check_model_FASD

import torch
import torch.nn as nn
import torch.optim as optim
import optuna
#全面取消证书认证
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def gd():
    # 固定随机种子
    X=755
    random.seed(X)
    torch.manual_seed(X)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(X)
        torch.cuda.manual_seed_all(X)  # 如果使用多GPU
    np.random.seed(X)
# 获取模型
def Model(args, num_classes, m):
    print("==> creating model '{}'".format(args.arch))

    if args.mode == 'DLB':
        if args.arch.endswith('resnext'):
            model = models.__dict__[args.arch](
                cardinality=args.cardinality,
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop,
            )
        elif args.arch.endswith('alexnet'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
            )
        elif args.arch.endswith('densenet'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                growthRate=args.growthRate,
                compressionRate=args.compressionRate,
                dropRate=args.drop,
            )
        elif args.arch.endswith('vgg'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
            )
        elif args.arch.endswith('wrn'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=28,
                widen_factor=10,
                dropRate=0.3,
            )
        elif args.arch.endswith('swin_v2'):  # swin

            model = models.__dict__[args.arch](

                pretrained=False,
                scriptable=None,
                exportable=None,

            )
        elif args.arch.endswith('resnet'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
                block_name=args.block_name,
            )
        elif args.arch.endswith('resnet50'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
                layers=args.layer,
                block_name=args.block_name,
            )
        elif args.arch.endswith('resnet18'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
            )
        elif args.arch.endswith('davit'):#双注意力
            model = models.__dict__[args.arch](
                #在Uterus_Dis_Cl/models/Uterus/davit.py控制参数
                #pretrained=True,
                # 这两个参数不会影响到模型在数据集上的准确率或其他
                # 性能指标，而是用于控制模型的部署和集成方式。
                scriptable=None,
                exportable=None,
                #drop_path_rate=args.drop
               # no_jit=None,
            )
        elif args.arch.endswith('vit'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
            )
        elif args.arch.endswith('vit_pre'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
            )
        elif args.arch.startswith('cmt'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
                pretrained=True,
                drop_rate=0.0,
                drop_path_rate=0.1,
                # drop_block_rate=args.drop_block,
                img_size=224,
            )
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, 3)
        elif args.arch.endswith('CmtTi'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
                input_resolution=(224, 224),
                qkv_bias=True,
                ape=False,
                rpe=True,
                pe_nd=False,
                drop_path_rate=0.1,
            )
        elif args.arch.endswith('alexnet_pre'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
            )
        elif args.arch.endswith('densenet_pre'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
            )
        elif args.arch.endswith('vgg_pre'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
            )
        elif args.arch.endswith('resnet50_pre'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
            )
        elif args.arch.endswith('resnet18_pre'):
            model = models.__dict__[args.arch](
                num_classes=num_classes,
            )

            model.num_features = 3
        elif args.arch.endswith('convnext_base'):
            model = models.__dict__[args.arch](
                pretrained=True,
                drop_path_rate=0.1,
                layer_scale_init_value=1e-6,
                head_init_scale=0.001,
            )
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, 3)
        elif args.arch.endswith('twins'):
            model = models.__dict__[args.arch](
            )
            model.num_features = 100

    elif args.mode == 'sla':
        model = check_model(args.arch, num_classes * m)
    elif args.mode == 'our':
        model = check_model(args.arch, num_classes * m, num_classes)
    elif args.mode == 'FASD':
        model = check_model_FASD(args.arch, num_classes)

    # elif args.mode == 'DLB':
    #     model = models.__dict__[args.arch](num_classes=num_classes)
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    # 分配到特定单卡/多卡
    #model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    # 用多GPU训练，并寻找最高配置
    cudnn.benchmark = True
    if args.aug is not None:
        model.num_transformations = m

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    return model
def getSeed():
    # 查看 Python 的随机种子
    random_state = random.getstate()
    print("Python random seed:", random_state[1][0])

    # 查看 NumPy 的随机种子
    numpy_state = np.random.get_state()
    print("NumPy random seed:", numpy_state[1][0])

    # 查看 PyTorch 的随机种子
    torch_seed = torch.seed()
    print("PyTorch random seed:", torch_seed)
    # # 设置 CUDA 设备的随机种子
    # torch.cuda.manual_seed(42)
    # 查看 CUDA 设备的随机种子
    cuda_seed = torch.cuda.initial_seed()
    print("CUDA random seed:", cuda_seed)

best_acc = 0  # best test accuracy
def main(args):
    time1 = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')  # 整个程序开始的时间
    global best_acc
    getSeed()
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    state = {k: v for k, v in args._get_kwargs()} #将paser中的所有的参数及其默认值对应

    # Data准备
    assert args.dataset == 'Uterus', 'Dataset can only be Uterus.' # Validate dataset
    print('==> Preparing dataset %s' % args.dataset)
    # 判断数据集是几分类
    if args.dataset == 'Uterus':
        num_classes = 3
    else:
        num_classes = 100

    # Transformation
    if args.aug is not None:
        transform, m = augmentations.__dict__[args.aug]()
    else:
        m = 0

    # 交叉验证
    if args.kflod_num:
        #getCsv_BUSI 在utils
        # 数据集准备
            # gen_csv(os.path.join(args.dataset_path, "v7_v10", "all_images.csv"), os.path.join(args.dataset_path))
            # gen_patients_csv(os.path.join(args.dataset_path, "v7_v10", "all_patients.csv"), os.path.join(args.dataset_path))
            # split_kflod(os.path.join(args.dataset_path, "v7_v10"), args.kflod_num)
            # gen_kflod_csv(os.path.join(args.dataset_path, "v7_v10"))

        #内膜
        #dataset_path= "/home/test/Uterus_Dis_Cl/10all_jpg/"
        #肾结石
        #dataset_path= "/home/test/Uterus_Dis_Cl/Kidney_stone"#有问题 二分类先不考虑
        #甲状腺
        dataset_path= "/home/test/Uterus_Dis_Cl/dataset_thyroid/train/"
        # 皮肤
        #dataset_path = "/home/test/Uterus_Dis_Cl/Dermatologic_Ultrasound/"
        #Dataset_BUSI_with_GT
        #dataset_path= "/home/test/Uterus_Dis_Cl/Dataset_BUSI_with_GT/"
        train_loader = []
        test_loader = []

        for i in range(1, args.kflod_num+1):
            #kflod_augmented{}
            train_csv_path = os.path.join(dataset_path, "v7_v10", 'kflod_{}_train.csv').format(i)#kflod_augmented
            train_set = MyDataset(csv_path=train_csv_path, state="train")
            train_loader.append(DataLoader(dataset=train_set, batch_size=args.train_batch, shuffle=True, num_workers=args.workers))

            test_csv_path = os.path.join(dataset_path, "v7_v10", 'kflod_{}_test.csv').format(i)
            test_set = MyDataset(csv_path=test_csv_path, state="test")
            test_loader.append(DataLoader(dataset=test_set, batch_size=args.test_batch, shuffle=False, num_workers=args.workers))

        # train test
        for i in range(args.kflod_num):
            model = Model(args, num_classes, m)
            pre_data, pre_joint_out, pre_single_out= None, None, None
            # 损失函数、优化器
            criterion = nn.CrossEntropyLoss()
            # criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([4, 2, 1])).float())
            criterion.cuda()
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            best_sing_acc = 0
            best_agg_acc = 0
            start_epoch = args.start_epoch
            state['lr'] = args.lr
            for epoch in range(start_epoch, args.epochs):
                print("------------------------------------", i, "------------------------------------")
                time2 = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')  # 保存每个epoch开始的时间
                #adjust_learning_rate(state, optimizer, epoch)
                adjust_cosine_decay_learning_rate(state, optimizer, epoch,state['lr'],args.epochs)
                print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
                if args.mode == 'DLB':
                    test_best_agg_acc = base_train_test(i, train_loader[i], test_loader[i], model, criterion, optimizer, epoch, use_cuda, time1, time2, state['lr'],
                                                    args.arch, args.train_batch, args.test_batch, args.checkpoint, args.epochs, best_sing_acc,pre_data,pre_single_out)
                #elif args.mode == 'sla+sd':
                elif args.mode == 'our':
                    test_best_agg_acc,test_best_sing_acc, pre_data, pre_joint_out, pre_single_out = sla_sd_train_test(criterion,args,i, train_loader[i], test_loader[i], model, optimizer, epoch, use_cuda, time1, time2, state['lr'],
                                                      args.arch, args.train_batch, args.test_batch,
                                                      args.checkpoint, args.epochs, transform, args.T, best_sing_acc,best_agg_acc, pre_data, pre_joint_out, pre_single_out)
                elif args.mode == 'sla':

                    test_best_agg_acc = sla_train_test(i, train_loader[i], test_loader[i], model, optimizer, epoch, use_cuda, time1, time2,
                                                      state['lr'], args.arch, args.train_batch, args.test_batch,
                                                      args.checkpoint, args.epochs, transform, best_acc)
                elif args.mode == 'FASD':
                    test_best_agg_acc = FASD_train_test(i, train_loader[i], test_loader[i], model, criterion, optimizer, epoch, use_cuda, time1, time2, state['lr'],
                                                    args.arch, args.train_batch, args.test_batch, args.checkpoint, args.epochs, best_sing_acc,pre_data,pre_single_out)
                else:
                    pass

                try:
                    best_sing_acc = test_best_sing_acc
                except NameError:
                    print("test_best_sing_acc is not defined")

                try:
                    best_sing_acc = test_best_agg_acc
                except NameError:
                    print("test_best_agg_acc is not defined")


    # 训练 测试
    else:
        exit()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)
    return best_acc
# #=============加入优化
# import numpy as np
#
#
# best_acc = 0  # best test accuracy
# def main(args):
#     time1 = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')  # 整个程序开始的时间
#     global best_acc
#     getSeed()
#     start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
#     state = {k: v for k, v in args._get_kwargs()} #将paser中的所有的参数及其默认值对应
#
#     # Data准备
#     assert args.dataset == 'Uterus', 'Dataset can only be Uterus.' # Validate dataset
#     print('==> Preparing dataset %s' % args.dataset)
#     # 判断数据集是几分类
#     if args.dataset == 'Uterus':
#         num_classes = 3
#     else:
#         num_classes = 100
#
#     # Transformation
#     if args.aug is not None:
#         transform, m = augmentations.__dict__[args.aug]()
#     else:
#         m = 0
#
#     # 交叉验证
#     if args.kflod_num:
#         #getCsv_BUSI 在utils
#
#         # 数据集准备
#         # gen_csv(os.path.join(args.dataset_path, "v7_v10", "all_images.csv"), os.path.join(args.dataset_path))
#         # gen_patients_csv(os.path.join(args.dataset_path, "v7_v10", "all_patients.csv"), os.path.join(args.dataset_path))
#         # split_kflod(os.path.join(args.dataset_path, "v7_v10"), args.kflod_num)
#         # gen_kflod_csv(os.path.join(args.dataset_path, "v7_v10"))
#
#         #内膜
#         dataset_path= "/home/lvxinpeng/uterus/Uterus_Dis_Cl/10all_jpg/"
#
#         #Dataset_BUSI_with_GT
#         #dataset_path= "/home/lvxinpeng/uterus/Uterus_Dis_Cl/Dataset_BUSI_with_GT/"
#         train_loader = []
#         test_loader = []
#
#         for i in range(1, args.kflod_num+1):
#             #kflod_augmented{}
#             train_csv_path = os.path.join(dataset_path, "v7_v10", 'kflod_augmented{}_train.csv').format(i)#kflod_augmented
#             train_set = MyDataset(csv_path=train_csv_path, state="train")
#             train_loader.append(DataLoader(dataset=train_set, batch_size=args.train_batch, shuffle=True, num_workers=args.workers))
#
#             test_csv_path = os.path.join(dataset_path, "v7_v10", 'kflod_{}_test.csv').format(i)
#             test_set = MyDataset(csv_path=test_csv_path, state="test")
#             test_loader.append(DataLoader(dataset=test_set, batch_size=args.test_batch, shuffle=False, num_workers=args.workers))
#
#         # train test
#
#         model = Model(args, num_classes, m)
#         pre_data, pre_joint_out, pre_single_out = None, None, None  # DBL
#         # 损失函数、优化器
#         criterion = nn.CrossEntropyLoss()
#         # criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([4, 2, 1])).float())
#         criterion.cuda()
#         optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
#         best_sing_acc = 0
#         best_agg_acc = 0
#         start_epoch = args.start_epoch
#         state['lr'] = args.lr
#         i=1
#         for epoch in range(start_epoch, args.epochs):
#             print("------------------------------------", i, "------------------------------------")
#             time2 = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')  # 保存每个epoch开始的时间
#             # adjust_learning_rate(state, optimizer, epoch)
#             adjust_cosine_decay_learning_rate(state, optimizer, epoch, state['lr'], args.epochs)
#             print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
#             if args.mode == 'baseline':
#                 test_best_agg_acc = base_train_test(i, train_loader[i], test_loader[i], model, criterion, optimizer,
#                                                     epoch, use_cuda, time1, time2, state['lr'],
#                                                     args.arch, args.train_batch, args.test_batch, args.checkpoint,
#                                                     args.epochs, best_acc)
#             elif args.mode == 'sla+sd':
#                 test_best_agg_acc, test_best_sing_acc, pre_data, pre_joint_out, pre_single_out = sla_sd_train_test(
#                     criterion, args, i, train_loader[i], test_loader[i], model, optimizer, epoch, use_cuda, time1,
#                     time2, state['lr'],
#                     args.arch, args.train_batch, args.test_batch,
#                     args.checkpoint, args.epochs, transform, args.T, best_sing_acc, best_agg_acc, pre_data,
#                     pre_joint_out, pre_single_out)
#             elif args.mode == 'sla':
#                 test_best_agg_acc = sla_train_test(i, train_loader[i], test_loader[i], model, optimizer, epoch,
#                                                    use_cuda, time1, time2,
#                                                    state['lr'], args.arch, args.train_batch, args.test_batch,
#                                                    args.checkpoint, args.epochs, transform, best_acc)
#
#             else:
#                 pass
#
#             best_sing_acc = test_best_sing_acc
#             best_agg_acc = test_best_agg_acc
#
#     # 训练 测试
#     else:
#         exit()
#
#     savefig(os.path.join(args.checkpoint, 'log.eps'))
#
#     print('Best acc:')
#     print(best_sing_acc)
#     return best_sing_acc
#
# def train_model_with_params(solution,args):
#     """
#     模型训练过程的封装。
#
#     参数：
#     - solution: 模型的参数。
#
#     返回：
#     - accuracy: 模型的准确率。
#     """
#
#     # 将向量转为具体参数
#     #args.lr = solution[0]
#     # args.momentum = solution[1]
#     # args.weight_decay = solution[2]
#     # args.gamma=solution[3]
#     # args.drop=solution[4]
#     args.T=solution[5]
#     args.ahadbl=solution[6]
#     args.ahasla=solution[7]
#     # 例如，这里可以是神经网络的训练过程
#     # 假设我们有一个函数 train_neural_network(learning_rate, regularization) 返回模型准确率
#     accuracy = main(args)
#
#     return accuracy
#
#
# def artemisinin_optimization(pop_size, dim, bounds, num_generations, num_parents, mutation_rate,agrs,param_keys):
#     """
#     使用Artemisinin optimization算法优化模型参数。
#
#     参数：
#     - pop_size: 种群大小。
#     - dim: 解决方案的维度。
#     - bounds: 参数的取值范围，字典形式。
#     - num_generations: 迭代次数。
#     - num_parents: 选择的父代数量。
#     - mutation_rate: 变异率。
#
#     返回：
#     - best_solution: 最优解。
#     - best_accuracy: 最优解的准确率。
#     """
#
#     # 初始化种群
#     def initialize_population(pop_size, dim, bounds):
#         population = []
#         for _ in range(pop_size):
#             individual = [np.random.uniform(bounds[param][0], bounds[param][1]) for param in param_keys]
#             population.append(individual)
#         return np.array(population)
#
#     # 选择
#     def selection(population, fitness_values, num_parents):
#         selected_indices = np.argsort(fitness_values)[-num_parents:]  # 选择最高的准确率
#         return population[selected_indices]
#
#     # 交叉
#     def crossover(parents, num_offspring):
#         offspring = []
#         for _ in range(num_offspring):
#             parent1, parent2 = parents[np.random.choice(parents.shape[0], 2, replace=False)]
#             crossover_point = np.random.randint(1, len(parent1))
#             child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
#             offspring.append(child)
#         return np.array(offspring)
#
#     # 变异
#     def mutation(offspring, mutation_rate, bounds):
#         for individual in offspring:
#             if np.random.rand() < mutation_rate:
#                 mutation_point = np.random.randint(len(individual))
#                 param = param_keys[mutation_point]
#                 individual[mutation_point] = np.random.uniform(bounds[param][0], bounds[param][1])
#         return offspring
#
#     # 初始化种群
#     population = initialize_population(pop_size, dim, bounds)
#
#     best_solution = None
#     best_accuracy = float('-inf')
#
#     # 打开文件以记录参数和准确率
#     with open('optimization_log.txt', 'w') as f:
#         # 开始训练过程
#         for generation in range(num_generations):
#             # 评估适应度，通过训练模型获得适应度值
#             fitness_values = np.array([train_model_with_params(individual,agrs) for individual in population])
#             #fitness_values = np.random.rand(pop_size)
#             # 记录当前代的最佳解
#             best_gen_idx = np.argmax(fitness_values)
#             best_gen_accuracy = fitness_values[best_gen_idx]
#
#             if best_gen_accuracy > best_accuracy:
#                 best_accuracy = best_gen_accuracy
#                 best_solution = population[best_gen_idx]
#
#             # 记录当前代的参数和准确率
#             best_gen_params = {param_keys[i]: population[best_gen_idx][i] for i in range(len(param_keys))}
#             f.write(f'Generation {generation}: Best Accuracy = {best_gen_accuracy}, Best Params = {best_gen_params}\n')
#             f.flush()  # 确保每次写入后立即刷新缓冲区
#             # 选择操作
#             parents = selection(population, fitness_values, num_parents)
#             # 计算需要生成的子代数量
#             offspring_size = pop_size - len(parents)
#
#             if offspring_size > 0:
#                 # 交叉操作
#                 offspring = crossover(parents, offspring_size)
#
#                 # 变异操作
#                 offspring = mutation(offspring, mutation_rate, bounds)
#
#                 # 生成新种群
#                 population = np.vstack((parents, offspring))
#             else:
#                 population = parents
#             # 输出当前代的最佳准确率
#             print(f'Generation {generation}: Best Accuracy = {best_accuracy}')
#
#     return best_solution, best_accuracy
#
#
# def optimize_model_parameters(args):
#     # 参数设置
#     # 定义参数维度
#     # 参数范围
#     parameter_bounds = {
#         'lr': (0.0001, 0.05),
#         'momentum': (0.8, 0.9),
#         'weight_decay': (0.000001, 0.0001),
#         'gamma': (0.1, 0.9),
#         'drop': (0.1, 0.9),
#         'T': (0.1, 3),
#         'ahadbl': (0.3, 1),
#         'ahasla': (0.3, 1),
#     }
#     dimension = len(parameter_bounds)
#
#     # 将参数转换为向量的索引
#     param_keys = list(parameter_bounds.keys())
#     population_size = 4#种群个数
#
#
#     num_generations = 100
#     num_parents = 2
#     mutation_rate = 0.2
#
#     # 执行Artemisinin optimization
#     best_solution, best_accuracy = artemisinin_optimization(population_size, dimension,parameter_bounds, num_generations, num_parents, mutation_rate,args,param_keys)
#
#     # 输出最优解的参数
#
#     best_params = {param_keys[i]: best_solution[i] for i in range(len(param_keys))}
#     print('Best Solution:', best_params)
#     print('Best Accuracy:', best_accuracy)
#
#     return best_params, best_accuracy

#=============加入优化


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_cosine_decay_learning_rate(state, optimizer, epoch, initial_lr, total_epochs):
    if epoch < total_epochs:
        if state['lr'] <= 0.0001:
            new_lr = 0.00035
        else:
            new_lr = initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
        state['lr'] = new_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
def adjust_learning_rate(state, optimizer, epoch):
    # global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser(description='Uterus') # 参数设置,使得我们能够手动输入命令行参数，让风格变得和Linux命令行差不多

    # Datasets
    parser.add_argument('--dataset', default='Uterus', type=str)
    # 路径
    #parser.add_argument('--dataset-path', default="/home/test/wym/Uterus_Dis_Cl/data/Uterus_v6", type=str,metavar='PATH')
    #parser.add_argument('--dataset-path', default="C:/Users/10433/Desktop/Uterus_Dis_Cl/Uterus_Dis_Cl/10all_jpg", type=str,metavar='PATH')
    parser.add_argument('--dataset-path', default="/home/test/Uterus_Dis_Cl/10all_jpg/", type=str,metavar='PATH')
    parser.add_argument('--split-dataset', default=False, type=bool)
    parser.add_argument('--kflod_num', default=0, type=int)
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # Optimization options
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--drop', '--dropout', default=0, type=float,
                        metavar='Dropout', help='Dropout ratio')
    parser.add_argument('--schedule', type=int, nargs='+', default=[25, 125, 225],#[25, 125, 225]
                            help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default= 0.11045636121793401, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9124260723081905, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,#2.174584692473294e-06
                        metavar='W', help='weight decay (default: 1e-4)')
    # Checkpoints
    #路径
    #parser.add_argument('--checkpoint', default='C:/Users/10433/Desktop/Uterus_Dis_Cl/checkpoint', type=str, metavar='PATH',
    #                    help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--checkpoint', default='/home/test/Uterus_Dis_Cl/checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # Architecture
    parser.add_argument('--mode', type=str, required=True) # 'baseline'
    parser.add_argument('--arch', metavar='ARCH', default='resnet20', choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('--depth', type=int, default=22, help='Model depth.')
    parser.add_argument('--block-name', type=str, default='Bottleneck',
                        help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
    parser.add_argument('--layer', type=int, default=[3,4,6,3], help='')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
    parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
    # Miscs
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--evaluate', dest='evaluate', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    #Device options

    parser.add_argument('--gpu-id', default='1', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    # SLA
    parser.add_argument('--aug', type=str, default=None)
    parser.add_argument('--T', type=float, default=1.0)
    parser.add_argument('--with-large-loss', action='store_true')

    #权重
    parser.add_argument('--ahadbl',type=float, default=0.4)
    parser.add_argument('--ahasla',type=float, default=0.6)

    args = parser.parse_args()


    # 设备准备
    # Random seed                                                  
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    # cpu or gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    #os.environ['CUDA_VISIBLE_DEVICES'] = "1,6,7"
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)# Use CPU
    else:
        device = torch.device("cpu")
        torch.manual_seed(args.manualSeed)

    main(args)
    #优化算法
    #optimize_model_parameters(args)




