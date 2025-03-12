import torch
import torch.nn as nn
import torch.nn.parallel

import numpy as np
from sklearn import metrics

import os
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt


import os, sys
sys.path.append("..")
import models.Uterus as models
import SLA.checkmodel as checkmodel


# os.environ['CUDA_VISIBLE_DEVICES'] = gpus
#
# manualSeed = random.randint(1, 10000)
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)
# torch.cuda.manual_seed_all(manualSeed)
# use_cuda = torch.cuda.is_available()


class Data_Get():

    def __init__(self):
        self.path = "/home/dkd/dataset_ct_ap_pvp/"
        self._root = "/home/temp55/wym/Uterus_Dis_Cl/checkpoint/Uterus/"
        self._criterion = nn.CrossEntropyLoss()

    def model_import(self,checkpoint):
        if checkpoint.find("resnet18_pre") != -1:
            name = "resnet18_pre"
            # model = models.__dict__[name](
            #     num_classes=128,
            # )
            # model.num_features = 128
        model = check_model(name, 3 * 4, 3)
        exit()

    def model_acquire(self):
        model_list = ["resnet18_pre", "resnet50_pre", "convnext_base"]
        checkpoint_list = ["sla+sd_resnet18_pre_20230516_1", "sla+sd_resnet50_pre_20230516_2", "sla+sd_convnext_base_pre_20230510_1"]
        # model = ViT('B_16_imagenet1k', pretrained=False, num_classes=2, image_size=image_size)
        # if use_cuda:
        #     model = torch.nn.DataParallel(model).cuda()

        # path_list = ["hcc_icc_224_acc_1.pth.tar", "hcc_met_224_best_acc_1.pth.tar", "icc_met_224_best_acc_4.pth.tar"] #
        # mode = ["hcc_icc", "hcc_met", "icc_met"]  #
        result_list = []
        for idx, checkpoint in enumerate(checkpoint_list):
            model_import(self, checkpoint)
            checkpoint_model_path = os.path.join(self._root, checkpoint)
            state_dict = torch.load(checkpoint_model_path)
            dict_state_model = state_dict["state_dict"]
            model.load_state_dict(dict_state_model)
            df = self.test(model=model, mode=mode[idx])
            result_list.append(df)

        result = pd.concat([result_list[0], result_list[1], result_list[2]]) #
        return result

    def test(self, model, mode):
        model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        # bar = Bar('Processing', max=len(self._testloader))
        people_id = []
        pred = []
        labels = []
        Liver_loader_test = CancerSeT_CSV(self.path, 'val', mode)
        dataloader = torch.utils.data.DataLoader(Liver_loader_test, batch_size=batchsize_test, shuffle=False)

        with torch.no_grad():
            for idx, data in enumerate(dataloader):
                inputs = data["img"].float()
                targets = data["label"].float()
                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                outputs = model(inputs)
                # loss = self._criterion(outputs, targets.long())
                # prec1 = accuracy(outputs.data, targets.long().data, topk=(1,))
                # losses.update(loss.item(), inputs.size(0))
                # top1.update(prec1[0].item(), inputs.size(0))

                people_id.extend(data['id'])
                if mode == "hcc_met":
                    # list = []
                    # targets_list = targets.detach().cpu().numpy().tolist()
                    # prediction = F.softmax(outputs, dim=1)
                    # key = prediction[:, 1]
                    # key = key.unsqueeze(1)
                    # prediction = torch.cat((prediction, key), 1)
                    # prediction[:, 1] = 0
                    # pred.extend(prediction.detach().cpu().numpy())
                    # for value in targets_list:
                    #     if value == 1.:
                    #         list.append(2.0)
                    #     else:
                    #         list.append(value)
                    # labels.extend(list)
                    targets_list = targets.detach().cpu().numpy().tolist()
                    labels.extend(targets_list)
                    # print(labels)
                    prediction = F.softmax(outputs, dim=1)
                    key = prediction[:, 1]
                    key = key.unsqueeze(1)
                    prediction = torch.cat((prediction, key), 1)
                    prediction[:, 1] = 0
                    pred.extend(prediction.detach().cpu().numpy())
                if mode == "icc_met":
                    # list = []
                    # targets_list = targets.detach().cpu().numpy().tolist()
                    # prediction = F.softmax(outputs, dim=1)
                    # key = prediction[:, 0]
                    # key = key.unsqueeze(1)
                    # prediction = torch.cat((key, prediction), 1)
                    # prediction[:, 0] = 0
                    # pred.extend(prediction.detach().cpu().numpy())
                    # for value in targets_list:
                    #     if value == 0.:
                    #         list.append(1.0)
                    #     else:
                    #         list.append(2.0)
                    # labels.extend(list)
                    targets_list = targets.detach().cpu().numpy().tolist()
                    labels.extend(targets_list)
                    prediction = F.softmax(outputs, dim=1)
                    key = prediction[:, 0]
                    key = key.unsqueeze(1)
                    prediction = torch.cat((key, prediction), 1)
                    prediction[:, 0] = 0
                    pred.extend(prediction.detach().cpu().numpy())
                elif mode == "hcc_icc":
                    targets_list = targets.detach().cpu().numpy().tolist()
                    labels.extend(targets_list)
                    prediction = F.softmax(outputs, dim=1)
                    key = prediction[:, 1]
                    key = key.unsqueeze(1)
                    prediction = torch.cat((prediction, key), 1)
                    prediction[:, 2] = 0
                    pred.extend(prediction.detach().cpu().numpy())
            df = pd.DataFrame({'people_id': people_id, 'preds': pred, 'labels': labels})
            return df


class result_Analysis():

    def __init__(self):
        self.result = Data_Get().model_acquire()

    def ACC_statistic(self, df):
        id_count = 0
        person_label = []
        person_preds = []
        person_preds_label = []
        for name, id_group in df:
            preds_pro = np.mean(id_group["preds"].values, axis=0)
            person_preds.append(list(preds_pro))
            preds_pro = float(np.argmax([preds_pro]))
            person_preds_label.append(preds_pro)
            labels = float(id_group["labels"].mean())
            person_label.append(labels)
            if preds_pro == labels:
                id_count += 1
        acc_statistic = id_count / len(df)
        return person_preds, person_label, person_preds_label, acc_statistic


    def auc(self,person_preds, person_label):
        roc_auc = dict()
        y_true = label_binarize(person_label, classes=[0, 1, 2])
        y_true = np.array(y_true)
        person_preds = np.array(person_preds)
        for i in range(3):
            roc_auc[i] = metrics.roc_auc_score(y_true[:, i], person_preds[:, i])
            print("class {} ".format(i) + 'statis auc ' + '{:.4f}'.format(roc_auc[i]))
        macro_auc = metrics.roc_auc_score(y_true, person_preds)
        return macro_auc

    def Confusion_Mat(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        confusion_data = confusion_matrix(y_true, y_pred, labels=[0., 1., 2.])
        print("confusion matrix: \n", confusion_data)
        plt.matshow(confusion_data, cmap=plt.cm.Reds)


    def ensemble(self):
        df = self.result
        df = df.groupby('people_id')[['labels', 'preds']]
        person_preds, person_label, person_preds_label, acc_statistic = self.ACC_statistic(df)
        print(person_preds, person_label)
        macro_auc = self.auc(person_preds, person_label)
        self.Confusion_Mat(person_label, person_preds_label)
        print("macro_auc is {}".format(macro_auc))

if __name__ == '__main__':
    Voting_Calculation = result_Analysis()
    Voting_Calculation.ensemble()








