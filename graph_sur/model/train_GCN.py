import os
from new_GCNmodel import *
import torch
import argparse
from utils import init_optim
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import pandas as pd


parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--label_csv', type=str, default='../label/test_NB_VS_Wilms.csv',
					help='experiment code')
parser.add_argument('--optim', dest='optimizer', default='adam')
parser.add_argument('--lr', default=0.001)
parser.add_argument('--weight_decay', default=0.0001, type=float)
parser.add_argument('--model_type',type = str,default = 'clam_sb')
parser.add_argument('--drop_out',type = str,default = 'True')
parser.add_argument('--n_classes',type = int,default = 4)
parser.add_argument('--model_size',type = str,default = 'small')
parser.add_argument('--epoch',default = 50,type = int)
args = parser.parse_args()

def calculate_similarity(feature_list):
    n = len(feature_list)
    matrix = np.zeros((n,n),dtype = int)
    thershold = 0.995
    for i in range(n):
        for j in range(i,n):
            if torch.cosine_similarity(torch.unsqueeze(feature_list[i],dim=0),torch.unsqueeze(feature_list[j],dim=0)) > thershold:
                matrix[i,j] = 1
                matrix[j,i] = 1
    return matrix

if __name__ == '__main__':

    model = GcnNet().cuda()
    print(model)
    model.train()
    optimizer = init_optim(args.optimizer, model.parameters(), args.lr, args.weight_decay)

    loss_fun = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    svs_file_path = './segment_cell/768-stage'
    svs_files = os.listdir(svs_file_path)

    train =0.7
    test = 0.3

    batch_num = 10
    best_acc = 0
    #
    label = pd.read_table('./TNMstage.txt').values

    for epoch in range(args.epoch):

        train_list = random.sample(svs_files, int(len(svs_files) * train))
        test_list = random.sample(svs_files, int(len(svs_files) * test))

        batch_loss = 0
        number = 0
        for svs_file in train_list:
            if svs_file.split('.')[-1]!='svs':
                print(svs_file)
                svs_path = os.path.join(svs_file_path,svs_file)
                folders = os.listdir(svs_path)

                feature_list = []
                for folder in folders:
                    if folder.isalnum():
                        cluster_path = os.path.join(svs_path,folder)
                        features = os.listdir(cluster_path)
                        for feature in features:
                            if feature.split('.')[0] == 'feature':
                                cluster_feature = torch.load(os.path.join(cluster_path,feature))
                                # cluster_feature = torch.mean(cluster_feature,dim=0)
                                # feature_list.append(cluster_feature)
                                if len(feature_list)>0:
                                    feature_list = torch.cat((feature_list,cluster_feature),dim=0)
                                else:
                                    feature_list = cluster_feature
                n = len(feature_list)
                if n == 0 :
                    continue
                similarity_matrix = calculate_similarity(feature_list)
                #
                # if svs_file.split('-')[3] ==  '01A':
                #     label = torch.tensor([1]).cuda()
                # else:
                #     label = torch.tensor([0]).cuda()

                for LABEL in label:
                    if LABEL[0] == svs_file[0:12]:
                        stage = LABEL[1]
                        break

                if stage == 't1':
                    Label = torch.tensor([0]).cuda(0)
                elif stage == 't2':
                    Label = torch.tensor([1]).cuda(0)
                elif stage == 't3':
                    Label = torch.tensor([2]).cuda(0)
                else :
                    Label = torch.tensor([3]).cuda(0)

                # if svs_file == 'TCGA-55-8619-11A-01-TS1':
                #     print(svs_file)
                # for i in range(n):
                #     feature_list[i] = feature_list[i].cpu().detach().numpy()
                # print(svs_file)
                final,_ = model(torch.tensor(similarity_matrix).float().cuda(),(torch.tensor(feature_list).float().squeeze().cuda()))
                loss = loss_fun(final, Label).float()

                loss.backward()
                # optimizer.step()
                number +=1
                batch_loss += loss
                if number % batch_num == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    print('epoch: ' + str(epoch) + '----' + 'batch_loss = ' + str(batch_loss))
                    batch_loss = 0

        correct = 0
        for svs_file in test_list:
            if svs_file.split('.')[-1] != 'svs':
                svs_path = os.path.join(svs_file_path, svs_file)
                folders = os.listdir(svs_path)
                feature_list = []
                for folder in folders:
                    if folder.isalnum():
                        cluster_path = os.path.join(svs_path,folder)
                        features = os.listdir(cluster_path)
                        for feature in features:
                            if feature.split('.')[0] == 'feature':
                                cluster_feature = torch.load(os.path.join(cluster_path,feature))
                                # cluster_feature = torch.mean(cluster_feature,dim=0)
                                feature_list.append(cluster_feature)
                n = len(feature_list)
                if n == 0 :
                    continue
                similarity_matrix = calculate_similarity(feature_list)

                # if svs_file.split('-')[3] == '01A':
                #     label = torch.tensor([1]).cuda()
                # else:
                #     label = torch.tensor([0]).cuda()

                for LABEL in label:
                    if LABEL[0] == svs_file[0:12]:
                        stage = LABEL[1]
                        break

                if stage == 't1':
                    Label = torch.tensor([0]).cuda(0)
                elif stage == 't2':
                    Label = torch.tensor([1]).cuda(0)
                elif stage == 't3':
                    Label = torch.tensor([2]).cuda(0)
                else :
                    Label = torch.tensor([3]).cuda(0)

                for i in range(n):
                    feature_list[i] = feature_list[i].cpu().detach().numpy()

                final = model(torch.tensor(similarity_matrix).float().cuda(),(torch.tensor(feature_list).float().squeeze().cuda()))
                value,index = torch.max(final,dim = 1)
                if index == label:
                    correct +=1
                print(str(F.softmax(final,dim = 1)) + ' ----label: '+ str(Label))

        best_acc = max(best_acc,correct/len(test_list))
        print('epoch:' + str(epoch) + '----correct number:' + str(correct) + '----Acc:' + str(correct/len(test_list)) + '-----best_acc:' + str(best_acc))

    model_path = './model_clam/whole-gcn' +str(best_acc) +'.pkl'
    save_model = {'net': model.state_dict()}
    torch.save(save_model, model_path)
    print('  trained')