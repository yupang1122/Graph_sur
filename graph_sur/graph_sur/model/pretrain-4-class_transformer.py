from models.Transformer import Transformer
import argparse
import os
import torch
import cv2
from utils import init_optim
import torch.nn as nn
from resnet import *
import pandas as pd
import ml_collections
import random
from torch.nn import functional as F


torch.cuda.set_device(0) if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--label_csv', type=str, default='../label/test_NB_VS_Wilms.csv',
					help='experiment code')
parser.add_argument('--optim', dest='optimizer', default='adam')
parser.add_argument('--lr', default=0.009)
parser.add_argument('--weight_decay', default=0.0001, type=float)
parser.add_argument('--data_dir', type=str, default='./segment_cell/stage_patches')
parser.add_argument('--model_type',type = str,default = 'clam_sb')
parser.add_argument('--drop_out',type = str,default = 'True')
parser.add_argument('--n_classes',type = int,default = 4)
parser.add_argument('--model_size',type = str,default = 'small')
parser.add_argument('--epoch',default = 50,type = int)
args = parser.parse_args()



def feature_extract(datas_path):

    feature_list = []

    Model = resnet18(pretrained=True)
    # del(Model.fc1)
    # del(Model.fc)
    # pretrained_dict = torch.load('./SLIC-segmentation/nine_class/best9classtensor_sgd0.9919_dropout_32.ckpt')
    # pretrained_dict.popitem(last=True)
    # pretrained_dict.popitem(last=True)
    # pretrained_dict.popitem(last=True)
    # pretrained_dict.popitem(last=True)
    # new_pretrained_dict = {}
    # for k, v in pretrained_dict.items():
    #     new_pretrained_dict[k[7:]] = v
    # model_dict = Model.state_dict()
    # model_dict.update(new_pretrained_dict)
    # Model.load_state_dict(model_dict)

    # del(Model.fc)
    # Model.to(device)
    Model.cuda()
    # Model.eval()


    datas = os.listdir(datas_path)
    for data in datas :
        if data.split('.')[-1] == 'jpg':
            data_path = os.path.join(datas_path,data)

            feature = Model(torch.tensor(cv2.imread(data_path)).resize(1, 3, 200, 200).to(torch.float32).cuda())  # 512

            coord_x = torch.tensor([int(data.split('_')[0]), int((data.split('_')[1]).split('.')[0]),get_area(cv2.imread(data_path))]).reshape([1, 3]).cuda()
            feature = torch.cat((feature.detach(), coord_x), dim=1)
            feature_list.append(feature[0].cpu().detach().numpy())

    return torch.tensor(feature_list).cuda()

def get_first_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 515
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.3
    config.transformer.dropout_rate = 0.1
    config.representation_size = None

    # custom
    config.classifier = 'seg'
    # config.resnet_pretrained_path = None
    # config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
    # config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config

def get_transformer_config():
    """Returns the Resnet50 + ViT-L/16 configuration. customized """
    config = get_first_config()
    # config.patches.grid = (16, 16)
    # config.resnet = ml_collections.ConfigDict()

    config.classifier = 'seg'

    config.n_classes = 2
    config.activation = 'softmax'
    return config


if __name__ == '__main__':

    model_dict = get_transformer_config()
    model = Transformer(model_dict,num_layer = 5).cuda()

    print(model)
    model.train()
    optimizer = init_optim(args.optimizer, model.parameters(), args.lr, args.weight_decay)

    loss_fun = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    label = pd.read_table('./TNMstage.txt').values
    datasets = os.listdir(args.data_dir)

    train = 0.8
    test = 0.2
    batch_num = 20
    best_acc = 0
    for epoch in range(args.epoch):
        i = 1
        batch_loss = 0

        train_list = random.sample(datasets, int(len(datasets) * train))
        test_list = random.sample(datasets, int(len(datasets) * test))

        for dataset in train_list:
            if dataset.split('.')[-1] != 'svs':
                i+=1

                dataset_path = os.path.join(args.data_dir,dataset)
                feature = feature_extract(dataset_path)

                attn_weight,feature,final = model(feature)

                # if dataset.split('-')[3] == '01A':
                #     label = torch.tensor([0,1])
                # else:
                #     label = torch.tensor([1,0])

                for LABEL in label:
                    if LABEL[0] == dataset[0:12]:
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

                loss = loss_fun(feature, Label).float()

                # loss = loss_fun(final, label).float()

                loss.backward()
                    # optimizer.step()
                batch_loss += loss
                if i%batch_num == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    print('epoch: ' + str(epoch) +'----'+'batch_loss = ' + str(batch_loss))
                    batch_loss = 0

        correct = 0
        for svs_file in test_list:
            if svs_file.split('.')[-1] != 'svs':
                svs_path = os.path.join(args.data_dir, svs_file)
                folders = os.listdir(svs_path)
                feature = feature_extract(svs_path)


                attn_weight,feature,final = model(feature)

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
                else:
                    Label = torch.tensor([3]).cuda(0)

                value, index = torch.max(feature, dim=1)
                if index == Label:
                    correct +=1
                print(str(F.softmax(feature,dim = 1)) + ' ----label: '+ str(Label))

        best_acc = max(best_acc,correct/len(test_list))
        print('epoch:' + str(epoch) + '----correct number:' + str(correct) + '----Acc:' + str(correct/len(test_list)) + '-----best_acc:' + str(best_acc))

    model_path = './model_clam/4-class-pretrain_transformer' + str(best_acc) + '.pkl'
    save_model = {'net': model.state_dict()}
    torch.save(save_model, model_path)
    print('  trained')
