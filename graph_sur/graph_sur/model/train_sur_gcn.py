import os
import random

import pandas
from pandas import DataFrame
import pandas as pd

import cv2
from survive_n import *
from utils import init_optim
import argparse


parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--text_path',type=str,default = './TCGA_MEASUREMENTS.xlsx')
parser.add_argument('--optim', dest='optimizer', default='adam')
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--weight_decay', default=0.00001, type=float)
parser.add_argument('--batch_size',default = 10,type = int)
parser.add_argument('--epoch',default = 10 ,type = int)
parser.add_argument('--feature_path',default = './stage_patches',type = str)
parser.add_argument('--model_path',default = './256-survive_model.pkl')
args = parser.parse_args()

os.environ['CUDA_LAUNCH_BLOCKING']= '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def get_adjacency(feature):
    n = len(feature)
    matrix = np.zeros((n,n),dtype = int)
    thershold = 0.90
    for i in range(n):
        for j in range(i,n):
            if torch.cosine_similarity(torch.unsqueeze(feature[i],dim=0),torch.unsqueeze(feature[j],dim=0)) > thershold:
                matrix[i,j] = 1
                matrix[j,i] = 1
    return matrix


if __name__ == '__main__':
    # model = Survive_GCN(input_dim=256,hidden_dim=256,num_layers=3).cuda()

    model = torch.load(args.model_path)
    # model.load_state_dict(Model.parameters)

    print(model)

    # # test and predict hazard
    # model.eval()
    #
    # text = pandas.read_excel(args.text_path)
    # slides = os.listdir(args.feature_path)
    #
    # for slide in slides:
    #     patient_name = slide
    #     if len(text.loc[text['ID'] == patient_name[0:12]]) > 0:
    #         target_feature = torch.load(os.path.join(args.feature_path, slide, '256_cnn_slide_feature.pkl')).cuda()
    #         target_adjancency = get_adjacency(target_feature)
    #         cluster_label = torch.load(os.path.join(args.feature_path, slide, '256_cnn_slide_feature_label.pkl'))
    #         label_patch = torch.load(os.path.join(args.feature_path, slide, '515-cluster_patch_labels.pkl'))
    #         target_risk,attention = model(target_adjancency, target_feature)
    #
    #         thumbnail_path =os.path.join(args.feature_path, slide,'thumbnail.png')
    #         thumbnail = cv2.imread(thumbnail_path)
    #         thumbnail = np.ndarray([(thumbnail.shape[0]//200)+1,(thumbnail.shape[1]//200)+1,3])
    #         atten_thumbnail_path = os.path.join(args.feature_path,slide,'256_atten_small_thumbnail.png')
    #
    #         for cluster in cluster_label:
    #             patches = label_patch[cluster]
    #             if attention[cluster] > attention.mean():
    #                 color = [0, 0, float(255 * (attention[cluster]) * 10)]
    #             else:
    #                 color = [float(255 * (attention[cluster]) * 10), 0, 0]
    #             for patch in patches:
    #                 patch_coord_x = int(patch.split('_')[0])
    #                 patch_coord_y = int((patch.split('_')[1]).split('.')[0])
    #                 thumbnail[patch_coord_x,patch_coord_y] = color
    #
    #         # text.loc[text['ID'] == patient_name[0:12],'cnn_predict_risk'] = target_risk.detach().cpu().numpy()
    #         # DataFrame(text).to_excel(args.text_path,sheet_name = 'Tabelle1',index = False,header = True)
    #         cv2.imwrite(atten_thumbnail_path,thumbnail)
    #         print(target_risk)





    with torch.autograd.set_detect_anomaly(True):

        model.train()
        optimizer = init_optim(args.optimizer, model.parameters(), args.lr, args.weight_decay)

        # loss_fun = nn.CrossEntropyLoss()
        optimizer.zero_grad()

        text = pandas.read_excel(args.text_path)

        slides = os.listdir(args.feature_path)
        # message = {}
        #   cross validation
        cross_validate = 10
        train_set = random.sample(slides,int(len(slides)*0.9))
        valid_set = random.sample(slides,int(len(slides)*0.1))
        risk_list = []
        batch_name = []
        message = {}
        for epoch in range(args.epoch):

            number = 0
            for slide in train_set:
                patient_name = slide
                if len(text.loc[text['ID'] == patient_name[0:12]])>0 :
                    if patient_name not in message:
                        message[patient_name] ={}
                        row = text.loc[text['ID'] == patient_name[0:12]]
                        message[patient_name]['sex'] = row['gender'].values[0]
                        message[patient_name]['age'] = row['years_to_birth'].values[0]
                        message[patient_name]['stage'] = row['pathologic_stage'].values[0]
                        message[patient_name]['primary'] = row['tumor_tissue_site'].values[0]
                        message[patient_name]['survive_time'] = row['days_to_event'].values[0]

                    batch_name.append(patient_name)
                    number +=1

                    if number %args.batch_size == 0:
                        batch_loss = 0

                        for target in batch_name:

                            target_feature = torch.load(os.path.join(args.feature_path, target, '256_cnn_slide_feature.pkl')).cuda()
                            target_adjancency = get_adjacency(target_feature)
                            target_risk,_ = model(target_adjancency,target_feature)

                            del target_feature
                            del target_adjancency

                            exp_risk = torch.tensor([0]).cuda()
                            # get survive time longer than target
                            for patient in batch_name:
                                if message[patient]['survive_time'] >= message[target]['survive_time']:

                                    patient_feature = torch.load(os.path.join(args.feature_path,patient,'256_cnn_slide_feature.pkl')).cuda()
                                    patient_adjancency = get_adjacency(patient_feature)
                                    patient_risk,_ = model(patient_adjancency,patient_feature)
                                    del patient_feature
                                    del patient_adjancency

                                    exp_risk = torch.add(exp_risk,torch.exp(patient_risk))

                            log_risk = torch.log(exp_risk)
                            uncensored_likelihood = target_risk - log_risk
                            censored_likelihood = uncensored_likelihood

                            neg_likelihood = torch.tensor([0]).cuda()-censored_likelihood
                            loss = neg_likelihood + 1e-6

                            if torch.isnan(loss) or torch.isinf(loss):
                                continue
                            loss.backward()
                            exp_risk = 0
                            batch_loss += loss
                        optimizer.step()
                        optimizer.zero_grad()
                        print('epoch : ' + str(epoch) + '-----------  batch_loss : ' + str(batch_loss))
                        number = 0
                        batch_name = []

            correct = 0
            for slide in valid_set:
                patient_name = slide
                if len(text.loc[text['ID'] == patient_name[0:12]]) > 0:
                    if patient_name not in message:
                        message[patient_name] = {}
                        row = text.loc[text['ID'] == patient_name[0:12]]
                        message[patient_name]['sex'] = row['gender'].values[0]
                        message[patient_name]['age'] = row['years_to_birth'].values[0]
                        message[patient_name]['stage'] = row['pathologic_stage'].values[0]
                        message[patient_name]['primary'] = row['tumor_tissue_site'].values[0]
                        message[patient_name]['survive_time'] = row['days_to_event'].values[0]

                    feature = torch.load(os.path.join(args.feature_path, slide, '256_cnn_slide_feature.pkl')).cuda()
                    adjancency = get_adjacency(feature)

                    risk = model(adjancency, feature) # no convariate elements

                    message[patient_name]['risk'] = risk

            for slide in valid_set:
                correct_number = 0
                patient_name = slide
                if len(text.loc[text['ID'] == patient_name[0:12]]) > 0:
                    patient_risk = message[patient_name]['risk']
                    for target in valid_set:
                        if len(text.loc[text['ID'] == target[0:12]]) > 0:
                            target_risk = message[target]['risk']
                            if (message[patient_name]['survive_time'] >= message[target]['survive_time'] and patient_risk >= target_risk) or (message[patient_name]['survive_time'] <= message[target]['survive_time'] and patient_risk <= target_risk):
                                correct_number +=1

                print('epoch : ' + str(epoch)+'---patient correct ratio: ' + str(correct_number/len(valid_set)))

        model_path = args.model_path
        torch.save(model,model_path)
        print('trained')




