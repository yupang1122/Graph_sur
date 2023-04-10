import os
import openslide
import torch
import cv2
import numpy as np
from resnet import *
import pandas as pd
import matplotlib.pyplot as plt

patches_path = './a6-6648'


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

model_path = './515_survive_model.pkl'
model = torch.load(model_path)
model.eval()

for step in os.listdir(patches_path):
    step_size = int(step)
    target_feature = torch.load(os.path.join(patches_path, step, '515_rnn_slide_feature.pkl')).cuda()
    target_adjancency = get_adjacency(target_feature)
    target_risk, atten = model(target_adjancency, target_feature)

    cluster_label = torch.load(os.path.join(patches_path, step, '515_rnn_slide_feature_label.pkl'))
    label_patch = torch.load(os.path.join(patches_path, step, '515-cluster_patch_labels.pkl'))

    thumbnail_path = os.path.join(patches_path, step, 'thumbnail.png')
    thumbnail = cv2.imread(thumbnail_path)
    # thumbnail = np.ndarray([(thumbnail.shape[0]//200)+1,(thumbnail.shape[1]//200)+1,3])
    atten_thumbnail_path = os.path.join(patches_path, step, 'atten_thumbnail.png')

    for cluster in cluster_label:
        patches = label_patch[cluster]

        # if atten[cluster] > atten.mean():
        #     red_value = float((atten[cluster] - atten.mean())/(atten.max() - atten.mean() + 0.0001))*100
        #     color = [255, 255,int(red_value)]
        # else:
        #     blue_value = float((atten.mean() - atten[cluster])/(atten.mean() - atten.min() + 0.0001)) *100
        #     color = [int(blue_value),255,255]

        color = [0,0,float(atten[cluster]/atten.max() + 0.0001)*100]


        for patch in patches:
            patch_coord_x = int(patch.split('_')[0])
            patch_coord_y = int((patch.split('_')[1]).split('.')[0])
            thumbnail[patch_coord_x * 200 +step_size :(patch_coord_x + 1) * 200 + step_size,patch_coord_y * 200 + step_size:(patch_coord_y + 1) * 200 + step_size] = color

    # text.loc[text['ID'] == patient_name[0:12],'rnn_predict_risk'] = target_risk.detach().cpu().numpy()
    # DataFrame(text).to_excel(args.text_path,sheet_name = 'Tabelle1',index = False,header = True)
    cv2.imwrite(atten_thumbnail_path, thumbnail)
    print(target_risk)