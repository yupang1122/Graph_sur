import os
import cv2
import os
import cv2
import numpy as np
from resnet import *
from PIL import Image
import torch
import pandas as pd

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

slide_images_path = './stage_patches'
cluster_images_path = './stage_patches'
local_path = './patches'

slide_images = os.listdir(slide_images_path)

color_list = [[255,255,255],[255,255,128],[255,128,255],[255,128,128],[128,255,255],[128,128,255],[128,128,128]]

Model = resnet18()
# del(Model.fc1)
# del(Model.fc)
pretrained_dict = torch.load('../SLIC-segmentation/nine_class/best9classtensor_sgd0.9919_dropout_32.ckpt')
# pretrained_dict.popitem(last=True)
# pretrained_dict.popitem(last=True)
# pretrained_dict.popitem(last=True)
# pretrained_dict.popitem(last=True)
new_pretrained_dict = {}
for k, v in pretrained_dict.items():
    new_pretrained_dict[k[7:]] = v
model_dict = Model.state_dict()
model_dict.update(new_pretrained_dict)
Model.load_state_dict(model_dict)
# del(Model.fc)
Model.to(device)
Model.eval()

def get_area(image):
    res = 0
    if image.max()<230:
        res = 1

    return res
if __name__ == "__main__":

    text_path = './TCGA_MEASUREMENTS.xlsx'
    text = pd.read_excel(text_path)
    pathology_T_stage_list = ['nan', 't1', 't2', 't3', 't4', 't4a', 't4b']
    pathology_N_stage_list = ['nan', 'n0', 'n1', 'n1a', 'n1b', 'n1c', 'n2', 'n2a', 'n2b', 'nx']
    pathology_M_stage_list = ['nan', 'm0', 'm1', 'm1a', 'm1b', 'mx', 'NA']
    gender_list = ['nan', 'male', 'female']

    for slide_image in slide_images:
        if len(text.loc[text['ID'] == slide_image[0:12]]) == 0:
            continue
        slide_feature = []
        patches = os.path.join(slide_images_path,slide_image)
        patch_images = os.listdir(patches)
        patch_feature = {}
        patch_feature_path = os.path.join(patches,'520-patch_feature.pkl')

        target = text.loc[text['ID'] == slide_image[0:12]]
        slide_feature.append(gender_list.index(str(target['gender'].values[0])))
        slide_feature.append(pathology_M_stage_list.index(str(target['pathology_M_stage'].values[0])))
        slide_feature.append(pathology_N_stage_list.index(str(target['pathology_N_stage'].values[0])))
        slide_feature.append(pathology_T_stage_list.index(str(target['pathology_T_stage'].values[0])))
        slide_feature.append(int(target['years_to_birth'].values))
        # slide_feature = np.array(slide_feature)

        if os.path.exists(patch_feature_path):
            continue
        ADI, DEB, LYM, MUC, MUS, NORM, STR, TUM, BACK = {}, {}, {}, {}, {}, {}, {}, {}, {}
        for patch_image in patch_images:
            if patch_image.split('.')[-1] != 'jpg' or patch_image.split('_')[0].isnumeric() != True:
                continue
            patch_image_path = os.path.join(patches,patch_image)
            image = cv2.imread(patch_image_path)

            feature,predict = Model(torch.tensor(image).resize(1,3,200,200).to(torch.float32).cuda())
            value,index = torch.max(predict,dim=1)
            if index.item() == 0:
                ADI[patch_image] = value.detach().cpu()
            elif index.item() == 1:
                DEB[patch_image] = value.detach().cpu()
            elif index.item() == 2:
                LYM[patch_image] = value.detach().cpu()
            elif index.item() == 3:
                MUC[patch_image] = value.detach().cpu()
            elif index.item() == 4:
                MUS[patch_image] = value.detach().cpu()
            elif index.item() == 5:
                NORM[patch_image] = value.detach().cpu()
            elif index.item() == 6:
                STR[patch_image] = value.detach().cpu()
            elif index.item() == 7:
                TUM[patch_image] = value.detach().cpu()
            elif index.item() == 8:
                BACK[patch_image] = value.detach().cpu()

            coord_x = torch.tensor([int(patch_image.split('_')[0]),int ((patch_image.split('_')[1]).split('.')[0]),get_area(image)] + slide_feature).reshape([1,8]).cuda()
            final_feature = torch.cat((feature.detach(),coord_x),dim=1)

            patch_feature[patch_image] = final_feature
            print(patch_image)

            final_feature = feature.detach()
            patch_feature[patch_image] = final_feature
            print(patch_image)

        patch_features= os.path.join(patches,'520-patch_feature.pkl')
        torch.save(patch_feature,patch_features)
        #sort dicitonary and choose 10%
        ADI = sorted(ADI.items(),key = lambda x:x[1])[0:int(len(MUS)*0.1)] if len(ADI) > 10 else sorted(ADI.items(),key = lambda x:x[1])
        DEB = sorted(DEB.items(), key=lambda x: x[1])[0:int(len(MUS)*0.1)] if len(DEB) > 10 else sorted(DEB.items(),key = lambda x:x[1])
        LYM = sorted(LYM.items(), key=lambda x: x[1])[0:int(len(MUS)*0.1)] if len(LYM) > 10 else sorted(LYM.items(),key = lambda x:x[1])
        MUC = sorted(MUC.items(), key=lambda x: x[1])[0:int(len(MUS)*0.1)] if len(MUC) > 10 else sorted(MUC.items(),key = lambda x:x[1])
        MUS = sorted(MUS.items(), key=lambda x: x[1])[0:int(len(MUS)*0.1)] if len(MUS) > 10 else sorted(MUS.items(),key = lambda x:x[1])
        NORM = sorted(NORM.items(), key=lambda x: x[1])[0:int(len(MUS)*0.1)] if len(NORM) > 10 else sorted(NORM.items(),key = lambda x:x[1])
        STR = sorted(STR.items(), key=lambda x: x[1])[0:int(len(MUS)*0.1)] if len(STR) > 10 else sorted(STR.items(),key = lambda x:x[1])
        TUM = sorted(TUM.items(), key=lambda x: x[1])[0:int(len(MUS)*0.1)] if len(TUM) > 10 else sorted(TUM.items(),key = lambda x:x[1])
        BACK = sorted(BACK.items(), key=lambda x: x[1])[0:int(len(MUS)*0.1)] if len(BACK) > 10 else sorted(BACK.items(),key = lambda x:x[1])
        #iteratly cluster all patches
        iteration = 5
        centroid = ADI+DEB+LYM+MUC+MUS+NORM+STR+TUM+BACK
        centroid_path = os.path.join(patches,'520-inital_centroid_patch_feature.pkl')
        torch.save(centroid,centroid_path)

        centroid_feature = [patch_feature[[i][0][0]] for i in centroid]
        centroid_index={}
        for i in range(len(centroid)):
            centroid_index[i] = []
        for _ in range (iteration):
            for patch in patch_feature:
                mid = []
                for i in range(len(centroid)):
                    mid.append(torch.cosine_similarity(centroid_feature[i],patch_feature[patch]).item())
                centroid_index[mid.index(max(mid))].append(patch) #get every patch cluster label

            for i in centroid_index:
                mid = []
                for j in centroid_index[i]:
                    mid.append(patch_feature[j])
                if len(mid) == 0:
                    centroid[i] = sum(mid)
                else:
                    centroid[i] = sum(mid)/len(mid)

        #save cluster index and feature
        cluster_centroid_path = os.path.join(patches,'520-cluster_centroid_feature.pkl')
        cluster_patches = os.path.join(patches,'520-cluster_patch_labels.pkl')


        torch.save(centroid,cluster_centroid_path)
        torch.save(centroid_index,cluster_patches)


        print(slide_image)
