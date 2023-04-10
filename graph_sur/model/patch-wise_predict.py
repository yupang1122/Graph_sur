import os
import torch
import torch.nn as  nn
import cv2
import numpy as np
from models.predict_model import predict_model
import openslide
from skimage.segmentation import slic

model_path = './SLIC-segmentation/nine_class/best9classtensor_sgd0.9919_dropout_32.ckpt'

svs_file  = './segment_cell/stage'

predict = predict_model().cuda()
parameter = torch.load(model_path)
new_pretrained_dict = {}
for k, v in parameter.items():
    new_pretrained_dict[k[7:]] = v
Parameter = {}
Parameter['classifier1.weight'] = new_pretrained_dict['fc.weight']
Parameter['classifier1.bias'] = new_pretrained_dict['fc.bias']
Parameter['classifier2.weight'] = new_pretrained_dict['fc1.weight']
Parameter['classifier2.bias'] = new_pretrained_dict['fc1.bias']

predict.load_state_dict(Parameter)

predict.eval()
print(predict)

svss = os.listdir(svs_file)
svs_files_path = './segment_cell/stage_patches'
svs_files = os.listdir(svs_files_path)
for svs in svss:
    if svs.split('.')[-1] == 'svs':
        svs_f = svs.split('.')[0]
        svs_path = os.path.join(svs_file,svs)
        if os.path.exists(os.path.join(svs_files_path, svs_f)) != True:
            continue
        SVS = openslide.OpenSlide(svs_path)

        thumbnail = SVS.get_thumbnail(SVS.level_dimensions[1])
        get_Image = cv2.cvtColor(np.asarray(thumbnail), cv2.COLOR_RGB2BGR)

        segmentation = slic(get_Image, n_segments=768, compactness=20, enforce_connectivity=True, convert2lab=True, sigma=1)
        svs_path = os.path.join(svs_files_path,svs_f)
        thumbnail_path = os.path.join(svs_path,'thumbnail.png')
        cv2.imwrite(thumbnail_path,get_Image)

        svs_clusters = os.listdir(svs_path)

        label = {}
        for svs_cluster in svs_clusters:
            if svs_cluster.isalnum():
                feature_path = os.path.join(svs_path,svs_cluster,'transformer_feature.pkl')
                cluster_feature = torch.load(feature_path).cuda()
                # cluster_feature = cluster_feature.unsqueeze(dim=0).cuda()
                final, index = predict(cluster_feature)
                print(index)
                label[int(svs_cluster)] = index

        mask = np.zeros([segmentation.shape[0],segmentation.shape[1],3])
        mask_path = os.path.join(svs_path,'mask.png')
        for i in range(segmentation.shape[0]):
            for j in range(segmentation.shape[1]):
                if segmentation[i,j] in label:
                    Label = label[segmentation[i,j]]
                    if Label == 0:
                        mask[i,j] = [0,0,255]
                    elif Label == 1:
                        mask[i,j] = [0,128,255]
                    elif Label == 2:
                        mask[i,j] = [0,255,255]
                    elif Label == 3:
                        mask[i,j] = [0,255,128]
                    elif Label == 4:
                        mask[i,j] = [128,255,0]
                    elif Label == 5:
                        mask[i,j] = [255,255,0]
                    elif Label == 6:
                        mask[i,j] = [255,128,0]
                    elif Label == 7:
                        mask[i,j] = [255,0,0]
                    elif Label == 8:
                        mask[i,j] = [255,0,127]

        cv2.imwrite(mask_path,mask)
        print(svs)


