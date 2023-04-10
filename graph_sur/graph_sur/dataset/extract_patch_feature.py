import os
import openslide
import torch
import cv2
import numpy as np
from resnet import *
import pandas as pd
import matplotlib.pyplot as plt



# for step in os.listdir(patches_path):
#     ADI, DEB, LYM, MUC, MUS, NORM, STR, TUM, BACK = {}, {}, {}, {}, {}, {}, {}, {}, {}
#     patch_feature = {}
#     patch_images = os.listdir(os.path.join(patches_path,step))
#     for patch_image in patch_images:
#         if patch_image.split('.')[-1] != 'jpg' or patch_image.split('_')[0].isnumeric() != True:
#             continue
#         patch_image_path = os.path.join(os.path.join(patches_path,step), patch_image)
#         image = cv2.imread(patch_image_path)
#
#         feature, predict = Model(torch.tensor(image).resize(1, 3, 200, 200).to(torch.float32).cuda())
#
#         value, index = torch.max(predict, dim=1)
#         if index.item() == 0:
#             ADI[patch_image] = value.detach().cpu()
#         elif index.item() == 1:
#             DEB[patch_image] = value.detach().cpu()
#         elif index.item() == 2:
#             LYM[patch_image] = value.detach().cpu()
#         elif index.item() == 3:
#             MUC[patch_image] = value.detach().cpu()
#         elif index.item() == 4:
#             MUS[patch_image] = value.detach().cpu()
#         elif index.item() == 5:
#             NORM[patch_image] = value.detach().cpu()
#         elif index.item() == 6:
#             STR[patch_image] = value.detach().cpu()
#         elif index.item() == 7:
#             TUM[patch_image] = value.detach().cpu()
#         elif index.item() == 8:
#             BACK[patch_image] = value.detach().cpu()
#
#         coord_x = torch.tensor([int(patch_image.split('_')[0]), int((patch_image.split('_')[1]).split('.')[0]),
#                                 get_area(image)]).reshape([1, 3]).cuda()
#         final_feature = torch.cat((feature.detach(), coord_x), dim=1)
#
#         final_feature = feature.detach()
#         patch_feature[patch_image] = final_feature
#         print(patch_image)
#
#     patch_features = os.path.join(os.path.join(patches_path,step), '515-patch_feature.pkl')
#     torch.save(patch_feature, patch_features)
#     # sort dicitonary and choose 10%
#     ADI = sorted(ADI.items(), key=lambda x: x[1])[0:int(len(MUS) * 0.1)] if len(ADI) > 10 else sorted(ADI.items(),
#                                                                                                       key=lambda x: x[
#                                                                                                           1])
#     DEB = sorted(DEB.items(), key=lambda x: x[1])[0:int(len(MUS) * 0.1)] if len(DEB) > 10 else sorted(DEB.items(),
#                                                                                                       key=lambda x: x[
#                                                                                                           1])
#     LYM = sorted(LYM.items(), key=lambda x: x[1])[0:int(len(MUS) * 0.1)] if len(LYM) > 10 else sorted(LYM.items(),
#                                                                                                       key=lambda x: x[
#                                                                                                           1])
#     MUC = sorted(MUC.items(), key=lambda x: x[1])[0:int(len(MUS) * 0.1)] if len(MUC) > 10 else sorted(MUC.items(),
#                                                                                                       key=lambda x: x[
#                                                                                                           1])
#     MUS = sorted(MUS.items(), key=lambda x: x[1])[0:int(len(MUS) * 0.1)] if len(MUS) > 10 else sorted(MUS.items(),
#                                                                                                       key=lambda x: x[
#                                                                                                           1])
#     NORM = sorted(NORM.items(), key=lambda x: x[1])[0:int(len(MUS) * 0.1)] if len(NORM) > 10 else sorted(NORM.items(),
#                                                                                                          key=lambda x:
#                                                                                                          x[1])
#     STR = sorted(STR.items(), key=lambda x: x[1])[0:int(len(MUS) * 0.1)] if len(STR) > 10 else sorted(STR.items(),
#                                                                                                       key=lambda x: x[
#                                                                                                           1])
#     TUM = sorted(TUM.items(), key=lambda x: x[1])[0:int(len(MUS) * 0.1)] if len(TUM) > 10 else sorted(TUM.items(),
#                                                                                                       key=lambda x: x[
#                                                                                                           1])
#     BACK = sorted(BACK.items(), key=lambda x: x[1])[0:int(len(MUS) * 0.1)] if len(BACK) > 10 else sorted(BACK.items(),
#                                                                                                          key=lambda x:
#                                                                                                          x[1])
#     # iteratly cluster all patches
#     iteration = 5
#     centroid = ADI + DEB + LYM + MUC + MUS + NORM + STR + TUM + BACK
#     centroid_path = os.path.join(os.path.join(patches_path,step), '515-inital_centroid_patch_feature.pkl')
#     torch.save(centroid, centroid_path)
#
#     centroid_feature = [patch_feature[[i][0][0]] for i in centroid]
#     centroid_index = {}
#     for i in range(len(centroid)):
#         centroid_index[i] = []
#     for _ in range(iteration):
#         for patch in patch_feature:
#             mid = []
#             for i in range(len(centroid)):
#                 mid.append(torch.cosine_similarity(centroid_feature[i], patch_feature[patch]).item())
#             centroid_index[mid.index(max(mid))].append(patch)  # get every patch cluster label
#
#         for i in centroid_index:
#             mid = []
#             for j in centroid_index[i]:
#                 mid.append(patch_feature[j])
#             if len(mid) == 0:
#                 centroid[i] = sum(mid)
#             else:
#                 centroid[i] = sum(mid) / len(mid)
#
#     # save cluster index and feature
#     cluster_centroid_path = os.path.join(os.path.join(patches_path,step), '515-cluster_centroid_feature.pkl')
#     cluster_patches = os.path.join(os.path.join(patches_path,step), '515-cluster_patch_labels.pkl')
#
#     torch.save(centroid, cluster_centroid_path)
#     torch.save(centroid_index, cluster_patches)
#
#     print(step)