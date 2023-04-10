import os
import cv2
import numpy as np
from resnet import *
import torch
from models import model_clam
from models.resnet_custom import resnet50_baseline
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from models.Transformer import Transformer
import ml_collections


def get_area(image):
    res = 0
    if image.max()<230:
        res = 1

    return res



if __name__ == '__main__':

    svs_path = './segment_cell/final-visualization'
    #2-class
    model_path = './model_clam/2-class-515-pretrain_transformer0.7227722772277227.pkl'
    # model_path = './model_clam/515-pretrain_clam.pkl'
    #4-class
    # model_path = './model_clam/4-class-pretrain_transformer0.6717948717948717.pkl'
    # model_path = './model_clam/pretrain_4-class0.6723549488054608_clam.pkl'
    svs_files = os.listdir(svs_path)
    # clam
    # model_dict = {'n_classes': 2,'size_arg' : 'small','dropout': 'True'}
    # model = model_clam.CLAM_SB(**model_dict)
    # model.to(device)
    # parmeter = torch.load(model_path,map_location='cuda:0')
    # new_pretrained_dict = {}
    # for k, v in parmeter['net'].items():
    #     if k.split('.')[0] == 'module':
    #         new_pretrained_dict[k[7:]] = v
    #     else:
    #         new_pretrained_dict[k] = v
    # model.load_state_dict(new_pretrained_dict, strict=True)
    # model.relocate()
    # model.eval()

#model_transformer
    def get_first_config():
        config = ml_collections.ConfigDict()
        config.patches = ml_collections.ConfigDict({'size': (16, 16)})
        config.hidden_size = 515
        config.transformer = ml_collections.ConfigDict()
        config.transformer.mlp_dim = 1024
        config.transformer.num_heads = 5
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

    model_dict = get_transformer_config()
    model = Transformer(model_dict,num_layer = 5).cuda()
    parmeter = torch.load(model_path)['net']
    model.load_state_dict(parmeter, strict=True)
    print(model)
    model.eval()




    for i in svs_files:
        print(i)
        if i.split('.')[-1]!='svs':
            clusters_path = os.path.join(svs_path,i,'515-cluster_patch_labels.pkl')
            patch_feature_path = os.path.join(svs_path,i,'515-patch_feature.pkl')
            cluster_patches = torch.load(clusters_path)
            patch_features = torch.load(patch_feature_path)

            slide_feature = []
            slide_feature_labels = []

            for cluster in cluster_patches:
                patches = cluster_patches[cluster]
                cluster_feature = []
                for patch in patches:
                    patch_path = os.path.join(svs_path,i,patch)
                    feature = patch_features[patch]
                    coord_x = torch.tensor([int(patch.split('_')[0]), int((patch.split('_')[1]).split('.')[0]),get_area(cv2.imread(patch_path))]).reshape([1, 3]).cuda()
                    feature = torch.cat((feature.detach(), coord_x), dim=1)
                    cluster_feature.append(feature[0].cpu().numpy())

                cluster_feature = torch.tensor(cluster_feature).squeeze().cuda()
                #clam
                # logits, Y_prob, Y_hat, A, _ = model(cluster_feature)
                # slide_feature.append(logits)
                # slide_feature_labels.append(cluster)
                #transformer
                attn_weight, feature, final = model(cluster_feature)
                attn_weight = sum(attn_weight)/len(attn_weight)
                final = torch.matmul(torch.mean(torch.mean(attn_weight, dim=1), dim=1).unsqueeze(dim=0), final)
                slide_feature.append(final.detach().cpu().numpy())
                slide_feature_labels.append(cluster)
                #max
                # cluster_feature,_ = torch.max(cluster_features,dim=0)
                # cluster_feature = torch.unsqueeze(cluster_feature,dim=0)
                # slide_feature.append(cluster_feature)
                # slide_feature_labels.append(cluster)
                #mean
                # cluster_feature = torch.mean(cluster_features)
                # slide_feature.append(cluster_feature)
                # slide_feature_labels.append(cluster)
            slide_feature_path = os.path.join(svs_path,i,'515_rnn_slide_feature.pkl')
            slide_feature_label_path = os.path.join(svs_path,i,'515_rnn_slide_feature_label.pkl')


            torch.save(torch.tensor(slide_feature).squeeze(),slide_feature_path)
            torch.save(slide_feature_labels,slide_feature_label_path)

            # for cluster in clusters:
            #     if cluster.isalnum():
            #         feature_path = os.path.join(clusters_path,cluster,'feature.pkl')
            #         if os.path.exists(feature_path):
            #             cluster_features = torch.load(feature_path).cuda()
            #             #clam
            #             # logits, Y_prob, Y_hat, A, _ = model(cluster_features)
            #             # torch.save(logits, os.path.join(clusters_path, cluster, 'clam_feature.pkl'))
            #             #transformer
            #             attn_weight, feature, final = model(cluster_features)
            #             attn_weight = sum(attn_weight)/len(attn_weight)
            #             final = torch.matmul(torch.mean(torch.mean(attn_weight, dim=1), dim=1).unsqueeze(dim=0), final)
            #             torch.save(final,os.path.join(clusters_path, cluster, 'transformer_feature.pkl'))
            #             #max
            #             # cluster_feature,_ = torch.max(cluster_features,dim=0)
            #             # cluster_feature = torch.unsqueeze(cluster_feature,dim=0)
            #             # torch.save(cluster_feature, os.path.join(clusters_path, cluster, 'max_feature.pkl'))
            #             #mean
            #             # cluster_feature = torch.mean(cluster_features)
            #             # torch.save(cluster_feature, os.path.join(clusters_path, cluster, 'mean_feature.pkl'))
            #
            #             print(cluster)
            # segmentation_path = os.path.join(svs_path,i,'segmentation.png')
            # coord_list_path = os.path.join(svs_path,i,'coord_list.pth')
            # feature_path = os.path.join(svs_path,i,'feature')
            # Image_path = os.path.join(svs_path,i,'cluster.png')
            # Image = cv2.imread(Image_path)
            # segmentation = cv2.imread(segmentation_path,0)
            # for cluster in range(2,int(segmentation.max())):
            #     cluster_path = os.path.join(svs_path,i,str(cluster))
            #     os.mkdir(cluster_path)
            #     cluster_feature_path = os.path.join(cluster_path,'cluster_feature.pth')
            #     feature_list = []
            #     coord_list = []
            #     for x in range(segmentation.shape[0]):
            #         for y in range(segmentation.shape[1]):
            #             if segmentation[x,y] == cluster:
            #                 image = str(x) + '_' + str(y) +'.jpg'
            #                 image_path = os.path.join(svs_path,i,image)
            #                 feature = Model(torch.tensor(cv2.imread(image_path)).resize(1,3,200,200).to(torch.float32).cuda()).cpu().detach().numpy() #512
            #                 #insert feature
            #                 feature_list.append(feature)
            #                 coord_list.append([x,y])
            #     if len(feature_list) == 1:
            #         continue
            #     logits, Y_prob, Y_hat, A, _ = model(torch.tensor(feature_list).squeeze().cuda()) # input feature is 1024
            #     #########
            #     torch.save(logits,cluster_feature_path)
            #     # A = A.view(-1, 1).detach().cpu().numpy()
            #     # n = len(coord_list)
            #     # patch_size = 200
            #     # for k in range(n):
            #     #     precentage = (A[k]-min(A))/(max(A)-min(A)) if max(A) != min(A) else 1
            #     #     image_name = str(coord_list[k][0]) + '_' + str(coord_list[k][1]) + '.jpg'
            #     #     image_path = os.path.join(svs_path,i,image_name)
            #     #
            #     #     patch_image = cv2.imread(image_path)
            #     #     r, g, b = cv2.split(patch_image)
            #     #     g = np.full([200, 200], int(255*(1 - precentage)), dtype=np.uint8)
            #
            #         # Image[(coord_list[k][0] * patch_size):(coord_list[k][0] + 1) * patch_size, (coord_list[k][1] * patch_size):(coord_list[k][1] + 1) * (patch_size)] = cv2.merge([r,g,b])
            #     # cluster_path = os.path.join(svs_path,i,str(cluster),'cluster.png')
            #     # cv2.imwrite(cluster_path, Image)
            #     print('cluster_number: ' + str(cluster))
