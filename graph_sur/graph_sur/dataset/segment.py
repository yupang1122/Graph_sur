import os
import openslide
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from resnet import *
import random
from skimage.segmentation import slic
from skimage import color



svs_files  = './stage'
patch_path = './stage_patches'
patch_size = 200
svs_list = os.listdir(svs_files)

def filter(patch_mask):
    nuclei_num = 0
    for i in range(patch_mask.shape[0]):
        for j in range(patch_mask.shape[1]):
            # patch_mask[i][j] = 0 if patch_mask[i][j] == 255 else 1
            if patch_mask[i][j] != 255:
                nuclei_num +=1

    return nuclei_num


for svs_file in svs_list :
    if svs_file.split('.')[-1] == 'svs':
        svs_path = os.path.join(svs_files,svs_file)
        SVS = openslide.OpenSlide(svs_path)
        patchs_path = os.path.join(patch_path,svs_file.split('.')[0])
        if os.path.exists(patchs_path) != True:
            os.mkdir(patchs_path)

        print(svs_file)
        thumbnail = SVS.get_thumbnail(SVS.level_dimensions[1])
        get_Image = cv2.cvtColor(np.asarray(thumbnail), cv2.COLOR_RGB2BGR)
        gray_Image = cv2.cvtColor(get_Image, cv2.COLOR_BGR2GRAY)
        ther, mask = cv2.threshold(gray_Image, 0, 255, cv2.THRESH_OTSU)
        new_mask = np.zeros([int(get_Image.shape[0] / patch_size),int(get_Image.shape[1] / patch_size)]) #segmentation mask

        #stage 2 random position
        # patch_images = []
        for i in range(int(get_Image.shape[0] / patch_size)):
            for j in range(int(get_Image.shape[1] / patch_size)):
                patch = get_Image[(i * patch_size):(i + 1) * patch_size, (j * patch_size):(j + 1) * (patch_size)]
                patch_mask = mask[(i * patch_size):(i + 1) * patch_size, (j * patch_size):(j + 1) * (patch_size)]
                nuclei_num = filter(patch_mask)
                if nuclei_num >= int(patch_size * patch_size / 10):
                    patch_name = str(i) + '_' + str(j) + '.jpg'
                    # patch_images.append(patch_name)
                    patches_path = os.path.join(patchs_path, patch_name)
                    cv2.imwrite(patches_path, patch)
                    print(patches_path)
                else:
                    get_Image[(i * patch_size):(i + 1) * patch_size, (j * patch_size):(j + 1) * (patch_size)] = [0,0,0]

        cv2.imwrite(os.path.join(patchs_path,'filter.png'),get_Image)
        cv2.imwrite(os.path.join(patchs_path,'thumbnail.png'),np.asarray(thumbnail))




