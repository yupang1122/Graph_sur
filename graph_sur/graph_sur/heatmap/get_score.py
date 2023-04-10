

import os
import openslide
import torch
import cv2
import numpy as np
from resnet import *
import pandas as pd
import matplotlib.pyplot as plt

patches_path = './a6-6648'

thumbnail1 = cv2.imread(os.path.join(patches_path, '0', 'atten_thumbnail.png'))
thumbnail2 = cv2.imread(os.path.join(patches_path, '20', 'atten_thumbnail.png'))
thumbnail3 = cv2.imread(os.path.join(patches_path, '40', 'atten_thumbnail.png'))
thumbnail4 = cv2.imread(os.path.join(patches_path, '60', 'atten_thumbnail.png'))
thumbnail5 = cv2.imread(os.path.join(patches_path, '80', 'atten_thumbnail.png'))
thumbnail6 = cv2.imread(os.path.join(patches_path, '100', 'atten_thumbnail.png'))
thumbnail7 = cv2.imread(os.path.join(patches_path, '120', 'atten_thumbnail.png'))
thumbnail8 = cv2.imread(os.path.join(patches_path, '140', 'atten_thumbnail.png'))
thumbnail9 = cv2.imread(os.path.join(patches_path, '160', 'atten_thumbnail.png'))
thumbnail10 = cv2.imread(os.path.join(patches_path, '180', 'atten_thumbnail.png'))
thumbnail11 = cv2.imread(os.path.join(patches_path, '10', 'atten_thumbnail.png'))
thumbnail12 = cv2.imread(os.path.join(patches_path, '30', 'atten_thumbnail.png'))
thumbnail13 = cv2.imread(os.path.join(patches_path, '50', 'atten_thumbnail.png'))
thumbnail14 = cv2.imread(os.path.join(patches_path, '70', 'atten_thumbnail.png'))
thumbnail15 = cv2.imread(os.path.join(patches_path, '90', 'atten_thumbnail.png'))
thumbnail16 = cv2.imread(os.path.join(patches_path, '110', 'atten_thumbnail.png'))
thumbnail17 = cv2.imread(os.path.join(patches_path, '130', 'atten_thumbnail.png'))
thumbnail18 = cv2.imread(os.path.join(patches_path, '150', 'atten_thumbnail.png'))
thumbnail19 = cv2.imread(os.path.join(patches_path, '170', 'atten_thumbnail.png'))
thumbnail20 = cv2.imread(os.path.join(patches_path, '190', 'atten_thumbnail.png'))
thumbnail = [thumbnail1, thumbnail2, thumbnail3, thumbnail4, thumbnail5, thumbnail6, thumbnail7, thumbnail8, thumbnail9,
             thumbnail10, thumbnail11, thumbnail12, thumbnail13, thumbnail14, thumbnail15, thumbnail16, thumbnail17,
             thumbnail18, thumbnail19, thumbnail20]

target = thumbnail1
target1 = np.zeros([len(thumbnail1), len(thumbnail1[0]), 1])

for i in range(len(target)):
    for j in range(len(target[0])):
        if target[i, j][1] == 0:
            red = target[i, j][2] / 100
            num1 = 1 if red != 0 else 0.001

            blue = target[i, j][0] / 100
            num2 = 1 if blue != 0 else 0.001

            for index, image in enumerate(thumbnail):
                step = (index + 1) * 20
                if image[i, j][1] == 0:
                    # if image[i,j][0] !=0:
                    #     blue = blue + (image[i,j][0])/100
                    #     num2 = num2 +1
                    #
                    # if image[i,j][2] != 0:
                    #     red  = red + (image[i,j][2])/100
                    #     num1 = num1 +1
                    red = red + (image[i, j][2]) / 100
                    num1 = num1 + 1

            target1[i, j] = (red / num1)
            print(target1[i, j])
            # if  0.8<(red/num1)<=1:
            #     target[i, j] = [0 , 0 ,(red/num1)*255]
            # elif    0.6<(red/num1)<=0.8:
            #     target[i, j] = [(1-(red / num1))*255, (1-(red / num1))*128, (red / num1) * 255]
            # elif    0.4<(red/num1)<=0.6:
            #     target[i, j] =[(1-(red / num1))*255, (1-(red / num1))*128, (red / num1) * 255]
            # elif    0.2<(red/num1)<=0.4:
            #     target[i, j] = [(1-(red / num1))*255, (1-(red / num1))*128, (red / num1) * 255]
            # else:
            #     target[i, j] = [(1-(red / num1))*255, (1-(red / num1))*128, (red / num1) * 255]
target1 *= 100
final_path = os.path.join(patches_path, 'score_final_visualization.png')
cv2.imwrite(final_path, target1)