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

parser.add_argument('--batch_size',default = 10,type = int)
parser.add_argument('--epoch',default = 10 ,type = int)
parser.add_argument('--feature_path',default = './stage_patches',type = str)
parser.add_argument('--model_path',default = './256-survive_model.pkl')
args = parser.parse_args()

slides = os.listdir(args.feature_path)
# message = {}
#   cross validation
cross_validate = 10
train_set = random.sample(slides, int(len(slides) * 0.8))
valid_set = random.sample(slides, int(len(slides) * 0.1))
test_set = random.sample(slides,int(len(slides)*0.1))

model = torch.load(args.model_path)
model.eval()



for slide in test_set:
    correct_number = 0
    patient_name = slide
    if len(text.loc[text['ID'] == patient_name[0:12]]) > 0:
        patient_risk = message[patient_name]['risk']
        for target in test_set:
            if len(text.loc[text['ID'] == target[0:12]]) > 0:
                target_risk = message[target]['risk']
                if (message[patient_name]['survive_time'] >= message[target][
                    'survive_time'] and patient_risk >= target_risk) or (
                        message[patient_name]['survive_time'] <= message[target][
                    'survive_time'] and patient_risk <= target_risk):
                    correct_number += 1

    print(str(correct_number / len(test_set)))