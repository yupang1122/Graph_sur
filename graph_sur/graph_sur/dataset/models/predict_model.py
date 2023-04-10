import torch.nn as  nn
import torch
import torch.nn.functional as F

class predict_model(nn.Module):

    def __init__(self,input_feature = 512):
        super(predict_model, self).__init__()
        self.classifier1 = nn.Linear(input_feature,32,bias = True)
        self.classifier2 = nn.Linear(32,9,bias=True)

    def forward(self, feature):
        fc = self.classifier1(feature)
        fc1 = self.classifier2(fc)

        value,index= torch.max(fc1,dim=1)
        return fc1,int(index)

