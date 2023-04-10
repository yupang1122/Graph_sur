
import torch.nn as nn
import torch
#import init
import torch.nn.functional as F
import torch.nn.init as init
import tensorflow as tf
import numpy as np
import copy
from torch.nn import LayerNorm,Dropout,Softmax,Linear,Conv2d

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta
        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim,output_dim).cuda())
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim).cuda())
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法
        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        support = torch.matmul(input_feature, self.weight)
        output = torch.matmul(adjacency, support)
        #output = torch.mm(adjacency,support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'

class risk_GCN(nn.Module):
    def __init__(self, input_dim,hidden_dim,num_layers, use_bias = True):
        """
            get risk output
        """
        super(risk_GCN,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        self.use_bias = use_bias
        self.layer = []
        self.Dropout = Dropout(0.3)

        self.norm1 = nn.BatchNorm1d(self.hidden_dim,momentum = 0.3)
        self.norm2 = nn.BatchNorm1d(16,momentum = 0.5)
        self.norm3 = LayerNorm(16,eps=0.001)
        self.norm4 = nn.BatchNorm1d(1,momentum = 0.5)

        for _ in range(num_layers):
            layer = GraphConvolution(self.input_dim,self.hidden_dim,self.use_bias)
            self.layer.append(copy.deepcopy(layer))

        self.atten1 = nn.Linear(input_dim,1)
        self.atten2 = nn.Linear(input_dim,1)

        self.final_layer = GraphConvolution(self.hidden_dim,16,self.use_bias)
        self.atten_layer = GraphConvolution(self.hidden_dim,1,self.use_bias)

        self.pred = GraphConvolution(16,1,self.use_bias)
        self.relu = nn.ReLU(inplace = False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, adjancency, feature):
        # different tissue value
        node_atten1 = self.atten1(feature)
        node_atten = self.atten1(feature)
        edge_atten = self.atten2(feature)
        # node_atten = self.norm4(self.atten1(feature))
        # edge_atten = self.norm4(self.atten2(feature))
        node_atten1 = torch.softmax(node_atten1, dim=0)
        feature = node_atten * feature
        adjancency = edge_atten * torch.tensor(adjancency).cuda()
        #
        hidden_feature = feature
        for layer in self.layer:
            last_feature = hidden_feature
            hidden_feature = layer(adjancency,hidden_feature)
            hidden_feature = self.tanh(self.Dropout(hidden_feature))
            hidden_feature = hidden_feature + last_feature

        aggregation_feature = self.final_layer(adjancency,hidden_feature)
        aggregation_feature = self.norm2(aggregation_feature)

        attention_matrix = self.atten_layer(adjancency,hidden_feature)

        attention_matrix = self.tanh(self.norm4(attention_matrix))

        pool_feature = torch.matmul(torch.transpose(attention_matrix,0,1), aggregation_feature)
        pool_feature = self.tanh(self.norm3(pool_feature))

        pool_matrix = torch.matmul(torch.transpose(attention_matrix,0,1), adjancency)
        pool_matrix =  torch.matmul(pool_matrix,attention_matrix)
        pool_matrix = self.tanh(pool_matrix)

        pred = self.pred(pool_matrix,pool_feature)
        prediction = pred

        return prediction,node_atten1

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'

class Survive_GCN(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers,use_bias = True):
        """
            input_dim : input_feature dimensity
            message: patient information includes : sex, age,cancer stage, metastase or primary .. and survive end.
        """
        super(Survive_GCN,self).__init__()
        self.input_dim = input_dim
        self.use_bias = use_bias
        # self.out_dim = 1 # hazard ratio by image
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim


        self.risk = risk_GCN(self.input_dim,self.hidden_dim,self.num_layers,self.use_bias)


    def forward(self,adjancency,input_feature):
        risk,attention = self.risk(adjancency,input_feature)


        return risk,attention


