import torch.nn as nn
import torch
#import init
import torch.nn.functional as F
import torch.nn.init as init
import tensorflow as tf
import numpy as np
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
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
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


# 模型定义
# 读者可以自己对GCN模型结构进行修改和实验
class GcnNet(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型
    """

    def __init__(self, input_dim = 512 ):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 256)
        self.gcn2 = GraphConvolution(256, 16)
        self.gcn3 = GraphConvolution(16, 9)
        self.lnn = nn.Linear(in_features=9 ,out_features=1)
        self.rnn = nn.Linear(in_features=9, out_features=4)
        self.Rnn = nn.ReLU(inplace=True)

        self.Batchnorm1 = nn.BatchNorm1d(num_features = 16,momentum = 0.3)
        self.Batchnorm2 = nn.BatchNorm1d(num_features = 9,momentum = 0.3)
        self.Batchnorm3 = nn.BatchNorm1d(num_features=256,momentum=0.3)
    #    self.Fit_layer = nn.Linear(in_features=10, out_features=1)
    def forward(self, adjacency, feature):
        h = F.relu(self.Batchnorm3(self.gcn1(adjacency, feature)))
        logits = F.relu(self.Batchnorm1(self.gcn2(adjacency,h)))
        logits = self.Batchnorm2(self.gcn3(adjacency, logits))
        logit = self.lnn(logits)
        logit1 = torch.matmul(logit.transpose(1,0),logits)
        logit2 = self.rnn(logit1)
    #    logit2 = self.Rnn(self.rnn(logit1))

#        result = logit2.transpose(0,1)
        result = logit2
        return result,h
 #        final = result.cpu().detach().numpy()
 # #       final = tf.concat([clincial, final], 1)
 # #       final = result[0].cpu().detach().numpy()
 #        final = torch.tensor(np.concatenate((clincial,final))).to('cuda').to(torch.float32)
 #        fit = self.Rnn((self.Fit_layer(final)).to(torch.float32))
 #
 #        fit = torch.tensor(fit,requires_grad=True)
 #        return fit