import torch
import torch.nn as nn
import numpy as np

import copy
import math

from torch.nn import LayerNorm,Dropout,Softmax,Linear,Conv2d

def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Attention(nn.Module):
    def __init__(self,config):
        super(Attention,self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # self.all_head_size = 515

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self,config):
        super(Mlp,self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self,config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self,x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class transformer(nn.Module):

    def __init__(self,config,num_layer=1):
        super(transformer,self).__init__()
        self_layers = num_layer
        self.layer = []
        # self.layer = nn.ModuleList
        self.attn_norm = LayerNorm(config.hidden_size,eps=1e-6)
        for _ in range(self_layers):
            layer = Block(config).cuda()
            self.layer.append(copy.deepcopy(layer))

    def forward(self,input_feature):
        attn_weight = []
        for layer_block in self.layer:
            input_feature,weight = layer_block(input_feature)
            attn_weight.append(weight)
        encoded = self.attn_norm(input_feature)
        return encoded,attn_weight


class Transformer(nn.Module):
    def __init__(self,config,num_layer = 1):
        super(Transformer, self).__init__()
        self.transformer = transformer(config,num_layer)
        self.classifier = Linear(config.hidden_size,2)
        self.att_class1 = Linear(config.hidden_size//config.transformer['num_heads'],1)
        self.att_class2 = Linear(config.hidden_size//config.transformer['num_heads'],1)
        self.Rnn = nn.ReLU(inplace=True)


    def forward(self,x):

        x,attn_weight = self.transformer(x)
        feature = x
        final = self.classifier(x)

        att = torch.squeeze(self.att_class1(attn_weight[0]),dim=2)
        att = self.att_class2(att)

        final = torch.matmul(att.transpose(1,0),final)

        return attn_weight,final,feature

