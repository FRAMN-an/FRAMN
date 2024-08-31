import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbones import Conv_4, ResNet
import math

def pdist(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    # x_norm = (x**2).sum(1).view(-1, 1)
    # y_t = torch.transpose(y, 0, 1)
    # y_norm = (y**2).sum(1).view(1, -1)
    # dist = x_norm + y_norm

    # n = x.size(0) #450
    # m = y.size(0) #30
    # d = x.size(1) #1600
    # assert d == y.size(1)
    # x = x.unsqueeze(1).expand(n,m,d)
    # y = y.unsqueeze(0).expand(n,m,d)

    dis1 = torch.pow(x - y, 2).sum(2)
    dis2 = torch.sqrt(torch.sum(x * y, dim=2))
    dist = torch.sqrt(dis1 + dis2)

    return dis1


class FRAMN(nn.Module):

    def __init__(self, way=None, shots=None, resnet=False):

        super().__init__()

        if resnet:
            # self.dim = 640
            num_channel = 640
            self.dim = [num_channel, 5, 5]
            self.feature_extractor = ResNet.resnet12()
            self.ca_module = CA_Module(num_channel, reduction=4)
        else:
            num_channel = 64
            self.feature_extractor = Conv_4.BackBone(num_channel)
            self.dim = [num_channel, 5, 5]
            # self.dim = num_channel*5*5
            self.ca_module = CA_Module(num_channel, reduction=4)

        self.shots = shots
        self.way = way
        self.resnet = resnet
        self.loss_type = 'mse'
        if self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        self.layer = Layer(self.dim, 8, self.loss_type)

        # temperature scaling, correspond to gamma in the paper
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def get_feature_vector(self, inp):

        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)
        feature_map = self.ca_module(feature_map)
        feature_map = self.layer(feature_map)

        if self.resnet:
            # feature_map = F.avg_pool2d(input=feature_map,kernel_size=feature_map.size(-1))
            feature_vector = feature_map.view(batch_size, *self.dim)
        else:
            feature_vector = feature_map.view(batch_size, *self.dim)

        return feature_vector

    def get_neg_l2_dist(self, inp, way, shot, query_shot):
        self.sigmoid = nn.Sigmoid()

        feature_vector = self.get_feature_vector(inp)

        support = feature_vector[:way * shot].view(way, shot, *self.dim)
        centroid = torch.mean(support, 1)  # way,dim
        query = feature_vector[way * shot:]  # way*query_shot,dim
        # support_mask = centroid.sum(1).view(way,1,5,5)
        # query_mask = query.sum(1).view(way*query_shot,1,5,5)


        support_mask = torch.sum(centroid, 1).view(way, -1)
        support_mask = self.sigmoid(support_mask)
        support_mask = support_mask.view(way, 1, 5, 5)

        query_mask = torch.sum(query, 1).view(way*query_shot, -1)
        query_mask = self.sigmoid(query_mask)
        query_mask = query_mask.view(way*query_shot, 1, 5, 5)

        sa_support_m = support_mask.unsqueeze(0).repeat(way * query_shot, 1, 1, 1, 1)
        sa_query_m = query_mask.unsqueeze(1).repeat(1, way, 1, 1, 1)
        sq_mask = sa_support_m * sa_query_m  # [450,30,1,5,5]
        s_feat = centroid.unsqueeze(0).repeat(way * query_shot,1,1,1,1)
        q_feat = query.unsqueeze(1).repeat(1,way,1,1,1)
        sa_s_feat = (sq_mask * s_feat).view(way * query_shot, way, -1)
        sa_q_feat = (sq_mask * q_feat).view(way * query_shot, way, -1)

        neg_l2_dist = pdist(sa_q_feat, sa_s_feat).neg()  # way*query_shot,way

        return neg_l2_dist

    def meta_test(self, inp, way, shot, query_shot):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                           way=way,
                                           shot=shot,
                                           query_shot=query_shot)

        _, max_index = torch.max(neg_l2_dist, 1)

        return max_index

    def forward(self, inp):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                           way=self.way,
                                           shot=self.shots[0],
                                           query_shot=self.shots[1])
        if self.resnet:
            logits = neg_l2_dist / 640 * self.scale
        else:
            logits = neg_l2_dist / 64 * self.scale
        # log_prediction = F.log_softmax(logits,dim=1)

        return logits



class CA_Module(nn.Module):
    def __init__(self, channels, reduction):
        super(CA_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        channel_feature = module_input * x
        return channel_feature


class Layer(nn.Module):
    def __init__(self, input_size, hidden_size, loss_type='mse'):
        super(Layer, self).__init__()
        self.loss_type = loss_type
        # padding = 1 if (input_size[1] < 10) and (input_size[2] < 10) else 0
        # when using Resnet, conv map without avgpooling is 7x7, need padding in block to do pooling
        self.layer1 = RelationConvBlock(input_size[0], input_size[0], padding=1)
        self.layer2 = RelationConvBlock(input_size[0], input_size[0], padding=1)  # self.input = [400,64,15,15]

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out

class RelationConvBlock(nn.Module):
    def __init__(self, indim, outdim, padding=0):
        super(RelationConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
        self.BN = nn.BatchNorm2d(outdim, momentum=1, affine=True)
        self.relu = nn.ReLU()
        self.pool   = nn.MaxPool2d(2)
        self.parametrized_layers = [self.C, self.BN, self.relu]
        for layer in self.parametrized_layers:
            backbone.init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


class backbone(nn.Module):
    def init_layer(L):
        # Initialization using fan-in
        if isinstance(L, nn.Conv2d):
            n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
            L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
        elif isinstance(L, nn.BatchNorm2d):
            L.weight.data.fill_(1)
            L.bias.data.fill_(0)
