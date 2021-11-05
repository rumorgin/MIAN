from pickle import LONG_BINPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.module import T
from pytorch_metric_learning.distances import LpDistance,CosineSimilarity
from pytorch_metric_learning import miners, losses
from .resnet import ResNet
from .proto_embedding import MultiHeadAttention_another,LinearClassifier,CenterLoss,Proto_Embedding

class Proto(nn.Module):

    def __init__(self, args, mode='meta'):
        super().__init__()

        self.mode = mode
        self.args = args

        self.encoder = ResNet(args=args,drop_rate=args.dropout)
        self.embedding=MultiHeadAttention_another(in_features=640,head_num=8,bias=True,activation=F.relu)
        self.miner = miners.MultiSimilarityMiner(distance=LpDistance(power=2))
        self.metric_loss_func = losses.TripletMarginLoss(distance=LpDistance(power=2))

        if self.mode == 'pre_train':
            self.fc = nn.Sequential(
                nn.Linear(640, 128),
                nn.Linear(128, self.args.num_class)
            )


    def forward(self, input):

        if self.mode == 'train_proto':
            data=self.encode(input)
            k = self.args.way * self.args.shot
            data_shot, data_query = data[:k], data[k:]
            logit,metric_loss= self.set_forward_loss(data_shot, data_query)
            return logit,metric_loss
        elif self.mode == 'pre_train':
            return self.pre_train_forward(input)
        else:
            raise ValueError('Unknown mode')

    def pre_train_forward(self, input):
        feature=self.encode(input, dense=False)
        feature=self.embedding(feature,feature,feature)
        return self.fc(feature)

    def encode(self, x, dense=True):

        if x.shape.__len__() == 5:  # batch of image patches
            num_data, num_patch = x.shape[:2]
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            x = self.encoder(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze()
            x = x.reshape(num_data, num_patch, -1)         
            return x

        else:
            x = self.encoder(x)
            if dense == False:
                x = F.adaptive_avg_pool2d(x, 1)
                return x
        return x

    def set_forward_loss(self, data_shot, data_query):

        
        data_shot=self.embedding(data_shot,data_shot,data_shot)
        data_query=self.embedding(data_query,data_query,data_query)

        if self.args.shot==5:
            data_shot=data_shot.reshape(self.args.shot,self.args.way,-1).permute(1,0,2)
            proto=data_shot.mean(1)
        elif self.args.shot==1:
            proto=data_shot
        else:
            raise ValueError('unsupport shot set')

        # some new metric learning loss
        if self.args.metric_learning:
            proto_label = torch.tensor([val for val in torch.arange(self.args.way) for i in range(self.args.shot*self.args.num_attention)])
            proto_label = proto_label.type(torch.LongTensor).cuda()
            hard_pairs = self.miner(data_shot.reshape(self.args.shot*self.args.way*self.args.num_attention,-1), proto_label)
            metric_loss = self.metric_loss_func(data_shot.reshape(self.args.shot*self.args.way*self.args.num_attention,-1), proto_label, hard_pairs)
        else:
            metric_loss=0

        logit = -1 * euclidean_dist(data_query, proto)
        # logit = -1*Hausdorff_Matrix(data_query,data_shot)
        return logit,metric_loss


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def sim_dist(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def Hausdorff_Matrix(test_bags, train_bags, use_gpu=True):
    w, h = len(test_bags), len(train_bags)
    if use_gpu:
        Matrix = torch.zeros(w, h).cuda()
    else:
        Matrix = torch.zeros(w, h)
    for i in range(0, w):
        for j in range(0, h):
            Matrix[i, j] = min_mean_hausdorff(test_bags[i], train_bags[j])

    return Matrix

def normal_hausdorff(X, Y):
    min_btoa, _ = torch.min(torch.cdist(X, Y), dim=0)
    min_atob, _ = torch.min(torch.cdist(Y, X), dim=0)
    Hausdorff_distance = torch.max(torch.max(min_btoa),
                                   torch.max(min_atob))
    return Hausdorff_distance

def min_hausdorff(X, Y):
    min_btoa, _ = torch.min(torch.cdist(X, Y), dim=0)
    min_atob, _ = torch.min(torch.cdist(Y, X), dim=0)
    Hausdorff_distance = torch.min(torch.max(min_btoa),
                                   torch.max(min_atob))
    return Hausdorff_distance

def min_mean_hausdorff(X, Y):
    min_btoa, _ = torch.min(torch.cdist(X, Y), dim=0)
    min_atob, _ = torch.min(torch.cdist(Y, X), dim=0)
    Hausdorff_distance = torch.max(torch.mean(min_btoa),
                                   torch.mean(min_atob))
    return Hausdorff_distance

def unidirection_mean_hausdorff(X, Y):
    min_atob, _ = torch.min(torch.cdist(Y, X), dim=0)
    Hausdorff_distance = min_atob.sort(descending=False).values[:min_atob.size(0)//2].mean()
    return Hausdorff_distance
