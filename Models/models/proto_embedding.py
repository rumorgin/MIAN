from torch import nn
from torch._C import device
import torch.nn.functional as F
import torch
import copy
import math
from collections import OrderedDict


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttention_another(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention_another, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)
        self.gathering = nn.Linear(10,1)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        y = y.permute(0,2,1)
        y = self.gathering(y)
        y = y.permute(0,2,1).squeeze()

        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.

        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, feature_dim, in_channel, dropout=0.1, out_channel=640):
        # Take in model size and number of heads.
        super(MultiHeadedAttention, self).__init__()
        assert feature_dim % h == 0

        self.d_k = feature_dim // h
        self.h = h
        self.linears = clones(nn.Linear(in_channel, in_channel), 3)
        self.linears.append(nn.Linear(out_channel, out_channel))
        self.collect = nn.Linear(in_channel,1)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        # batch_size
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # query, key, value = [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #                     for x in (query, key, value)]

        # 2) Apply attention on all the projected vectors in batch.
        # input=(n,h,l,d_model//h)
        x, self.attn = attention(query, key, value, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # output=(n,l,d_model)
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k)
        x= self.linears[-1](x).squeeze()
        x = x.transpose(1, 2)
        x= self.collect(x).squeeze()
        return x
        # return x


def attention(query, key, value, dropout=None):
    # "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class Proto_Embedding(nn.Module):
    def __init__(self, feat_dim, way):
        super(Proto_Embedding, self).__init__()

        self.attention = clones(nn.Sequential(
            nn.Linear(feat_dim, feat_dim//8),
            nn.Tanh(),
            nn.Linear(feat_dim//8, 5)
        ), way)
        self.way = way

    def forward(self, input):
        output = [None]*self.way
        for i in range(self.way):
            instance_i = input[i]
            atten = self.attention[i](instance_i)
            atten = torch.transpose(atten, 1, 0)
            atten = torch.softmax(atten, dim=1)

            M = torch.mm(atten, instance_i).squeeze()
            output[i] = M

        output = torch.stack(output, 0)

        return output


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.classifier(x)
        return out


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(
                self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        # x = (x - x.min()) / (x.max() - x.min() + 1e-8) * 2 - 1
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
            torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(
                self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        # classes = torch.arange(self.num_classes).long()
        # if self.use_gpu: classes = classes.cuda()
        # labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        # mask = labels.eq(classes.expand(batch_size, self.num_classes))

        mask = F.one_hot(labels)

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
