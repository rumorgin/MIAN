import torch
from torch import nn
from collections import OrderedDict

class Self_Supervise(nn.Module):
    def __init__(self, feat_dim, n_instance_in_bag):
        super(Self_Supervise, self).__init__()

        self.fc6 = nn.Sequential(OrderedDict([
            ('fc6_s1', nn.Linear(feat_dim, feat_dim)),
            ('relu6_s1', nn.ReLU(inplace=True)),
            ('drop6_s1', nn.Dropout(p=0.5)),
        ]))

        self.fc7 = nn.Sequential(OrderedDict([
            ('fc7', nn.Linear(n_instance_in_bag * feat_dim, 1280)),
            ('relu7', nn.ReLU(inplace=True)),
            ('drop7', nn.Dropout(p=0.5))
        ]))

        # self.classifier = nn.Linear(4096, 35) #1000: number of permutation
        self.classifier = nn.Sequential(nn.Linear(1280,128),nn.LeakyReLU(0.2),nn.Linear(128,35))

    def forward(self, input, B, T):
        x_ = input.view(B, T, -1)
        x_ = x_.transpose(0, 1)  # torch.Size([9, 75, 512])

        x_list = []
        for i in range(9):
            z = self.fc6(x_[i])  # torch.Size([75, 512])
            z = z.view([B, 1, -1])  # torch.Size([75, 1, 512])
            x_list.append(z)

        x_ = torch.cat(x_list, 1)  # torch.Size([75, 9, 512])
        x_ = self.fc7(x_.view(B, -1))  # torch.Size([75, 9*512])
        x_ = self.classifier(x_)
        x_ = nn.functional.softmax(x_,dim=1)
        return x_