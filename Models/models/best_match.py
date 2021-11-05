import torch

support=torch.randn(5,10,640)
query=torch.randn(25,10,640)

w=len(support)
h=len(query)

dist=torch.zeros(h,w)

for i in range(h):
    for j in range(w):
        X=query[i]
        Y=support[j]
        min_btoa, _ = torch.min(torch.cdist(X, Y), dim=0)
        min_atob, _ = torch.min(torch.cdist(Y, X), dim=0)
        dist[i,j] = torch.max(torch.mean(min_btoa),
                                       torch.mean(min_atob))

