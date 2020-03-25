import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


class GroupLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, T=10, num_classes=751, num_anchors=0):
        super(GroupLoss, self).__init__()

        self.T = T
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.nllloss = nn.NLLLoss()
        # self.cross_entropy=nn.CrossEntropyLoss()

    def forward(self, features, X, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n, m = X.size()
        device = X.device
        # compute pearson r
        ff = features.clone().detach()
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff)).cpu().numpy()
        coef = np.corrcoef(ff)

        # features_ = features.detach().cpu().numpy()
        # coef = np.corrcoef(features_)

        diago = np.arange(coef.shape[0])
        coef[diago, diago] = 0
        # W = F.relu(torch.tensor((coef - np.diag(np.diag(coef))),
        #                         dtype=torch.float, device=device))
        W = F.relu(torch.tensor(coef,
                                dtype=torch.float, device=device))
        # print(W,'wwwwwwwwwwww')
        for i in range(n):
            if torch.sum(W[i]) == 0:
                # print(W,'wwwwwwwwwwww')

                W[i, i] = 1
                # print(W,'wwwwwwwwwwww')

        # print(W,'wwwwwwwww')
        X = F.softmax(X, dim=1)
        # print(X)
        # print(torch.argmax(X,dim=1))
        # ramdom select anchors
        ids = torch.unique(targets)
        # num_samples = n / len(ids)
        # print(X.dtype)
        # print(targets)
        # print(id(X))
        # X_=X.clone().detach()
        anchors = []
        for id_ in ids:
            anchor = list(np.random.choice(torch.where(targets == id_)[
                0].cpu(), size=self.num_anchors, replace=False))
            # print(id,'ididiid')
            # print(torch.sum(X[anchors]))
            # print(torch.argmax(X[anchors]))
            anchors += anchor

            # print(torch.argmax(X[anchors]))

        # print(X[:20,:5],'xxxxxxx')
        # print(id(X))
        # print(torch.where(X==torch.max(X,dim=1)))

        for i in range(self.T):
            X_ = X.clone().detach()
            X_[anchors] = torch.tensor(F.one_hot(
                targets[anchors], self.num_classes), dtype=torch.float, device=device)
            # print(i)
            # print(X,'xxxxxxxxxxxx')
            # print(X_,'---------')
            Pi = torch.mm(W, X_)
            # print(Pi)
            # print(Pi, 'pipipi')

            PX = torch.mul(X, Pi)

            # X = F.normalize(PX, dim=1, p=1)

            # print(PX,'pxpxpx')
            # print(PX.shape)

            # 111111111111111111111111
            # Norm = np.sum(PX.detach().cpu().numpy(),
            #               axis=1).reshape(-1)  # .expand(n,m)
            # # print(Norm,'norm')
            # Q = 1 / Norm
            # # print(Q,'QQQQQQQQQ')
            # Q = torch.diag(torch.tensor(Q, dtype=torch.float, device=device))

            # 2222222222222222222222222
            # denom = PX.detach().norm(p=1, dim=1, keepdim=True).clamp_min(1e-12).expand_as(PX)
            # X=PX/denom

            # 3333333333333333333333
            # Q = torch.diag(1 / PX.norm(p=1, dim=1).clamp_min(1e-12))
            Q = torch.diag(1 / PX.detach().norm(p=1, dim=1).clamp_min(1e-12))
            X = torch.mm(Q, PX)

            # 444444444444444444444444444444
            # Q = torch.diag(1 / torch.matmul(
            #     PX, torch.ones(m, dtype=torch.float, device=device)))
            # print(Q,'qqqqq')
            # X = torch.matmul(Q, PX)
            # Q=torch.pow(Q,-1)
            # print(X)

            # 555555555555555555555555555555555555
        # X = F.softmax(PX, dim=1)

        # print(X.requires_grad)
        loss = self.nllloss(torch.log(X.clamp_min(1e-12)), targets)

        # loss= self.cross_entropy(X,targets)
        return loss

        # #inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # # Compute pairwise distance, replace by the official when merged
        # dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # # For each anchor, find the hardest positive and negative
        # mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        # print(mask[:8, :8])
        # dist_ap, dist_an = [], []
        # for i in range(n):
        #     dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
        #     dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        # dist_ap = torch.cat(dist_ap)
        # dist_an = torch.cat(dist_an)
        # # Compute ranking hinge loss
        # y = torch.ones_like(dist_an)
        # loss = self.ranking_loss(dist_an, dist_ap, y)
        # if self.mutual:
        #     return loss, dist
        # return loss
