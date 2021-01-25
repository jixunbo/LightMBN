"""The Group Loss for Deep Metric Learning

Reference:
Elezi et al. The Group Loss for Deep Metric Learning. ECCV 2020.

Code adapted from https://github.com/dvl-tum/group_loss

"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def dynamics(W, X, tol=1e-6, max_iter=5, mode='replicator', **kwargs):
    """
    Selector for dynamics
    Input:
    W:  the pairwise nxn similarity matrix (with zero diagonal)
    X:  an (n,m)-array whose rows reside in the n-dimensional simplex
    tol:  error tolerance
    max_iter:  maximum number of iterations
    mode: 'replicator' to run the replicator dynamics
    """

    if mode == 'replicator':
        X = _replicator(W, X, tol, max_iter)
    else:
        raise ValueError('mode \'' + mode + '\' is not defined.')

    return X


def _replicator(W, X, tol, max_iter):
    """
    Replicator Dynamics
    Output:
    X:  the population(s) at convergence
    i:  the number of iterations needed to converge
    prec:  the precision reached by the dynamical system
    """

    i = 0
    while i < max_iter:
        X = X * torch.matmul(W, X)
        X /= X.sum(dim=X.dim() - 1).unsqueeze(X.dim() - 1)
        i += 1

    return X


class GroupLoss(nn.Module):
    def __init__(self, total_classes, tol=-1., max_iter=5, num_anchors=3, tem=79, mode='replicator', device='cuda:0'):
        super(GroupLoss, self).__init__()
        self.m = total_classes
        self.tol = tol
        self.max_iter = max_iter
        self.mode = mode
        self.device = device
        self.criterion = nn.NLLLoss().to(device)
        self.num_anchors = num_anchors
        self.temperature = tem

    def _init_probs_uniform(self, labs, L, U):
        """ Initialized the probabilities of GTG from uniform distribution """
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = 1. / self.m
        ps[L, labs] = 1.

        # check if probs sum up to 1.
        assert torch.allclose(ps.sum(dim=1), torch.ones(n).cuda())
        return ps

    def _init_probs_prior(self, probs, labs, L, U):
        """ Initiallized probabilities from the softmax layer of the CNN """
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = probs[U, :]
        ps[L, labs] = 1.

        # check if probs sum up to 1.
        assert torch.allclose(ps.sum(dim=1), torch.ones(n).cuda())
        return ps

    def _init_probs_prior_only_classes(self, probs, labs, L, U, classes_to_use):
        """ Different version of the previous version when it considers only classes in the minibatch,
            might need tuning in order to reach the same performance as _init_probs_prior """
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = probs[torch.meshgrid(
            torch.tensor(U), torch.from_numpy(classes_to_use))]
        ps[L, labs] = 1.
        ps /= ps.sum(dim=ps.dim() - 1).unsqueeze(ps.dim() - 1)
        return ps

    def set_negative_to_zero(self, W):
        return F.relu(W)

    def _get_W(self, x):

        x = (x - x.mean(dim=1).unsqueeze(1))
        norms = x.norm(dim=1)
        W = torch.mm(x, x.t()) / torch.ger(norms, norms)

        W = self.set_negative_to_zero(W.cuda())
        return W

    def get_labeled_and_unlabeled_points(self, labels, num_points_per_class, num_classes=100):
        labs, L, U = [], [], []
        labs_buffer = np.zeros(num_classes)
        num_points = labels.shape[0]
        for i in range(num_points):
            if labs_buffer[labels[i]] == num_points_per_class:
                U.append(i)
            else:
                L.append(i)
                labs.append(labels[i])
                labs_buffer[labels[i]] += 1
        return labs, L, U

    def forward(self, fc7, labels, probs, classes_to_use=None):
        # print(fc7)
        # print(type(fc7))
        # print(labels)
        # print(type(labels))
        # print(probs)
        # print(type(probs))
        probs = F.softmax(probs / self.temperature)
        labs, L, U = self.get_labeled_and_unlabeled_points(
            labels, self.num_anchors, self.m)
        W = self._get_W(fc7)
        if type(probs) is type(None):
            ps = self._init_probs_uniform(labs, L, U)
        else:
            if type(classes_to_use) is type(None):
                ps = probs
                ps = self._init_probs_prior(ps, labs, L, U)
            else:
                ps = probs
                ps = self._init_probs_prior_only_classes(
                    ps, labs, L, U, classes_to_use)
        ps = dynamics(W, ps, self.tol, self.max_iter, self.mode)
        probs_for_gtg = torch.log(ps + 1e-12)
        loss = self.criterion(probs_for_gtg, labels)
        return loss


# import torch
# from torch import nn
# import torch.nn.functional as F

# import numpy as np


# class GroupLoss(nn.Module):
#     """Triplet loss with hard positive/negative mining.

#     Reference:
#     Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

#     Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

#     Args:
#         margin (float): margin for triplet.
#     """

#     def __init__(self, T=10, num_classes=751, num_anchors=0):
#         super(GroupLoss, self).__init__()

#         self.T = T
#         self.num_classes = num_classes
#         self.num_anchors = num_anchors
#         self.nllloss = nn.NLLLoss()
#         # self.cross_entropy=nn.CrossEntropyLoss()

#     def forward(self, features, X, targets):
#         """
#         Args:
#             inputs: feature matrix with shape (batch_size, feat_dim)
#             targets: ground truth labels with shape (num_classes)
#         """
#         n, m = X.size()
#         device = X.device
#         # compute pearson r
#         ff = features.clone().detach()
#         fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
#         ff = ff.div(fnorm.expand_as(ff)).cpu().numpy()
#         coef = np.corrcoef(ff)

#         # features_ = features.detach().cpu().numpy()
#         # coef = np.corrcoef(features_)

#         diago = np.arange(coef.shape[0])
#         coef[diago, diago] = 0
#         # W = F.relu(torch.tensor((coef - np.diag(np.diag(coef))),
#         #                         dtype=torch.float, device=device))
#         W = F.relu(torch.tensor(coef,
#                                 dtype=torch.float, device=device))
#         # print(W,'wwwwwwwwwwww')
#         for i in range(n):
#             if torch.sum(W[i]) == 0:
#                 # print(W,'wwwwwwwwwwww')

#                 W[i, i] = 1
#                 # print(W,'wwwwwwwwwwww')

#         # print(W,'wwwwwwwww')
#         X = F.softmax(X, dim=1)
#         # print(X)
#         # print(torch.argmax(X,dim=1))
#         # ramdom select anchors
#         ids = torch.unique(targets)
#         # num_samples = n / len(ids)
#         # print(X.dtype)
#         # print(targets)
#         # print(id(X))
#         # X_=X.clone().detach()
#         anchors = []
#         for id_ in ids:
#             anchor = list(np.random.choice(torch.where(targets == id_)[
#                 0].cpu(), size=self.num_anchors, replace=False))
#             # print(id,'ididiid')
#             # print(torch.sum(X[anchors]))
#             # print(torch.argmax(X[anchors]))
#             anchors += anchor

#             # print(torch.argmax(X[anchors]))

#         # print(X[:20,:5],'xxxxxxx')
#         # print(id(X))
#         # print(torch.where(X==torch.max(X,dim=1)))

#         for i in range(self.T):
#             X_ = X.clone().detach()
#             X_[anchors] = torch.tensor(F.one_hot(
#                 targets[anchors], self.num_classes), dtype=torch.float, device=device)
#             # print(i)
#             # print(X,'xxxxxxxxxxxx')
#             # print(X_,'---------')
#             Pi = torch.mm(W, X_)
#             # print(Pi)
#             # print(Pi, 'pipipi')

#             PX = torch.mul(X, Pi)

#             # X = F.normalize(PX, dim=1, p=1)

#             # print(PX,'pxpxpx')
#             # print(PX.shape)

#             # 111111111111111111111111
#             # Norm = np.sum(PX.detach().cpu().numpy(),
#             #               axis=1).reshape(-1)  # .expand(n,m)
#             # # print(Norm,'norm')
#             # Q = 1 / Norm
#             # # print(Q,'QQQQQQQQQ')
#             # Q = torch.diag(torch.tensor(Q, dtype=torch.float, device=device))

#             # 2222222222222222222222222
#             # denom = PX.detach().norm(p=1, dim=1, keepdim=True).clamp_min(1e-12).expand_as(PX)
#             # X=PX/denom

#             # 3333333333333333333333
#             # Q = torch.diag(1 / PX.norm(p=1, dim=1).clamp_min(1e-12))
#             Q = torch.diag(1 / PX.detach().norm(p=1, dim=1).clamp_min(1e-12))
#             X = torch.mm(Q, PX)

#             # 444444444444444444444444444444
#             # Q = torch.diag(1 / torch.matmul(
#             #     PX, torch.ones(m, dtype=torch.float, device=device)))
#             # print(Q,'qqqqq')
#             # X = torch.matmul(Q, PX)
#             # Q=torch.pow(Q,-1)
#             # print(X)

#             # 555555555555555555555555555555555555
#         # X = F.softmax(PX, dim=1)

#         # print(X.requires_grad)
#         loss = self.nllloss(torch.log(X.clamp_min(1e-12)), targets)

#         # loss= self.cross_entropy(X,targets)
#         return loss

#         # #inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
#         # # Compute pairwise distance, replace by the official when merged
#         # dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
#         # dist = dist + dist.t()
#         # dist.addmm_(1, -2, inputs, inputs.t())
#         # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
#         # # For each anchor, find the hardest positive and negative
#         # mask = targets.expand(n, n).eq(targets.expand(n, n).t())
#         # print(mask[:8, :8])
#         # dist_ap, dist_an = [], []
#         # for i in range(n):
#         #     dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
#         #     dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
#         # dist_ap = torch.cat(dist_ap)
#         # dist_an = torch.cat(dist_an)
#         # # Compute ranking hinge loss
#         # y = torch.ones_like(dist_an)
#         # loss = self.ranking_loss(dist_an, dist_ap, y)
#         # if self.mutual:
#         #     return loss, dist
#         # return loss
