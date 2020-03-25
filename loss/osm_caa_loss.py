import torch
import torch.nn as nn
from torch.autograd import Variable


class OSM_CAA_Loss(nn.Module):
    def __init__(self, alpha=1.2, l=0.5, use_gpu=True, osm_sigma=0.8):
        super(OSM_CAA_Loss, self).__init__()
        self.use_gpu = use_gpu
        self.alpha = alpha  # margin of weighted contrastive loss, as mentioned in the paper
        self.l = l  # hyperparameter controlling weights of positive set and the negative set
        # I haven't been able to figure out the use of \sigma CAA 0.18
        self.osm_sigma = osm_sigma  # \sigma OSM (0.8) as mentioned in paper

    def forward(self, x, embd, labels):
        '''
        x : feature vector : (n x d)
        labels : (n,)
        embd : Fully Connected weights of classification layer (dxC), C is the number of classes: represents the vectors for class
        '''
        x = nn.functional.normalize(x, p=2, dim=1)  # normalize the features
        n = x.size(0)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, x, x.t())
        dist = dist.clamp(min=1e-12).sqrt()
        # print(dist,'dist')
        S = torch.exp(-1.0 * torch.pow(dist, 2) /
                      (self.osm_sigma * self.osm_sigma))
        # max (0, self.alpha - dij )
        # print(S,'ssssssssss')
        S_ = torch.clamp(self.alpha - dist, min=1e-12)
        p_mask = labels.expand(n, n).eq(labels.expand(n, n).t())
        p_mask = p_mask.float()
        n_mask = 1 - p_mask
        S = S * p_mask.float()
        S = S + S_ * n_mask.float()
        # embd = nn.functional.normalize(embd, p=2, dim=0)
        # denominator = torch.exp(torch.mm(x, embd))
        # A = []
        # for i in range(n):
        #     a_i = denominator[i][labels[i]] / torch.sum(denominator[i])
        #     A.append(a_i)
        # atten_class = torch.stack(A)
        A = []
        # print(labels,'label')
        for i in range(n):
            A.append(embd[i][labels[i]])
        atten_class = torch.stack(A)

        # pairwise minimum of attention weights
        A = torch.min(atten_class.expand(n, n),
                      atten_class.view(-1, 1).expand(n, n))
        W = S * A
        W_P = W * p_mask.float()
        W_N = W * n_mask.float()
        if self.use_gpu:
            # dist between (xi,xi) not necessarily 0, avoiding precision error
            W_P = W_P * (1 - torch.eye(n, n).float().cuda())
            W_N = W_N * (1 - torch.eye(n, n).float().cuda())
        else:
            W_P = W_P * (1 - torch.eye(n, n).float())
            W_N = W_N * (1 - torch.eye(n, n).float())
        L_P = 1.0 / 2 * torch.sum(W_P * torch.pow(dist, 2)) / torch.sum(W_P)
        L_N = 1.0 / 2 * torch.sum(W_N * torch.pow(S_, 2)) / torch.sum(W_N)
        # print(L_P,'lplplplplp')
        L = (1 - self.l) * L_P + self.l * L_N
        return L


if __name__ == '__main__':
    # Here I left a simple forward function.
    # Test the model, before you train it.
    import argparse

    parser = argparse.ArgumentParser(description='MGN')
    parser.add_argument('--num_classes', type=int, default=751, help='')
    parser.add_argument('--bnneck', type=bool, default=True)
    parser.add_argument('--parts', type=int, default=3)
    parser.add_argument('--feats', type=int, default=256)

    args = parser.parse_args()
    net = OSM_CAA_Loss(use_gpu=False)
    # net.classifier = nn.Sequential()
    # print([p for p in net.parameters()])
    # a=filter(lambda p: p.requires_grad, net.parameters())
    # print(a)

    print(net)
    d = 256
    c = 751
    x = Variable(torch.FloatTensor(8, d))
    label = Variable(torch.arange(8))
    embd = Variable(torch.FloatTensor(d, 751))

    output = net(x, embd, label)
    print('net output size:')
    # print(len(output))
    print(output.shape)
