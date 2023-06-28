import copy
import torch
from torch import nn
from .osnet import osnet_x1_0, OSBlock,osnet_x1_25, osnet_x0_75, osnet_x0_5, osnet_x0_25, osnet_ibn_x1_0
from .attention import BatchDrop, BatchFeatureErase_Top, PAM_Module, CAM_Module, SE_Module, Dual_Module
from .bnneck import BNNeck, BNNeck3
from torch.nn import functional as F

from torch.autograd import Variable


class LMBN_p_fc(nn.Module):
    def __init__(self, args):
        super(LMBN_p_fc, self).__init__()

        self.osnet_size(args)
        channels = self.channels

        self.n_ch = 2
        self.chs = channels // self.n_ch

        #osnet = osnet_x1_0(pretrained=True)
        FullyConLayer = FC(channels, args.feats)
        self.FC1 = copy.deepcopy(FullyConLayer)
        self.FC2 = copy.deepcopy(FullyConLayer)
        self.FC3 = copy.deepcopy(FullyConLayer)
        self.FC4 = copy.deepcopy(FullyConLayer)
        self.FC5 = copy.deepcopy(FullyConLayer)
        self.FC6 = copy.deepcopy(FullyConLayer)
        self.FC7 = copy.deepcopy(FullyConLayer)


    def forward(self, x):
        # if self.batch_drop_block is not None:
        #     x = self.batch_drop_block(x)

        [glo_drop, glo, g_par, p0, p1, c0, c1] = x

        glo = self.FC1(glo)
        g_par = self.FC2(g_par)
        p0 = self.FC3(p0)
        p1 = self.FC4(p1)
        glo_drop = self.FC5(glo_drop)
        c0 = self.FC6(c0)
        c1 = self.FC7(c1)

        return [glo_drop, glo, g_par, p0, p1, c0, c1]
    def osnet_size(self, args):
        if args.osnet_size.lower() == 'osnet_x1_0':

            self.channels = 512
        if args.osnet_size.lower() == 'osnet_x1_25':

            self.channels = 640
        if args.osnet_size.lower() == 'osnet_x0_75':

            self.channels = 384
        if args.osnet_size.lower() == 'osnet_x0_5':

            self.channels = 256
        if args.osnet_size.lower() == 'osnet_x0_25':

            self.channels = 128
        if args.osnet_size.lower() == 'osnet_x1_0':

            self.channels = 512
        else:

            self.channels = 512



class FC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FC, self).__init__()
        self.FC = nn.Linear(in_dim, out_dim, bias=False)
        self.ac = nn.ReLU()

    def forward(self, x):
        x = self.FC(x)
        return self.ac(x)
