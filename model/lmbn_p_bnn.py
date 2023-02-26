import copy
import torch
from torch import nn
from .osnet import osnet_x1_0, OSBlock,osnet_x1_25, osnet_x0_75, osnet_x0_5, osnet_x0_25, osnet_ibn_x1_0
from .attention import BatchDrop, BatchFeatureErase_Top, PAM_Module, CAM_Module, SE_Module, Dual_Module
from .bnneck import BNNeck, BNNeck3
from torch.nn import functional as F

from torch.autograd import Variable


class LMBN_p_bnn(nn.Module):
    def __init__(self, args):
        super(LMBN_p_bnn, self).__init__()

        self.osnet_size(args)
        channels = self.channels

        self.n_ch = 2
        self.chs = channels // self.n_ch

        reduction = BNNeck(
            args.feats, args.num_classes, return_f=True)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)

        self.shared = nn.Sequential(nn.Conv2d(
            self.chs, args.feats, 1, bias=False), nn.BatchNorm2d(args.feats), nn.ReLU(True))
        self.weights_init_kaiming(self.shared)
        self.reduction_ch_0 = BNNeck(
            args.feats, args.num_classes, return_f=True)
        self.reduction_ch_1 = BNNeck(
            args.feats, args.num_classes, return_f=True)

        self.batch_drop_block = BatchFeatureErase_Top(channels_in=channels,channels_out=channels, bottleneck_type=OSBlock)

        self.activation_map = args.activation_map

    def forward(self, x):
        [glo_drop, glo, g_par, p0, p1, c0, c1] = x

        f_glo = self.reduction_0(glo)
        f_p0 = self.reduction_1(g_par)
        f_p1 = self.reduction_2(p0)
        f_p2 = self.reduction_3(p1)
        f_glo_drop = self.reduction_4(glo_drop)
        f_c0 = self.reduction_ch_0(c0)
        f_c1 = self.reduction_ch_1(c1)

        ################

        fea = [f_glo[-1], f_glo_drop[-1], f_p0[-1]]

        if not self.training:

            return torch.stack([f_glo[0], f_glo_drop[0], f_p0[0], f_p1[0], f_p2[0], f_c0[0], f_c1[0]], dim=2)
            # return torch.stack([f_glo_drop[0], f_p0[0], f_p1[0], f_p2[0], f_c0[0], f_c1[0]], dim=2)

        return [f_glo[1], f_glo_drop[1], f_p0[1], f_p1[1], f_p2[1], f_c0[1], f_c1[1]], fea

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

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


if __name__ == '__main__':
    # Here I left a simple forward function.
    # Test the model, before you train it.
    import argparse

    parser = argparse.ArgumentParser(description='MGN')
    parser.add_argument('--num_classes', type=int, default=751, help='')
    parser.add_argument('--bnneck', type=bool, default=True)
    parser.add_argument('--pool', type=str, default='max')
    parser.add_argument('--feats', type=int, default=512)
    parser.add_argument('--drop_block', type=bool, default=True)
    parser.add_argument('--w_ratio', type=float, default=1.0, help='')

    args = parser.parse_args()
    net = MCMP_n(args)
    # net.classifier = nn.Sequential()
    # print([p for p in net.parameters()])
    # a=filter(lambda p: p.requires_grad, net.parameters())
    # print(a)

    print(net)
    input = Variable(torch.FloatTensor(8, 3, 384, 128))
    net.eval()
    output = net(input)
    print(output.shape)
    print('net output size:')
    # print(len(output))
    # for k in output[0]:
    #     print(k.shape)
    # for k in output[1]:
    #     print(k.shape)
class FC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FC, self).__init__()
        self.reduction = nn.Conv2d(in_dim, in_dim, 1, bias=False)
        self.bn = nn.BatchNorm1d(in_dim)
        self.FC = nn.Linear(in_dim, out_dim, bias=False)
        self.ac = nn.ReLU()

    def forward(self, x):
        x = self.reduction(x)
        x = self.bn(x.flatten(1))
        x = self.FC(x)
        return self.ac(x)

