import copy
import torch
from torch import nn
from .osnet import osnet_x1_0, OSBlock, osnet_x1_25, osnet_x0_75, osnet_x0_5, osnet_x0_25, osnet_ibn_x1_0
from .attention import BatchDrop, BatchFeatureErase_Top, PAM_Module, CAM_Module, SE_Module, Dual_Module
from .bnneck import BNNeck, BNNeck3
from torch.nn import functional as F

from torch.autograd import Variable


class LMBN_p_cnn(nn.Module):
    def __init__(self, args):
        super(LMBN_p_cnn, self).__init__()

        self.osnet_size(args)
        osnet = self.osnet_model
        channels = self.channels

        self.n_ch = 2
        self.chs = channels // self.n_ch

        # osnet = osnet_x1_0(pretrained=True)

        self.backbone = nn.Sequential(
            osnet.conv1,
            osnet.maxpool,
            osnet.conv2,
            osnet.conv3[0])

        conv3 = osnet.conv3[1:]

        self.global_branch = nn.Sequential(copy.deepcopy(
            conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

        self.partial_branch = nn.Sequential(copy.deepcopy(
            conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

        self.channel_branch = nn.Sequential(copy.deepcopy(
            conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

        self.global_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.partial_pooling = nn.AdaptiveAvgPool2d((2, 1))
        self.channel_pooling = nn.AdaptiveAvgPool2d((1, 1))


        # if args.drop_block:
        #     print('Using batch random erasing block.')
        #     self.batch_drop_block = BatchRandomErasing()
        # print('Using batch drop block.')
        # self.batch_drop_block = BatchDrop(
        #     h_ratio=args.h_ratio, w_ratio=args.w_ratio)
        self.batch_drop_block = BatchFeatureErase_Top(channels_in=channels, channels_out=channels,
                                                      bottleneck_type=OSBlock)

        self.activation_map = args.activation_map

        self.con1x1_0 = Conv1x1(channels, channels)
        self.con1x1_1 = Conv1x1(channels, channels)
        self.con1x1_2 = Conv1x1(channels, channels)
        self.con1x1_3 = Conv1x1(channels, channels)
        self.con1x1_4 = Conv1x1(channels, channels)
        self.con1x1_5 = Conv1x1(self.chs, channels)
        self.con1x1_6 = Conv1x1(self.chs, channels)


    def forward(self, x):
        # if self.batch_drop_block is not None:
        #     x = self.batch_drop_block(x)

        x = self.backbone(x)  # x: torch.Size([1, 384, 27, 27])

        glo = self.global_branch(x)
        par = self.partial_branch(x)
        cha = self.channel_branch(x)

        if self.activation_map:
            glo_ = glo

        if self.batch_drop_block is not None:
            glo_drop, glo = self.batch_drop_block(glo)

        if self.activation_map:
            _, _, h_par, _ = par.size()

            fmap_p0 = par[:, :, :h_par // 2, :]
            fmap_p1 = par[:, :, h_par // 2:, :]
            fmap_c0 = cha[:, :self.chs, :, :]
            fmap_c1 = cha[:, self.chs:, :, :]
            print('Generating activation maps...')

            return glo, glo_, fmap_c0, fmap_c1, fmap_p0, fmap_p1

        glo_drop = self.global_pooling(glo_drop)
        glo = self.channel_pooling(glo)  # shape:(batchsize, 512,1,1)
        g_par = self.global_pooling(par)  # shape:(batchsize, 512,1,1)
        p_par = self.partial_pooling(par)  # shape:(batchsize, 512,2,1)
        cha = self.channel_pooling(cha)  # shape:(batchsize, 256,1,1)

        p0 = p_par[:, :, 0:1, :]
        p1 = p_par[:, :, 1:2, :]

        c0 = cha[:, :self.chs, :, :]
        c1 = cha[:, self.chs:, :, :]

        glo_drop = self.con1x1_0(glo_drop)
        glo = self.con1x1_1(glo)
        g_par = self.con1x1_2(g_par)
        p0 = self.con1x1_3(p0)
        p1 = self.con1x1_4(p1)
        c0 = self.con1x1_5(c0)
        c1 = self.con1x1_6(c1)




        return [glo_drop, glo, g_par, p0, p1, c0, c1]

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
            self.osnet_model = osnet_x1_0(pretrained=True)
            self.channels = 512
        if args.osnet_size.lower() == 'osnet_x1_25':
            self.osnet_model = osnet_x1_25(pretrained=True)
            self.channels = 640
        if args.osnet_size.lower() == 'osnet_x0_75':
            self.osnet_model = osnet_x0_75(pretrained=True)
            self.channels = 384
        if args.osnet_size.lower() == 'osnet_x0_5':
            self.osnet_model = osnet_x0_5(pretrained=True)
            self.channels = 256
        if args.osnet_size.lower() == 'osnet_x0_25':
            self.osnet_model = osnet_x0_25(pretrained=True)
            self.channels = 128
        if args.osnet_size.lower() == 'osnet_x1_0':
            self.osnet_model = osnet_ibn_x1_0(pretrained=True)
            self.channels = 512
        else:
            self.osnet_model = osnet_x1_0(pretrained=True)
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


class Conv1x1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv1x1, self).__init__()
        self.reduction = nn.Conv2d(in_dim, out_dim, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = self.reduction(x)
        return self.bn(x.flatten(1))
