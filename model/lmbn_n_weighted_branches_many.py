import copy
import torch
from torch import nn
from .osnet import osnet_x1_0, OSBlock,osnet_x1_25, osnet_x0_75, osnet_x0_5, osnet_x0_25, osnet_ibn_x1_0
from .attention import BatchDrop, BatchFeatureErase_Top, PAM_Module, CAM_Module, SE_Module, Dual_Module
from .bnneck import BNNeck, BNNeck3
from torch.nn import functional as F

from torch.autograd import Variable


class LMBN_n_weighted_branches_many(nn.Module):
    def __init__(self, args):
        super(LMBN_n_weighted_branches_many, self).__init__()

        self.osnet_size(args)
        osnet = self.osnet_model
        channels = self.channels

        self.n_ch = 2
        self.chs = channels // self.n_ch

        #osnet = osnet_x1_0(pretrained=True)

        self.backbone = nn.Sequential(
            osnet.conv1,
            osnet.maxpool,
            osnet.conv2,
            osnet.conv3[0]
        )

        conv3 = osnet.conv3[1:]

        self.global_branch = nn.Sequential(copy.deepcopy(
            conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

        self.partial_branch = nn.Sequential(copy.deepcopy(
            conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

        self.channel_branch = nn.Sequential(copy.deepcopy(
            conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

        self.quater_branch = nn.Sequential(copy.deepcopy(
            conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

        self.vertical_branch = nn.Sequential(copy.deepcopy(
            conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

        self.global_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.partial_pooling = nn.AdaptiveAvgPool2d((2, 1))
        self.vertical_pooling = nn.AdaptiveAvgPool2d((1, 2))
        self.quater_pooling =  nn.AdaptiveAvgPool2d((2, 2))
        self.channel_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.num_groups = 15
        self.bn = nn.BatchNorm2d((self.num_groups)*channels)
        self.bn_s = nn.BatchNorm2d(channels)
        self.GDN = GDN(channels*(self.num_groups),self.num_groups)


        reduction = BNNeck3(channels, args.num_classes,
                            args.feats, return_f=True)

        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)
        self.reduction_8 = copy.deepcopy(reduction)
        self.reduction_9 = copy.deepcopy(reduction)
        self.reduction_10 = copy.deepcopy(reduction)
        self.reduction_11 = copy.deepcopy(reduction)
        self.reduction_12 = copy.deepcopy(reduction)

        self.shared = nn.Sequential(nn.Conv2d(
            self.chs, args.feats, 1, bias=False), nn.BatchNorm2d(args.feats), nn.ReLU(True))
        self.weights_init_kaiming(self.shared)

        self.reduction_ch_0 = BNNeck(
            args.feats, args.num_classes, return_f=True)
        self.reduction_ch_1 = BNNeck(
            args.feats, args.num_classes, return_f=True)

        # if args.drop_block:
        #     print('Using batch random erasing block.')
        #     self.batch_drop_block = BatchRandomErasing()
        # print('Using batch drop block.')
        # self.batch_drop_block = BatchDrop(
        #     h_ratio=args.h_ratio, w_ratio=args.w_ratio)
        self.batch_drop_block = BatchFeatureErase_Top(channels_in=channels,channels_out=channels, bottleneck_type=OSBlock)

        self.activation_map = args.activation_map

    def forward(self, x):
        # if self.batch_drop_block is not None:
        #     x = self.batch_drop_block(x)

        x = self.backbone(x) # x: torch.Size([1, 384, 27, 27])

        glo = self.global_branch(x)
        par = self.partial_branch(x)
        cha = self.channel_branch(x)
        ver = self.vertical_branch(x)
        qua = self.quater_branch(x)

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
        p_ver = self.vertical_pooling(ver)
        g_var = self.global_pooling(ver)
        p_qua = self.quater_pooling(qua)
        g_qua = self.global_pooling(qua)


        p0 = p_par[:, :, 0:1, :]
        p1 = p_par[:, :, 1:2, :]
        q0 = p_qua[:, :, 0:1, 0:1]
        q1 = p_qua[:, :, 0:1, 1:2]
        q2 = p_qua[:, :, 1:2, 0:1]
        q3 = p_qua[:, :, 1:2, 1:2]
        v0 = p_ver[:, :, :, 0:1]
        v1 = p_ver[:, :, :, 1:2]

        c0 = cha[:, :self.chs, :, :]
        c1 = cha[:, self.chs:, :, :]
        c0 = self.shared(c0)
        c1 = self.shared(c1)


        groups = (glo_drop, glo, g_par,p0, p1, c0, c1, g_qua, q0, q1, q2, q3 ,g_var, v0, v1)
        bz,_,_,_ = glo.size()
        all_par = torch.cat(groups,1)
        all_par = self.bn(all_par)
        weights = self.GDN(all_par.flatten(1))
        weights = torch.reshape(weights, (bz, self.num_groups, 1, 1))
        i = 0
        l = list()
        for group in groups:
            w = torch.reshape(weights[:,i,:,:], (bz,1,1,1))
            group = torch.mul(w,self.bn_s(group))
            l.append(group)

        glo_drop, glo, g_par,p0, p1, c0, c1, g_qua, q0, q1, q2, q3 ,g_var, v0, v1 = l

        f_glo = self.reduction_0(glo)
        f_p0 = self.reduction_1(g_par)
        f_p1 = self.reduction_2(p0)
        f_p2 = self.reduction_3(p1)
        f_glo_drop = self.reduction_4(glo_drop)
        f_c0 = self.reduction_ch_0(c0)
        f_c1 = self.reduction_ch_1(c1)
        f_g_qua = self.reduction_6(g_qua)
        f_q0 = self.reduction_7(q0)
        f_q1 =  self.reduction_8(q1)
        f_q2 = self.reduction_9(q2)
        f_q3 = self.reduction_10(q3)
        f_g_var = self.reduction_11(g_var)
        f_v0 = self.reduction_12(v0)
        f_v1 = self.reduction_5(v1)

        ################

        fea = [f_glo[-1], f_glo_drop[-1], f_p0[-1]]

        if not self.training:

            return torch.stack([f_glo[0], f_glo_drop[0], f_p0[0], f_p1[0], f_p2[0], f_c0[0], f_c1[0],f_g_qua[0],
                                f_q0[0],f_q1[0],f_q2[0],f_q3[0],f_g_var[0],f_v0[0],f_v1[0]], dim=2)
            # return torch.stack([f_glo_drop[0], f_p0[0], f_p1[0], f_p2[0], f_c0[0], f_c1[0]], dim=2)

        return [f_glo[1], f_glo_drop[1], f_p0[1], f_p1[1], f_p2[1], f_c0[1], f_c1[1],f_g_qua[1],
                                f_q0[1],f_q1[1],f_q2[1],f_q3[1],f_g_var[1],f_v0[1],f_v1[1]], fea

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


class FC(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(FC, self).__init__()
        self.fc = nn.Linear(inplanes, outplanes, bias=False)
        self.bn = nn.BatchNorm1d(outplanes)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.fc(x)
        return self.act(x)


class GDN(nn.Module):
    def __init__(self, inplanes, outplanes, intermediate_dim=256):
        super(GDN, self).__init__()
        self.fc1 = FC(inplanes, intermediate_dim)
        self.fc2 = FC(intermediate_dim, outplanes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        intermediate = self.fc1(x)
        out = self.fc2(intermediate)
        # return intermediate, self.softmax(out)
        return torch.softmax(out, dim=1)