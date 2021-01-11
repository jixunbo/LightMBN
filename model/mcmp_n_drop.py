import copy

import torch
from torch import nn
from .osnet import osnet_x1_0, OSBlock
from .attention import BatchDrop, BatchRandomErasing, PAM_Module, CAM_Module, SE_Module, Dual_Module
from .bnneck import BNNeck, BNNeck3
from torch.nn import functional as F


from torch.autograd import Variable


class BatchDropTop(nn.Module):
    def __init__(self, h_ratio):
        super(BatchDropTop, self).__init__()
        self.h_ratio = h_ratio

    def forward(self, x, visdrop=False):
        if self.training or visdrop:
            b, c, h, w = x.size()
            rh = round(self.h_ratio * h)
            act = (x**2).sum(1)
            act = act.view(b, h * w)
            act = F.normalize(act, p=2, dim=1)
            act = act.view(b, h, w)
            max_act, _ = act.max(2)
            ind = torch.argsort(max_act, 1)
            ind = ind[:, -rh:]
            mask = []
            for i in range(b):
                rmask = torch.ones(h)
                rmask[ind[i]] = 0
                mask.append(rmask.unsqueeze(0))
            mask = torch.cat(mask)
            mask = torch.repeat_interleave(mask, w, 1).view(b, h, w)
            mask = torch.repeat_interleave(mask, c, 0).view(b, c, h, w)
            if x.is_cuda:
                mask = mask.cuda()
            if visdrop:
                return mask
            x = x * mask
        return x


class BatchFeatureErase_Top(nn.Module):
    def __init__(self, channels, h_ratio=0.33, w_ratio=1., double_bottleneck=False):
        super(BatchFeatureErase_Top, self).__init__()
        # if double_bottleneck:
        #     self.drop_batch_bottleneck = nn.Sequential(
        #         Bottleneck(channels, 512),
        #         Bottleneck(channels, 512)
        #     )
        # else:
        #     self.drop_batch_bottleneck = Bottleneck(channels, 512)
        if double_bottleneck:
            self.drop_batch_bottleneck = nn.Sequential(
                OSBlock(channels, 512),
                OSBlock(channels, 512)
            )
        else:
            self.drop_batch_bottleneck = OSBlock(channels, 512)

        # self.drop_batch_drop_basic = BatchDrop(h_ratio, w_ratio)
        self.drop_batch_drop_top = BatchDropTop(h_ratio)

    def forward(self, x, drop_top=True, bottleneck_features=True, visdrop=False):
        features = self.drop_batch_bottleneck(x)
        if drop_top:
            x = self.drop_batch_drop_top(features, visdrop=visdrop)

        # if drop_top:
        #     x = self.drop_batch_drop_top(x, visdrop=visdrop)
        # else:
        #     x = self.drop_batch_drop_basic(features, visdrop=visdrop)
        if visdrop:
            return x  # x is dropmask
        if bottleneck_features:
            return x, features
        else:
            return x


class MCMP_n_drop(nn.Module):
    def __init__(self, args):
        super(MCMP_n_drop, self).__init__()

        self.n_ch = 2
        self.chs = 512 // self.n_ch

        osnet = osnet_x1_0(pretrained=True)

        self.backone = nn.Sequential(
            osnet.conv1,
            osnet.maxpool,
            osnet.conv2,
            # attention,
            osnet.conv3[0]
        )

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

        reduction = BNNeck3(512, args.num_classes,
                            args.feats, return_f=True)
        # reduction = BNNeck(
        #     args.feats, args.num_classes, return_f=True)
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

        # if args.drop_block:
        #     print('Using batch random erasing block.')
        #     self.batch_drop_block = BatchRandomErasing()
        # print('Using batch drop block.')
        # self.batch_drop_block = BatchDrop(
        #     h_ratio=args.h_ratio, w_ratio=args.w_ratio)
        self.batch_drop_block = BatchFeatureErase_Top(512)

        self.activation_map = args.activation_map

    def forward(self, x):
        # if self.batch_drop_block is not None:
        #     x = self.batch_drop_block(x)

        x = self.backone(x)

        glo = self.global_branch(x)
        par = self.partial_branch(x)
        cha = self.channel_branch(x)

        if self.activation_map:

            _, _, h_par, _ = par.size()

            fmap_p0 = par[:, :, :h_par // 2, :]
            fmap_p1 = par[:, :, h_par // 2:, :]
            fmap_c0 = cha[:, :self.chs, :, :]
            fmap_c1 = cha[:, self.chs:, :, :]
            print('activation_map')

            return glo, fmap_c0, fmap_c1, fmap_p0, fmap_p1

        if self.batch_drop_block is not None:
            glo_drop, glo = self.batch_drop_block(glo)

        glo_drop = self.global_pooling(glo_drop)
        glo = self.channel_pooling(glo)  # shape:(batchsize, 512,1,1)
        g_par = self.global_pooling(par)  # shape:(batchsize, 512,1,1)
        p_par = self.partial_pooling(par)  # shape:(batchsize, 512,3,1)
        cha = self.channel_pooling(cha)

        p0 = p_par[:, :, 0:1, :]
        p1 = p_par[:, :, 1:2, :]

        f_glo = self.reduction_0(glo)
        f_p0 = self.reduction_1(g_par)
        f_p1 = self.reduction_2(p0)
        f_p2 = self.reduction_3(p1)
        f_glo_drop = self.reduction_4(glo_drop)

        ################

        c0 = cha[:, :self.chs, :, :]
        c1 = cha[:, self.chs:, :, :]
        c0 = self.shared(c0)
        c1 = self.shared(c1)
        f_c0 = self.reduction_ch_0(c0)
        f_c1 = self.reduction_ch_1(c1)

        ################

        fea = [f_glo[-1], f_p0[-1], f_glo_drop[-1]]

        if not self.training:

            return torch.stack([f_glo[0], f_glo_drop[0], f_p0[0], f_p1[0], f_p2[0], f_c0[0], f_c1[0]], dim=2)

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
