import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck
from torch.autograd import Variable


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(self.weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(self.weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return f, x
        else:
            x = self.classifier(x)
            return x

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv') != -1:
            # For old pytorch, you may use kaiming_normal.
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, std=0.001)
            nn.init.constant_(m.bias.data, 0.0)


class BNNeck(nn.Module):
    def __init__(self, input_dim, class_num, return_f=False):
        super(BNNeck, self).__init__()
        self.return_f = return_f
        self.bn = nn.BatchNorm2d(input_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(input_dim, class_num, bias=False)
        self.bn.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):
        before_neck = x
        # print(before_neck.shape)
        after_neck = self.bn(before_neck).squeeze(3).squeeze(2)
        if self.return_f:
            score = self.classifier(after_neck)
            return after_neck, score, before_neck
        else:
            x = self.classifier(x)
            return x

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

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


class new_BNNeck(nn.Module):
    def __init__(self, input_dim, class_num, feat_dim, return_f=False):
        super(new_BNNeck, self).__init__()
        self.return_f = return_f
        # self.reduction = nn.Linear(input_dim, feat_dim)
        # self.bn = nn.BatchNorm1d(feat_dim)

        self.reduction = nn.Conv2d(
            input_dim, feat_dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(feat_dim)

        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, class_num, bias=False)
        self.bn.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):
        x = self.reduction(x)
        before_neck = x.squeeze(dim=3).squeeze(dim=2)
        after_neck = self.bn(x).squeeze(dim=3).squeeze(dim=2)
        if self.return_f:
            score = self.classifier(after_neck)
            return after_neck, score, before_neck
        else:
            x = self.classifier(x)
            return x

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

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)

# Multi Channel Network


class MCN(nn.Module):
    def __init__(self, args):
        super(MCN, self).__init__()
        self.n_c = args.parts
        self.chs = 2048 // self.n_c

        resnet_ = resnet50(pretrained=True)

        self.layer0 = nn.Sequential(
            resnet_.conv1,
            resnet_.bn1,
            resnet_.relu,
            resnet_.maxpool)
        self.layer1 = resnet_.layer1
        self.layer2 = resnet_.layer2
        self.layer3 = resnet_.layer3
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        self.layer4.load_state_dict(resnet_.layer4.state_dict())

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.global_branch = new_BNNeck(2048, args.num_classes, 256,return_f=True)

        self.shared = nn.Sequential(nn.Conv2d(
            self.chs, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True))
        self.weights_init_kaiming(self.shared)

        for i in range(self.n_c):
            name = 'bnneck_' + str(i)
            setattr(self, name, BNNeck(256, args.num_classes, return_f=True))

        # self.global_branch = new_BNNeck(
        #     2048, args.num_classes, args.feats, return_f=True)
        # print('PCB_conv divide into {} parts, using {} dims feature.'.format(
        #     args.parts, args.feats))
        # self.global_branch = BNNeck(2048, args.num_classes, return_f=True)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        global_feat = self.global_branch(x)
        # featself.global_branch


        # feat_to_global_branch = self.avgpool_before_triplet(x)
        # x = self.dropout(x)
        # print(x.shape)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.n_c):
            part[i] = x[:, i * self.chs:(i + 1) * self.chs]
            part[i] = self.shared(part[i])
            # print(part[i].shape,'kkkkk')
            name = 'bnneck_' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
            # print(predict[i][0].shape,'jjjjj')

        # glfoobal_feat = [x.view(x.size(0), x.size(1), x.size(2))]

        # feat_global_branch = self.global_branch(feat_to_global_branch)
        # y = [x.view(x.size(0), -1)]

        score = [global_feat[1]]
        after_neck = [global_feat[0]]
        # print(y[0].shape)
        for i in range(self.n_c):

            score.append(predict[i][1])
            after_neck.append(predict[i][0])
        # print(y[0].shape)
        # print(y[0][1].shape)
        # return torch.cat([y[0][0],y[1][0],y[2][0],y[3][0],y[4][0],y[5][0]],dim=1),y[0][1],y[1][1],y[2][1],y[3][1],y[4][1],y[5][1]
            # return [torch.stack(after_neck,dim=2)]+score+[global_feat_to_triplet]
        # print(len(after_neck))
        # print(len(score))
        # print(after_neck[0].shape)
        # print(torch.stack(after_neck, dim=2))
        # print(score)
        return [torch.stack(after_neck, dim=2), score, [global_feat[-1]]]

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

    # def weights_init_classifier(m):
    #     classname = m.__class__.__name__
    #     if classname.find('Linear') != -1:
    #         nn.init.normal_(m.weight, std=0.001)
    #         if m.bias:
    #             nn.init.constant_(m.bias, 0.0)


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
    net = MCN(args)
    # net.classifier = nn.Sequential()
    # print([p for p in net.parameters()])
    # a=filter(lambda p: p.requires_grad, net.parameters())
    # print(a)

    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 128))
    output = net(input)
    print('net output size:')
    print(len(output))
    print(output[0].shape)
    for k in output[1]:
        print(k.shape)
