import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck
import random


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|

class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx + rh, sy:sy + rw] = 0
            x = x * mask
        return x

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
        self.bn = nn.BatchNorm1d(input_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(input_dim, class_num, bias=False)
        self.bn.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):
        before_neck = x
        after_neck = self.bn(before_neck)
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
        self.reduction = nn.Linear(input_dim, feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, class_num, bias=False)
        self.bn.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):
        x = self.reduction(x)
        before_neck = x
        after_neck = self.bn(before_neck)
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


# Define the ResNet50-based Model
class ResNet501(nn.Module):

    # def __init__(self, class_num, droprate=0.5, stride=2):
    def __init__(self, args, droprate=0.5, stride=2):

        super(ResNet501, self).__init__()
        resnet = resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            resnet.layer4[0].downsample[0].stride = (1, 1)
            resnet.layer4[0].conv2.stride = (1, 1)
        resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = resnet
        self.bnneck = args.bnneck
        self.drop_block = args.drop_block
        if args.drop_block:
            print('Using batch drop block.')
            resnet.avgpool = nn.AdaptiveMaxPool2d((1, 1))
            self.batch_drop_block = BatchDrop(h_ratio=args.h_ratio, w_ratio=args.w_ratio)
        if self.bnneck:
            self.classifier = BNNeck(2048, args.num_classes, return_f=True)
            # self.classifier = new_BNNeck(2048, args.num_classes, 256, return_f=True)

        else:

            self.classifier = ClassBlock(
                2048, args.num_classes, droprate, return_f=True)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if self.drop_block:
            x = self.batch_drop_block(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))

        x = self.classifier(x)
        if not self.training:
            return x[0]

        return x[1],x[-1]

