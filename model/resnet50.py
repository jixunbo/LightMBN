import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck
import random
from .bnneck import BNNeck, BNNeck3, ClassBlock
from torch.autograd import Variable


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


# Define the ResNet50-based Model
class ResNet50(nn.Module):

    # def __init__(self, class_num, droprate=0.5, stride=2):
    def __init__(self, args, droprate=0.5, stride=1):

        super(ResNet50, self).__init__()
        resnet = resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            resnet.layer4[0].downsample[0].stride = (1, 1)
            resnet.layer4[0].conv2.stride = (1, 1)
        resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = resnet
        self.bnneck = args.bnneck
        self.drop_block = args.drop_block

        if args.feat_inference == 'before':
            self.before_neck = True
            print('Using before_neck inference')
        else:
            self.before_neck = False
        if args.drop_block:
            print('Using batch drop block.')
            resnet.avgpool = nn.AdaptiveMaxPool2d((1, 1))
            self.batch_drop_block = BatchDrop(
                h_ratio=args.h_ratio, w_ratio=args.w_ratio)
        if self.bnneck:
            self.classifier = BNNeck(2048, args.num_classes,  # feat_dim=512,
                                     return_f=True)
            # self.classifier = BNNeck3(2048, args.num_classes, feat_dim=512,
            #  return_f=True)

        else:

            self.classifier = ClassBlock(
                2048, args.num_classes, num_bottleneck=args.feats, return_f=True)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # print(x.size())
        if self.drop_block:
            x = self.batch_drop_block(x)
        x = self.model.avgpool(x)
        # x = x.view(x.size(0), x.size(1))
        # print(x.shape)
        x = self.classifier(x)
        # print(x[0].shape)
        # print(x[1].shape)
        # print(x[2].shape)

        if not self.training:
            if self.before_neck:
                return x[-1]
            return x[0]
        # print(x[1].size())
        # print(x[-1].size())
        return [x[1]], [x[-1]]


if __name__ == '__main__':
    # Here I left a simple forward function.
    # Test the model, before you train it.
    import argparse

    parser = argparse.ArgumentParser(description='MGN')
    parser.add_argument('--num_classes', type=int, default=751, help='')
    parser.add_argument('--bnneck', type=bool, default=False)
    parser.add_argument('--pool', type=str, default='max')
    parser.add_argument('--feats', type=int, default=256)
    parser.add_argument('--drop_block', type=bool, default=True)
    parser.add_argument('--w_ratio', type=float, default=1.0, help='')
    parser.add_argument('--h_ratio', type=float, default=0.33, help='')

    args = parser.parse_args()
    net = ResNet50(args)
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
