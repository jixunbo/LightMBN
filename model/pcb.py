import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck
from torch.autograd import Variable
from .bnneck import BNNeck, BNNeck3, ClassBlock


class PCB(nn.Module):
    def __init__(self, args):
        super(PCB, self).__init__()

        self.part = args.parts  # We cut the pool5 to 6 parts
        model_ft = resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))

        self.avgpool_before_triplet = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        self.bnneck = args.bnneck
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier' + str(i)
            if self.bnneck:
                # setattr(self, name, BNNeck(2048, args.num_classes, return_f=True))
                setattr(self, name, BNNeck3(
                    2048, args.num_classes, args.feats, return_f=True))
            else:

                setattr(self, name, ClassBlock(2048, args.num_classes, droprate=0.5,
                                               relu=False, bnorm=True, num_bottleneck=args.feats, return_f=True))
        self.global_branch = BNNeck3(
            2048, args.num_classes, args.feats, return_f=True)
        print('PCB_conv divide into {} parts, using {} dims feature.'.format(
            args.parts, args.feats))
        # self.global_branch = BNNeck(2048, args.num_classes, return_f=True)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        feat_to_global_branch = self.avgpool_before_triplet(x)
        x = self.avgpool(x)
        # x = self.dropout(x)
        # print(x.shape)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = x[:, :, i].unsqueeze(dim=3)
            # part[i] = torch.squeeze(x[:, :,:, i])

            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])

        global_feat = [x.view(x.size(0), x.size(1), x.size(2))]

        feat_global_branch = self.global_branch(feat_to_global_branch)
        # y = [x.view(x.size(0), -1)]

        score = []
        after_neck = []
        # print(y[0].shape)
        for i in range(self.part):

            score.append(predict[i][1])
            if self.bnneck:

                after_neck.append(predict[i][0])

        if not self.training:
            return torch.stack(after_neck + [feat_global_branch[0]], dim=2)
        return score + [feat_global_branch[1]], feat_global_branch[-1]


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
    net = PCB(args)
    # net.classifier = nn.Sequential()
    # print([p for p in net.parameters()])
    # a=filter(lambda p: p.requires_grad, net.parameters())
    # print(a)
    net.eval()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 128))
    output = net(input)
    print('net output size:')
    print(len(output))
    print(output.shape)
    # for k in output[0]:
    #     print(k.shape)
    # print(output[-1].shape)
