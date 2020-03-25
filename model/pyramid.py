import torch
from torch import nn, optim
import torch.nn.init as init
import torch.nn.functional as F
from .resnet_pyramid import resnet101
from torch.autograd import Variable



class Pyramid(nn.Module):
    def __init__(
            self,
            args,
            last_conv_stride=1,
            last_conv_dilation=1,
            num_stripes=6,  # number of sub-parts
            used_levels=[1, 1, 1, 1, 1, 1],
            num_conv_out_channels=128,
            global_conv_out_channels=256,
    ):

        super(Pyramid, self).__init__()

        print("num_stripes:{}".format(num_stripes))
        print("num_conv_out_channels:{},".format(num_conv_out_channels))

        self.base = resnet101(
            pretrained=True,
            last_conv_stride=last_conv_stride,
            last_conv_dilation=last_conv_dilation)

        self.dropout_layer = nn.Dropout(p=0.2)

        # ============================================================================== pyramid
        self.num_classes = args.num_classes
        self.num_stripes = num_stripes
        self.used_levels = used_levels

        # ==============================================================================pyramid
        input_size = 2048
        self.pyramid_conv_list0 = nn.ModuleList()
        self.pyramid_fc_list0 = nn.ModuleList()
        Pyramid.register_basic_branch(self, num_conv_out_channels,
                                      input_size,
                                      self.pyramid_conv_list0,
                                      self.pyramid_fc_list0)

        # ==============================================================================pyramid
        input_size1 = 1024
        self.pyramid_conv_list1 = nn.ModuleList()
        self.pyramid_fc_list1 = nn.ModuleList()
        Pyramid.register_basic_branch(self, num_conv_out_channels,
                                      input_size1,
                                      self.pyramid_conv_list1,
                                      self.pyramid_fc_list1)

    def forward(self, x):
        """
        Returns:
        feat_list: each member with shape [N, C]
        logits_list: each member with shape [N, num_classes]
        """
        # shape [N, C, H, W]
        feat = self.base(x)
        # print(feat.shape)

        assert feat.size(2) % self.num_stripes == 0

        # ============================================================================== pyramid
        feat_list = []
        logits_list = []

        Pyramid.pyramid_forward(self, feat,
                                self.pyramid_conv_list0,
                                self.pyramid_fc_list0,
                                feat_list,
                                logits_list)

        return [torch.stack(feat_list,dim=2)]+logits_list
        # ============================================================================== pyramid

    @staticmethod
    def register_basic_branch(self, num_conv_out_channels,
                              input_size,
                              pyramid_conv_list,
                              pyramid_fc_list):
        # the level indexes are defined from fine to coarse,
        # the branch will contain one more part than that of its previous level
        # the sliding step is set to 1
        self.num_in_each_level = [i for i in range(self.num_stripes, 0, -1)]
        self.num_levels = len(self.num_in_each_level)
        self.num_branches = sum(self.num_in_each_level)

        idx_levels = 0
        for idx_branches in range(self.num_branches):
            if idx_branches >= sum(self.num_in_each_level[0:idx_levels + 1]):
                idx_levels += 1

            if self.used_levels[idx_levels] == 0:
                continue

            pyramid_conv_list.append(nn.Sequential(
                nn.Conv2d(input_size, num_conv_out_channels, 1),
                nn.BatchNorm2d(num_conv_out_channels),
                nn.ReLU(inplace=True)))

        # ============================================================================== pyramid
        idx_levels = 0
        for idx_branches in range(self.num_branches):
            if idx_branches >= sum(self.num_in_each_level[0:idx_levels + 1]):
                idx_levels += 1

            if self.used_levels[idx_levels] == 0:
                continue

            fc = nn.Linear(num_conv_out_channels, self.num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            pyramid_fc_list.append(fc)

    @staticmethod
    def pyramid_forward(self, feat,
                        pyramid_conv_list,
                        pyramid_fc_list,
                        feat_list,
                        logits_list):

        basic_stripe_size = int(feat.size(2) / self.num_stripes)

        idx_levels = 0
        used_branches = 0
        for idx_branches in range(self.num_branches):

            if idx_branches >= sum(self.num_in_each_level[0:idx_levels + 1]):
                idx_levels += 1

            if self.used_levels[idx_levels] == 0:
                continue

            idx_in_each_level = idx_branches - sum(self.num_in_each_level[0:idx_levels])

            stripe_size_in_level = basic_stripe_size * (idx_levels + 1)

            st = idx_in_each_level * basic_stripe_size
            ed = st + stripe_size_in_level

            local_feat = F.avg_pool2d(feat[:, :, st: ed, :],
                                      (stripe_size_in_level, feat.size(-1))) + F.max_pool2d(feat[:, :, st: ed, :],
                                                                                            (stripe_size_in_level,
                                                                                             feat.size(-1)))

            local_feat = pyramid_conv_list[used_branches](local_feat)
            local_feat = local_feat.view(local_feat.size(0), -1)
            feat_list.append(local_feat)

            local_logits = pyramid_fc_list[used_branches](self.dropout_layer(local_feat))
            logits_list.append(local_logits)

            used_branches += 1


def load_ckpt(modules_optims, ckpt_file, load_to_cpu=True, verbose=True, strict=True):
    map_location = (lambda storage, loc: storage) if load_to_cpu else None
    ckpt = torch.load(ckpt_file, map_location=map_location)
    modules_optims[0].load_state_dict(ckpt['state_dicts'][0], strict=strict)
    modules_optims[1].load_state_dict(ckpt['state_dicts'][1])

    if verbose:
        print('Resume from ckpt {}, \nepoch {}, \nscores {}'.format(
            ckpt_file, ckpt['ep'], ckpt['scores']))
    return ckpt['ep'], ckpt['scores']

if __name__ == '__main__':
    market_classes = 751
    duke_classes = 702
    cuhk_classes = 767

    model = Pyramid(num_classes=market_classes)
    # model = model.cuda()
    finetuned_params = list(model.base.parameters())
    # To train from scratch
    new_params = [p for n, p in model.named_parameters()
                  if not n.startswith('base.')]
    param_groups = [{'params': finetuned_params, 'lr': 0.01},
                    {'params': new_params, 'lr': 0.1}]
    optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=5e-4)

    modules_optims = [model, optimizer]
    input = Variable(torch.FloatTensor(8, 3, 384, 128))
    output = model(input)
    print('net output size:')
    print(len(output))
    print(output[0].shape)
    print(output[2].shape)




    # resume_ep, scores = load_ckpt(modules_optims, './market/ckpt_ep112_re02_bs64_dropout02_GPU0_mAP0.882439013042_market.pth')
    print('Resume from EP: {}'.format(resume_ep))
