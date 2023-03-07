import os
import numpy as np
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from .triplet import TripletLoss, TripletSemihardLoss, CrossEntropyLabelSmooth
from .grouploss import GroupLoss
from loss.multi_similarity_loss import MultiSimilarityLoss
from loss.focal_loss import FocalLoss
from loss.osm_caa_loss import OSM_CAA_Loss
from loss.center_loss import CenterLoss
#from loss.build import make_loss_MALW
from .metric_learning import ContrastiveLoss
from .metric_learning import ContrastiveLoss, SupConLoss
from .hard_mine_triplet_loss import TripletLoss_mine

class LossFunction():
    def __init__(self, args, ckpt):
        super(LossFunction, self).__init__()
        ckpt.write_log('[INFO] Making loss...')

        self.nGPU = args.nGPU
        self.args = args
        self.loss = []
        self.MALW = args.MALW_active
        self.count = 0
        self.update_iter_interval = args.MALW_update_iter
        self.cross_en_loss_history = []
        self.metric_loss_history = []
        self.ckpt = ckpt

        if self.MALW:
            # -------- ID Loss ------- #
            if args.if_labelsmooth:
                loss_function = CrossEntropyLabelSmooth(
                    num_classes=args.num_classes)
                ckpt.write_log('[INFO] Label Smoothing On.')
            else:
                loss_function = nn.CrossEntropyLoss()

            self.ID_LOSS_WEIGHT = float(args.ID_LOSS_WEIGHT)
            self.loss.append({
                'type': 'CrossEntropy',
                'weight': float(self.ID_LOSS_WEIGHT),
                'function': loss_function
                })
            # -------- Matric Loss ------- #
            if args.METRIC_LOSS_TYPE_MALW == 'Triplet':
                loss_function = TripletLoss(args.margin)
                self.MATRIC_LOSS_WEIGHT = float(args.METRIC_LOSS_WEIGHT)
                self.loss.append({
                    'type': args.METRIC_LOSS_TYPE_MALW,
                    'weight': self.MATRIC_LOSS_WEIGHT,
                    'function': loss_function
                })
            if args.METRIC_LOSS_TYPE_MALW == 'Triplet_mine':
                loss_function = TripletLoss_mine(args.margin)
                self.MATRIC_LOSS_WEIGHT = float(args.METRIC_LOSS_WEIGHT)
                self.loss.append({
                    'type': args.METRIC_LOSS_TYPE_MALW,
                    'weight': self.MATRIC_LOSS_WEIGHT,
                    'function': loss_function
                })
            elif args.METRIC_LOSS_TYPE_MALW == 'Contrastive':
                loss_function = ContrastiveLoss(args.margin)
                self.MATRIC_LOSS_WEIGHT = float(args.METRIC_LOSS_WEIGHT)
                self.loss.append({
                    'type': args.METRIC_LOSS_TYPE_MALW,
                    'weight': self.MATRIC_LOSS_WEIGHT,
                    'function': loss_function
                })


        else:
            for loss in args.loss.split('+'):
                weight, loss_type = loss.split('*')
                if loss_type == 'CrossEntropy':
                    if args.if_labelsmooth:
                        loss_function = CrossEntropyLabelSmooth(
                            num_classes=args.num_classes)
                        ckpt.write_log('[INFO] Label Smoothing On.')
                    else:
                        loss_function = nn.CrossEntropyLoss()
                    self.ID_LOSS_WEIGHT = float(weight)
                elif loss_type == 'Triplet':
                    loss_function = TripletLoss(args.margin)
                    self.MATRIC_LOSS_WEIGHT = float(weight)
                elif loss_type == 'Triplet_mine':
                    loss_function = TripletLoss_mine(args.margin)
                    self.MATRIC_LOSS_WEIGHT = float(weight)
                elif loss_type == 'GroupLoss':
                    loss_function = GroupLoss(
                        total_classes=args.num_classes, max_iter=args.T, num_anchors=args.num_anchors)
                elif loss_type == 'MSLoss':
                    loss_function = MultiSimilarityLoss(margin=args.margin)
                elif loss_type == 'Focal':
                    loss_function = FocalLoss(reduction='mean')
                elif loss_type == 'OSLoss':
                    loss_function = OSM_CAA_Loss()
                elif loss_type == 'CenterLoss':
                    loss_function = CenterLoss(
                        num_classes=args.num_classes, feat_dim=args.feats)

                self.loss.append({
                    'type': loss_type,
                    'weight': float(weight),
                    'function': loss_function
                })

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        self.log = torch.Tensor()

    def compute(self, outputs, labels):
        losses = []

        if self.MALW:
            #_id_loss = id_loss_func(score, target)
            #_metric_loss = metric_loss_func(feat, target)
            #self.cross_en_loss_history.append(_id_loss.item())
            #self.metric_loss_history.append(_metric_loss.item())
            if len(self.cross_en_loss_history) == 0:
                pass
            elif (len(self.cross_en_loss_history) % self.update_iter_interval == 0):

                _id_history = np.array(self.cross_en_loss_history)
                id_mean = _id_history.mean()
                id_std = _id_history.std()

                _metric_history = np.array(self.metric_loss_history)
                metric_mean = _metric_history.mean()
                metric_std = _metric_history.std()

                id_weighted = id_std
                metric_weighted = metric_std
                if id_weighted > metric_weighted:
                    new_weight = 1 - (id_weighted - metric_weighted) / id_weighted
                    self.ID_LOSS_WEIGHT = self.ID_LOSS_WEIGHT * 0.9 + new_weight * 0.1
                    # self.ID_LOSS_WEIGHT = weight CrossEntropy

                self.cross_en_loss_history = []
                self.metric_loss_history = []
                self.ckpt.write_log('\n[INFO] update weighted loss ID_LOSS_WEIGHT={},{}_LOSS_WEIGHT={}'.format(round(self.ID_LOSS_WEIGHT, 3),self.args.METRIC_LOSS_TYPE_MALW.upper() ,self.MATRIC_LOSS_WEIGHT))
         #       print('\n')
          #      print(
           #         f"update weighted loss ID_LOSS_WEIGHT={round(self.ID_LOSS_WEIGHT, 3)},MATRIC_LOSS_WEIGHT={self.MATRIC_LOSS_WEIGHT}")



        for i, l in enumerate(self.loss):
            if l['type'] in ['CrossEntropy']:

                if isinstance(outputs[0], list):
                    loss = [l['function'](output, labels)
                            for output in outputs[0]]
                elif isinstance(outputs[0], torch.Tensor):
                    loss = [l['function'](outputs[0], labels)]
                else:
                    raise TypeError(
                        'Unexpected type: {}'.format(type(outputs[0])))

                loss = sum(loss)
                if self.MALW:
                    effective_loss = self.ID_LOSS_WEIGHT * loss
                else:
                    effective_loss = l['weight'] * loss
                losses.append(effective_loss)

                self.log[-1, i] += effective_loss.item()
                self.cross_en_loss_history.append(effective_loss.item())

            elif l['type'] in ['Triplet', 'MSLoss', 'Contrastive', 'Triplet_mine']:
                if isinstance(outputs[-1], list):
                    loss = [l['function'](output, labels)
                            for output in outputs[-1]]
                elif isinstance(outputs[-1], torch.Tensor):
                    loss = [l['function'](outputs[-1], labels)]
                else:
                    raise TypeError(
                        'Unexpected type: {}'.format(type(outputs[-1])))
                loss = sum(loss)
                if self.MALW:
                    effective_loss = self.MATRIC_LOSS_WEIGHT * loss
                else:
                    effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
                self.metric_loss_history.append(effective_loss.item())

            elif l['type'] in ['GroupLoss']:
                if isinstance(outputs[-1], list):
                    loss = [l['function'](output[0], labels, output[1])
                            for output in zip(outputs[-1], outputs[0][:3])]
                elif isinstance(outputs[-1], torch.Tensor):
                    loss = [l['function'](outputs[-1], labels)]
                else:
                    raise TypeError(
                        'Unexpected type: {}'.format(type(outputs[-1])))
                loss = sum(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()

            elif l['type'] in ['CenterLoss']:
                if isinstance(outputs[-1], list):
                    loss = [l['function'](output, labels)
                            for output in outputs[-1]]
                elif isinstance(outputs[-1], torch.Tensor):
                    loss = [l['function'](outputs[-1], labels)]
                else:
                    raise TypeError(
                        'Unexpected type: {}'.format(type(outputs[-1])))

                loss = sum(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()

            else:
                pass

        loss_sum = sum(losses)
        self.count_iter()

        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, batches):
        self.log[-1].div_(batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.6f}]'.format(l['type'], c / n_samples)) # have to devide n_semles to the for loop in def compute

        return ''.join(log)

    def count_iter(self):
        self.count = self.count + 1


    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)

            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.jpg'.format(apath, l['type']))
            plt.close(fig)

    # Following codes not being used

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def get_loss_module(self):
        if self.nGPU == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)):
                    l.scheduler.step()


def make_loss(args, ckpt):
    return LossFunction(args, ckpt)
