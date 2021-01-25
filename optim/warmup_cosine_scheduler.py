# encoding: utf-8


import torch
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim.lr_scheduler as lrs

import math


class WarmupCosineAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, multiplier, warmup_epoch, epochs, min_lr=3.5e-7, last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError(
                'multiplier should be greater thant or equal to 1.')
        self.warmup_epoch = warmup_epoch
        self.last_epoch = last_epoch
        self.eta_min = min_lr
        self.T_max = float(epochs - warmup_epoch)
        self.after_scheduler = True

        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch - 1:

            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch -
                                             self.warmup_epoch) / (self.T_max - 1))) / 2
                    for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch + 1) / self.warmup_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.) for base_lr in self.base_lrs]



if __name__ == '__main__':
    v = torch.zeros(10)
    optim1 = torch.optim.SGD([v], lr=3.5e-4)

    scheduler2 = WarmupCosineAnnealingLR(
        optim1, multiplier=1, warmup_epoch=10, epochs=120, min_lr=3.5e-7,last_epoch=-1)

    a = []
    b = []
    for i in range(1, 121):
        
        print('kk1', scheduler2.get_last_lr())
        print('3333333', scheduler2.last_epoch+1)
        if scheduler2.last_epoch ==120:
            break
        a.append(scheduler2.last_epoch+1)
        b.append(optim1.param_groups[0]['lr'])
        print(i, optim1.param_groups[0]['lr'])
        # optim.step()
        scheduler2.step()    


    print(dir(scheduler))
    tick_spacing = 5
    plt.figure(figsize=(20,10))
    plt.rcParams['figure.dpi'] = 300 #分辨率

    plt.plot(a, b, "-", lw=2)


    plt.yticks([3.5e-5, 3.5e-4], ['3.5e-5', '3.5e-4'])

    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")



    optim = torch.optim.SGD([v], lr=3.5e-4)
    scheduler1 = WarmupCosineAnnealingLR(
        optim, multiplier=1, warmup_epoch=10, epochs=120, min_lr=3.5e-7,last_epoch=-1)

    a = []
    b = []
    for i in range(1, 71):
        
        print('kk1', scheduler1.get_last_lr())
        print('3333333', scheduler1.last_epoch+1)
        if scheduler1.last_epoch ==120:
            break
        a.append(scheduler1.last_epoch+1)
        b.append(optim.param_groups[0]['lr'])
        print(i, optim.param_groups[0]['lr'])
        # optim.step()
        scheduler1.step()    

    scheduler = WarmupCosineAnnealingLR(
        optim, multiplier=1, warmup_epoch=10, epochs=120, min_lr=3.5e-7,last_epoch=69)
    print(dir(scheduler))
    tick_spacing = 5
    plt.plot(a, b, "-", lw=2)

    # plt.xticks(3.5e-4)

    # plt.plot(n, m1, 'r-.', n, m2, 'b')

    # plt.xlim((-2, 4))
    # plt.ylim((-5, 15))

    # x_ticks = np.linspace(-5, 4, 10)
    # plt.xticks(x_ticks)

    # 将对应标度位置的数字替换为想要替换的字符串，其余为替换的不再显示
    plt.yticks([3.5e-5, 3.5e-4], ['3.5e-5', '3.5e-4'])

    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    a = []
    b = []
    for i in range(1, 120):
        
        print('kk', scheduler.get_last_lr())
        print('3333333', scheduler.last_epoch+1)
        if scheduler.last_epoch ==126:
            break
        a.append(scheduler.last_epoch+1)
        b.append(optim.param_groups[0]['lr'])
        print(i, optim.param_groups[0]['lr'])
        optim.step()
        scheduler.step()  

        # plt.plot(t, s, "o-", lw=4.1)
    # plt.plot(t, s2, "o-", lw=4.1)

    tick_spacing = 10
    plt.plot(a, b, "-", lw=2)

    # plt.xticks(3.5e-4)

    # plt.plot(n, m1, 'r-.', n, m2, 'b')

    # plt.xlim((-2, 4))
    # plt.ylim((-5, 15))

    # x_ticks = np.linspace(-5, 4, 10)
    # plt.xticks(x_ticks)

    # 将对应标度位置的数字替换为想要替换的字符串，其余为替换的不再显示
    plt.yticks([3.5e-5, 3.5e-4], ['3.5e-5', '3.5e-4'])

    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
