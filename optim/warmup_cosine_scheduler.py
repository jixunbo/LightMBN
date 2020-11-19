import torch
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim.lr_scheduler as lrs


class WarmupCosineAnnealingLR(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        warmup_epoch: target learning rate is reached at warmup_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, warmup_epoch, epochs,  last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.warmup_epoch = warmup_epoch
        self.last_epoch = last_epoch
        if last_epoch <10:
            self.after_scheduler = lrs.CosineAnnealingLR(
                optimizer, float(epochs-warmup_epoch),last_epoch=-1
            )
        else:
            self.after_scheduler = lrs.CosineAnnealingLR(
                optimizer, float(epochs-warmup_epoch),last_epoch=last_epoch-warmup_epoch
            )
        # self.after_scheduler = after_scheduler
        self.finished = False

        super(WarmupCosineAnnealingLR, self).__init__(optimizer,last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.warmup_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.warmup_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.warmup_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    # print('koooo',self.last_epoch)
                    self.after_scheduler.step()
                    self.last_epoch += 1

                else:
                    self.after_scheduler.step(epoch - self.warmup_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
                # self.last_epoch = self.after_scheduler.last_epoch + self.warmup_epoch
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


if __name__ == '__main__':
    v = torch.zeros(10)
    optim = torch.optim.SGD([v], lr=3.5e-4)

    scheduler = GradualWarmupScheduler(optim, multiplier=1, warmup_epoch=10, epochs=120,last_epoch=-1)
    print(dir(scheduler))
    a = []
    b = []
    for i in range(1, 120):
        scheduler.step()
        print('kk', scheduler.get_last_lr())
        print('3333333',scheduler.last_epoch)
        a.append(scheduler.last_epoch)
        b.append(optim.param_groups[0]['lr'])
        print(i, optim.param_groups[0]['lr'])

        # plt.plot(t, s, "o-", lw=4.1)
    # plt.plot(t, s2, "o-", lw=4.1)

    tick_spacing = 5
    plt.plot(a,b,"-", lw=2)

    # plt.xticks(3.5e-4)


    # plt.plot(n, m1, 'r-.', n, m2, 'b')

    # plt.xlim((-2, 4))
    # plt.ylim((-5, 15))

    # x_ticks = np.linspace(-5, 4, 10)
    # plt.xticks(x_ticks)

    # 将对应标度位置的数字替换为想要替换的字符串，其余为替换的不再显示
    plt.yticks([3.5e-5, 3.5e-4], ['3.5e-5','3.5e-4'])

    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")

    # plt.show()

    # plt.ylabel()

    # # plt.title("Simple plot $\\frac{\\alpha}{2}$")
    # # plt.grid(True)

    # plt.show()
# import matplotlib2tikz

# matplotlib2tikz.save("test.tex")

