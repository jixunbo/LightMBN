import os
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import os.path as osp

import yaml
from collections import OrderedDict
from shutil import copyfile, copytree
import pickle
import warnings

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.log = torch.Tensor()
        self.since = datetime.datetime.now()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '':
            if args.save == '':
                args.save = now
            self.dir = ROOT_PATH + '/experiment/' + args.save

            # Only works if using google drive
            if ROOT_PATH[:8] == '/content':
                self.model_save_dir = osp.join(
                    ROOT_PATH, '..', '..', 'experiment' + args.save)
            else:
                self.model_save_dir = 'none'

        else:
            self.dir = ROOT_PATH + '/experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = ''
            else:
                # pass
                # if args.resume != 0:
                #     self.add_log(torch.tensor(
                #         [args.resume, 0, 0, 0, 0, 0], dtype=torch.float32).reshape(1, 6))
                # else:
                #     self.log = torch.load(self.dir + '/map_log.pt')
                if os.path.exists(self.dir + '/map_log.pt'):
                    self.log = torch.load(self.dir + '/map_log.pt')
                # print('Continue from epoch {}...'.format(
                #     len(self.log) * args.test_every))

        print('Experiment results will be saved in {} '.format(self.dir))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.dir)

        if not args.test_only:

            # _make_dir(self.dir + '/model')
            _make_dir(self.dir + '/scripts')

            # copytree(os.path.join(ROOT_PATH, 'model'), self.dir + '/scripts/model' +
            #          datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
            # copytree(os.path.join(ROOT_PATH, 'loss'), self.dir + '/scripts/loss' +
            #          datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        with open(self.dir + '/config.yaml', open_type) as fp:
            dic = vars(args).copy()
            del dic['load'], dic['save'], dic['pre_train'], dic['test_only'], dic['re_rank']
            yaml.dump(dic, fp, default_flow_style=False)

    # def save(self, trainer, epoch, is_best=False):
    #     trainer.model.save(self.dir, epoch, is_best=is_best)
    #     trainer.loss.save(self.dir)
    #     # trainer.loss.plot_loss(self.dir, epoch)

    #     self.plot_map_rank(epoch)
    #     torch.save(self.log, os.path.join(self.dir, 'map_log.pt'))
    #     torch.save({'state_dict': trainer.optimizer.state_dict(), 'epoch': epoch},
    #                os.path.join(self.dir, 'model',
    #                             'optimizer.pt')
    #                )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False, end='\n'):
        time_elapsed = (datetime.datetime.now() - self.since).seconds
        log = log + ' Time used: {} m {} s'.format(
            time_elapsed // 60, time_elapsed % 60)
        print(log, end=end)
        if end != '':
            self.log_file.write(log + end)
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_map_rank(self, epoch):
        axis = np.linspace(1, epoch, self.log.size(0))
        label = 'Reid on {}'.format(self.args.data_test)
        labels = ['mAP', 'rank1', 'rank3', 'rank5', 'rank10']
        fig = plt.figure()
        plt.title(label)
        for i in range(len(labels)):
            plt.plot(axis, self.log[:, i + 1].numpy(), label=labels[i])

        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('mAP/rank')
        plt.grid(True)
        plt.savefig('{}/test_{}.jpg'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        pass

    def save_checkpoint(
        self, state, save_dir, is_best=False, remove_module_from_keys=False
    ):
        r"""Saves checkpoint.

        Args:
            state (dict): dictionary.
            save_dir (str): directory to save checkpoint.
            is_best (bool, optional): if True, this checkpoint will be copied and named
                ``model-best.pth.tar``. Default is False.
            remove_module_from_keys (bool, optional): whether to remove "module."
                from layer names. Default is False.

        Examples::
            >>> state = {
            >>>     'state_dict': model.state_dict(),
            >>>     'epoch': 10,
            >>>     'rank1': 0.5,
            >>>     'optimizer': optimizer.state_dict()
            >>> }
            >>> save_checkpoint(state, 'log/my_model')
        """
        def mkdir_if_missing(dirname):
            """Creates dirname if it is missing."""
            if not osp.exists(dirname):
                try:
                    os.makedirs(dirname)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
        mkdir_if_missing(save_dir)
        if remove_module_from_keys:
            # remove 'module.' in state_dict's keys
            state_dict = state['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                new_state_dict[k] = v
            state['state_dict'] = new_state_dict
        # save
        # fpath = osp.join(save_dir, 'model.pth.tar-' + str(epoch))
        fpath = osp.join(save_dir, 'model-latest.pth')
        torch.save(state, fpath)
        self.write_log('[INFO] Checkpoint saved to "{}"'.format(fpath))
        if is_best:
            # shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model-best.pth.tar'))
            torch.save(state['state_dict'], osp.join(
                save_dir, 'model-best.pth'))
        if 'log' in state.keys():

            torch.save(state['log'], os.path.join(save_dir, 'map_log.pt'))

    def load_checkpoint(self, fpath):
        # """Loads checkpoint.
        # ``UnicodeDecodeError`` can be well handled, which means
        # python2-saved files can be read from python3.
        # Args:
        #     fpath (str): path to checkpoint.
        # Returns:
        #     dict
        # Examples::
        #     >>> from torchreid.utils import load_checkpoint
        #     >>> fpath = 'log/my_model/model.pth.tar-10'
        #     >>> checkpoint = load_checkpoint(fpath)
        # """
        if fpath is None:
            raise ValueError('File path is None')
        if not osp.exists(fpath):
            raise FileNotFoundError('File is not found at "{}"'.format(fpath))
        map_location = None if torch.cuda.is_available() else 'cpu'
        try:
            checkpoint = torch.load(fpath, map_location=map_location)
        except UnicodeDecodeError:
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            checkpoint = torch.load(
                fpath, pickle_module=pickle, map_location=map_location
            )
        except Exception:
            print('Unable to load checkpoint from "{}"'.format(fpath))
            raise
        return checkpoint

    def load_pretrained_weights(self, model, weight_path):
        r"""Loads pretrianed weights to model.
        Features::
            - Incompatible layers (unmatched in name or size) will be ignored.
            - Can automatically deal with keys containing "module.".
        Args:
            model (nn.Module): network model.
            weight_path (str): path to pretrained weights.
        Examples::
            >>> from torchreid.utils import load_pretrained_weights
            >>> weight_path = 'log/my_model/model-best.pth.tar'
            >>> load_pretrained_weights(model, weight_path)
        """
        checkpoint = self.load_checkpoint(weight_path)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model_dict = model.state_dict()
        # print(model_dict.keys())
        new_state_dict = OrderedDict()
        matched_layers, discarded_layers = [], []
        for k, v in state_dict.items():
            # print(k)
            if k.startswith('module.'):
                k = 'model.' + k[7:]  # discard module.

            if k in model_dict and model_dict[k].size() == v.size():

                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)

        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)

        if len(matched_layers) == 0:
            warnings.warn(
                'The pretrained weights "{}" cannot be loaded, '
                'please check the key names manually '
                '(** ignored and continue **)'.format(weight_path)
            )
        else:
            self.write_log('[INFO] Successfully loaded pretrained weights from "{}"'.
                           format(weight_path))
            if len(discarded_layers) > 0:
                print(
                    '** The following layers are discarded '
                    'due to unmatched keys or layer size: {}'.
                    format(discarded_layers)
                )

    def resume_from_checkpoint(self, fpath, model, optimizer=None, scheduler=None):
        r"""Resumes training from a checkpoint.

        This will load (1) model weights and (2) ``state_dict``
        of optimizer if ``optimizer`` is not None.

        Args:
            fpath (str): path to checkpoint.
            model (nn.Module): model.
            optimizer (Optimizer, optional): an Optimizer.
            scheduler (LRScheduler, optional): an LRScheduler.

        Returns:
            int: start_epoch.

        Examples::
            >>> from torchreid.utils import resume_from_checkpoint
            >>> fpath = 'log/my_model/model.pth.tar-10'
            >>> start_epoch = resume_from_checkpoint(
            >>>     fpath, model, optimizer, scheduler
            >>> )
        """
        self.write_log('[INFO] Loading checkpoint from "{}"'.format(fpath))
        checkpoint = self.load_checkpoint(fpath)

        model.load_state_dict(checkpoint['state_dict'])
        self.write_log('[INFO] Model weights loaded')
        if optimizer is not None and 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            self.write_log('[INFO] Optimizer loaded')
        if scheduler is not None and 'scheduler' in checkpoint.keys():
            scheduler.load_state_dict(checkpoint['scheduler'])
            self.write_log('[INFO] Scheduler loaded')
        start_epoch = checkpoint['epoch']
        self.write_log('[INFO] Last epoch = {}'.format(start_epoch))
        if 'rank1' in checkpoint.keys():
            self.write_log(
                '[INFO] Last rank1 = {:.1%}'.format(checkpoint['rank1']))
        if 'log' in checkpoint.keys():
            self.log = checkpoint['log']

        return start_epoch, model, optimizer
