import torch
import numpy as np
from utils.functions import evaluation
from utils.re_ranking import re_ranking, re_ranking_gpu


class Engine():
    def __init__(self, args, model, optimizer, scheduler, loss, loader, ckpt):
        self.args = args
        self.train_loader = loader.train_loader
        self.test_loader = loader.test_loader
        self.query_loader = loader.query_loader
        self.testset = loader.galleryset
        self.queryset = loader.queryset

        self.ckpt = ckpt
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss

        self.lr = 0.
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        if torch.cuda.is_available():
            self.ckpt.write_log('[INFO] GPU: ' + torch.cuda.get_device_name(0))

        self.ckpt.write_log(
            '[INFO] Starting from epoch {}'.format(self.scheduler.last_epoch + 1))

    def train(self):

        epoch = self.scheduler.last_epoch
        lr = self.scheduler.get_last_lr()[0]

        if lr != self.lr:
            self.ckpt.write_log(
                '[INFO] Epoch: {}\tLearning rate: {:.2e}  '.format(epoch + 1, lr))
            self.lr = lr
        self.loss.start_log()
        self.model.train()

        for batch, d in enumerate(self.train_loader):
            inputs, labels = self._parse_data_for_train(d) # get inputs = d[0] and labels = d[1]

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs) #input: torch.Size([16, 3, 384, 128])
                                         # outputs[0][-1].size(): torch.Size([16, 202])
            loss = self.loss.compute(outputs, labels)

            loss.backward()
            self.optimizer.step()

            self.ckpt.write_log('\r[INFO] [{}/{}]\t{}/{}\t{}'.format(
                epoch + 1, self.args.epochs,
                batch + 1, len(self.train_loader),
                self.loss.display_loss(batch)),
                end='' if batch + 1 != len(self.train_loader) else '\n')

        self.scheduler.step()
        self.loss.end_log(len(self.train_loader))
        # self._save_checkpoint(epoch, 0., self.ckpt.dir, is_best=True)

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckpt.write_log('\n[INFO] Test:')
        self.model.eval()

        self.ckpt.add_log(torch.zeros(1, 6))

        with torch.no_grad():

            qf_flip, qf_wof, query_ids, query_cams = self.extract_feature(
                self.query_loader, self.args)
            gf_flip, gf_wof, gallery_ids, gallery_cams = self.extract_feature(
                self.test_loader, self.args)


        # q_g_dist = np.dot(qf, np.transpose(gf))
        # q_q_dist = np.dot(qf, np.transpose(qf))
        # g_g_dist = np.dot(gf, np.transpose(gf))
        # dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        lambda_value = self.args.re_rank_lambda_value
        k1 = self.args.re_rank_k1
        k2 = self.args.re_rank_k2

        dist_Rerank_flip = re_ranking_gpu(qf_flip, gf_flip, k1, k2, lambda_value)
        #dist_Rerank_wof = re_ranking_gpu(qf_wof, gf_wof, k1, k2, lambda_value)
        dist_flip = 1 - torch.mm(qf_flip, gf_flip.t()).cpu().numpy()
        r_flip, m_ap_flip = evaluation(dist_flip, query_ids, gallery_ids, query_cams, gallery_cams, 50)
        #dist_wof = 1 - torch.mm(qf_wof, gf_wof.t()).cpu().numpy()
        if m_ap_flip > 0.5:
            lambda_value = m_ap_flip
            dist_Rerank_flip2 = re_ranking_gpu(qf_flip, gf_flip, k1, k2, lambda_value)
            dist = dist_Rerank_flip
            printing = 'ReRank+Lamda=mAP '

        for i in range(3):
            if i == 0:
                dist = dist_Rerank_flip
                printing = 'ReRank+Lamda=0.5'
            elif i == 1:
                dist = dist_flip
                printing = 'Normal+Distance'
            elif i == 2 and m_ap_flip > 0.5:
                dist = dist_Rerank_flip2
                printing = 'ReRank+Lamda=mAP '


            if not i == 1:
                r, m_ap = evaluation(dist, query_ids, gallery_ids, query_cams, gallery_cams, 50)
            else:
                r = r_flip
                m_ap = m_ap_flip


            self.ckpt.log[-1, 0] = epoch
            self.ckpt.log[-1, 1] = m_ap
            self.ckpt.log[-1, 2] = r[0]
            self.ckpt.log[-1, 3] = r[2]
            self.ckpt.log[-1, 4] = r[4]
            self.ckpt.log[-1, 5] = r[9]
            best = self.ckpt.log.max(0)

            self.ckpt.write_log(
                '[INFO] {} mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}, {} (Best: {:.4f} @epoch {})'.format(
                    i+1,
                    m_ap,
                    r[0], r[2], r[4], r[9],
                    printing,
                    best[0][1], self.ckpt.log[best[1][1], 0]
                ), refresh=True
            )

        if not self.args.test_only:

            self._save_checkpoint(epoch, r[0], self.ckpt.dir, is_best=(
                self.ckpt.log[best[1][1], 0] == epoch))
            self.ckpt.plot_map_rank(epoch)

    def fliphor(self, inputs):
        inv_idx = torch.arange(inputs.size(
            3) - 1, -1, -1).long()  # N x C x H x W
        return inputs.index_select(3, inv_idx)

    def extract_feature(self, loader, args):
        features_wof = torch.FloatTensor()
        features_flip = torch.FloatTensor()
        pids, camids = [], []

        for d in loader:
            inputs, pid, camid = self._parse_data_for_eval(d)
            input_img = inputs.to(self.device)
            outputs = self.model(input_img)

            f1 = outputs.data.cpu()
            # flip
            inputs = inputs.index_select(
                3, torch.arange(inputs.size(3) - 1, -1, -1))
            input_img = inputs.to(self.device)
            outputs = self.model(input_img)
            f2 = outputs.data.cpu()

            ff1 = f1 + f2

            ff2 = f1
            for i in range(2):
                if i == 0:
                    ff = ff1
                else:
                    ff = ff2
                if ff.dim() == 3:
                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)  # * np.sqrt(ff.shape[2])
                    ff = ff.div(fnorm.expand_as(ff))
                    ff = ff.view(ff.size(0), -1)

                else:
                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                    ff = ff.div(fnorm.expand_as(ff))


                if i == 0:
                    features_flip = torch.cat((features_flip, ff), 0)

                else:
                    features_wof = torch.cat((features_wof, ff), 0)


            pids.extend(pid)
            camids.extend(camid)


        return features_flip, features_wof, np.asarray(pids), np.asarray(camids)

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1

            return epoch > self.args.epochs

    # tools for reid datamanager data_v2
    def _parse_data_for_train(self, data):
        imgs = data[0]
        pids = data[1]
        return imgs, pids

    def _parse_data_for_eval(self, data):
        imgs = data[0]
        pids = data[1]
        camids = data[2]
        return imgs, pids, camids

    def _save_checkpoint(self, epoch, rank1, save_dir, is_best=False):
        self.ckpt.save_checkpoint(
            {
                'state_dict': self.model.state_dict(),
                'epoch': epoch,
                'rank1': rank1,
                'optimizer': self.optimizer.state_dict(),
                'log': self.ckpt.log,
                # 'scheduler': self.scheduler.state_dict(),
            },
            save_dir,
            is_best=is_best
        )
