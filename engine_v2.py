import os
import torch
import numpy as np
import utils.utility as utility
from scipy.spatial.distance import cdist
from utils.functions import cmc, mean_ap, cmc_baseline, eval_liaoxingyu
from utils.re_ranking import re_ranking
import scipy.io
from torchvision import datasets, transforms
from data_v1.sampler import a_RandomIdentitySampler
from loss.multi_similarity_loss import MultiSimilarityLoss
from loss.triplet import CrossEntropyLabelSmooth


class Engine():
    def __init__(self, args, model, optimizer,scheduler,loss, loader, ckpt):
        self.args = args

        # if args.data_train == 'GTA':
        #     transform_train_list = [
        #         # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        #         transforms.Resize((384, 128), interpolation=3),
        #         transforms.Pad(10),
        #         transforms.RandomCrop((384, 128)),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ]
        #     # train_dataset = datasets.ImageFolder(os.path.join(args.datadir, 'pytorch', 'train_all'),
        #     #                                      transforms.Compose(transform_train_list))
        #     train_dataset = datasets.ImageFolder(os.path.join(args.datadir, 'train'),
        #                                          transforms.Compose(transform_train_list))
        #     self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchid * args.batchimage, sampler=a_RandomIdentitySampler(
        #         train_dataset, args.batchid * args.batchimage, args.batchimage), num_workers=8, pin_memory=True)  # 8 workers may work faster
        #     print('GTA has {} classes'.format(train_dataset.classes))
        # else:
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


        #################
        self.weight_t = 1
        self.weight_x = 1
        
        # self.criterion_t = TripletLoss(margin=margin)
        self.criterion_t = MultiSimilarityLoss(margin=args.margin)
        self.criterion_x = CrossEntropyLabelSmooth(702)

        #################

        self.lr = 0.
        # self.optimizer = utility.make_optimizer(args, self.model)
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        # last_epoch = -1

        if torch.cuda.is_available():
            self.ckpt.write_log('[INFO] ' + torch.cuda.get_device_name(0))

        # if args.load != '':

        #     checkpointer = torch.load(os.path.join(ckpt.dir, 'model','optimizer.pt'))
        #     last_epoch = checkpointer['epoch'] - 1
        #     self.optimizer.load_state_dict(checkpointer['state_dict'])

        #     # self.optimizer.load_state_dict(
        #     #     torch.load(os.path.join(ckpt.dir, 'optimizer.pt')))
        #     # last_epoch = int(ckpt.log[-1, 0]) - 1
        #     self.ckpt.write_log('[INFO] Optimizer loaded.')

        #     # for _ in range(last_epoch):
        #     #     self.scheduler.step()

        # if args.pre_train != '' and args.resume:
        #     resume_epoch = args.pre_train.split(
        #         '/')[-1].split('.')[0].split('_')[-1]
        #     # optimizer_path = 
        #     self.optimizer.load_state_dict(
        #         torch.load(args.pre_train.replace('model', 'optimizer'))
        #     )
        #     # for _ in range(len(ckpt.log) * args.test_every):
        #     #     self.scheduler.step()
        #     last_epoch = resume_epoch - 1

        # self.scheduler = utility.make_scheduler(
        #     args, self.optimizer, last_epoch)

        self.ckpt.write_log(
            '[INFO] Continue from epoch {}'.format(self.scheduler.last_epoch))

        print(ckpt.log)
        # print(self.scheduler._last_lr)

    def train(self):
        # self.loss.step()
        epoch = self.scheduler.last_epoch
        lr = self.scheduler.get_last_lr()[0]

        if lr != self.lr:
            self.ckpt.write_log(
                '[INFO] Epoch: {}\tLearning rate: {:.2e}  '.format(epoch + 1, lr))
            self.lr = lr
        self.loss.start_log()
        self.model.train()
        # for k in self.model.state_dict():
        #     print(k)
        #     print(self.model.state_dict()[k].shape)
        #     print(self.model.state_dict()[k].requires_grad)

        for batch, d in enumerate(self.train_loader):
            inputs, labels = self._parse_data_for_train(d)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            # outputs = self.model(inputs)
            # loss = self.loss(outputs, labels)
            
            ###################
            output1, output2,output3,output4, fea = self.model(inputs)
            # loss_c1 = self._compute_loss(self.criterion_c1, fea[0], pids)
            # loss_c2 = self._compute_loss(self.criterion_c2, fea[1], pids)

            loss_t1 = self.criterion_t(fea[0], labels)
            loss_t2 = self.criterion_t(fea[1], labels)

            loss_x1 = self.criterion_x(output1, labels)
            loss_x2 = self.criterion_x(output2, labels)
            loss_x3 = self.criterion_x(output3, labels)
            loss_x4 = self.criterion_x(output4, labels)
            loss1 = (self.weight_x * loss_x1 + self.weight_x * loss_x2+ self.weight_x * loss_x3+ self.weight_x * loss_x4) * 0.5

            loss2 = (self.weight_t * loss_t1 + self.weight_t * loss_t2) * 0.5
            # loss3 = (loss_c1 + loss_c2) * 0.5
            loss3 = 0.

            loss = loss1 + loss2 + 0.0005 * loss3


            loss.backward()
            self.optimizer.step()

            # self.ckpt.write_log('\r[INFO] [{}/{}]\t{}/{}\t{}'.format(
            #     epoch + 1, self.args.epochs,
            #     batch + 1, len(self.train_loader),
            #     self.loss.display_loss(batch)),
            #     end='' if batch + 1 != len(self.train_loader) else '\n')
            self.ckpt.write_log('\r[INFO] [{}/{}]\t{}/{}\t{}'.format(
                epoch + 1, self.args.epochs,
                batch + 1, len(self.train_loader),
                loss),
                end='' if batch + 1 != len(self.train_loader) else '\n')


        self.scheduler.step()
        self.loss.end_log(len(self.train_loader))

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckpt.write_log('\n[INFO] Test:')
        self.model.eval()

        self.ckpt.add_log(torch.zeros(1, 6))
        # qf = self.extract_feature(self.query_loader,self.args).numpy()
        # gf = self.extract_feature(self.test_loader,self.args).numpy()

        qf, query_ids, query_cams = self.extract_feature(
            self.query_loader, self.args)
        gf, gallery_ids, gallery_cams = self.extract_feature(
            self.test_loader, self.args)

        # qf = self.extract_feature(self.query_loader)
        # gf = self.extract_feature(self.test_loader)

        # query_ids = np.asarray(self.queryset.ids)
        # gallery_ids = np.asarray(self.testset.ids)
        # query_cams = np.asarray(self.queryset.cameras)
        # gallery_cams = np.asarray(self.testset.cameras)
        # print(query_ids.shape)
        # print(gallery_ids.shape)
        # print(query_cams.shape)
        # print(gallery_cams.shape)
        # np.save('gf',gf.numpy())
        # np.save('qf',qf.numpy())
        # np.save('qc',query_cams)
        # np.save('gc',gallery_cams)
        # np.save('qi',query_ids)
        # np.save('gi',gallery_ids)
        # qf=np.load('/content/qf.npy')
        # gf=np.load('/content/gf.npy')
        # print('save')
        # result = scipy.io.loadmat('pytorch_result.mat')
        # qf = torch.FloatTensor(result['query_f']).cuda()
        # query_cam = result['query_cam'][0]
        # query_label = result['query_label'][0]
        # gf = torch.FloatTensor(result['gallery_f']).cuda()
        # gallery_cam = result['gallery_cam'][0]
        # gallery_label = result['gallery_label'][0]
        # print(query_cam.shape)
        # print(gallery_cam.shape)
        # print(query_label.shape)
        # print(gallery_label.shape)

        if self.args.re_rank:
            q_g_dist = np.dot(qf, np.transpose(gf))
            q_q_dist = np.dot(qf, np.transpose(qf))
            g_g_dist = np.dot(gf, np.transpose(gf))
            dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        else:
            # dist = cdist(qf, gf,metric='cosine')

            # cosine distance
            dist = 1 - torch.mm(qf, gf.t()).cpu().numpy()

            # m, n = qf.shape[0], gf.shape[0]

            # dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            #     torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            # dist.addmm_(1, -2, qf, gf.t())
            # dist = np.dot(qf,np.transpose(gf))
        # print('2')

        # r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
        #         separate_camera_set=False,
        #         single_gallery_shot=False,
        #         first_match_break=True)
        # m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids,
        #                self.queryset.cameras, self.testset.cameras)
        # r = cmc(dist, query_label, gallery_label, query_cam, gallery_cam,
        #         separate_camera_set=False,
        #         single_gallery_shot=False,
        #         first_match_break=True)
        # m_ap = mean_ap(dist, query_label, gallery_label, query_cam, gallery_cam)
        # r, m_ap = cmc_baseline(dist, query_label, gallery_label, query_cam, gallery_cam,
        #         separate_camera_set=False,
        #         single_gallery_shot=False,
        #         first_match_break=True)
        # r, m_ap = cmc_baseline(dist, query_ids, gallery_ids, query_cams, gallery_cams,
        #                        separate_camera_set=False,
        #                        single_gallery_shot=False,
        #                        first_match_break=True)
        # r,m_ap=eval_liaoxingyu(dist, query_label, gallery_label, query_cam, gallery_cam, 50)
        r, m_ap = eval_liaoxingyu(
            dist, query_ids, gallery_ids, query_cams, gallery_cams, 50)

        self.ckpt.log[-1, 0] = epoch
        self.ckpt.log[-1, 1] = m_ap
        self.ckpt.log[-1, 2] = r[0]
        self.ckpt.log[-1, 3] = r[2]
        self.ckpt.log[-1, 4] = r[4]
        self.ckpt.log[-1, 5] = r[9]
        best = self.ckpt.log.max(0)

        # self.ckpt.write_log(
        #     '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} (Best: {:.4f} @epoch {})'.format(
        #         m_ap,
        #         r[0], r[2], r[4], r[9],
        #         best[0][0],
        #         (best[1][0] + 1) * self.args.test_every
        #     )
        # )
        self.ckpt.write_log(
            '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} (Best: {:.4f} @epoch {})'.format(
                m_ap,
                r[0], r[2], r[4], r[9],
                best[0][1], self.ckpt.log[best[1][1], 0]
            )
        )
        # if not self.args.test_only:
        #     self.ckpt.save(self, epoch, is_best=(
        #         (best[1][0] + 1) * self.args.test_every == epoch))
        if not self.args.test_only:
            # self.ckpt.save(self, epoch, is_best=(
            #     self.ckpt.log[best[1][1], 0] == epoch))
            self._save_checkpoint(epoch,r[0],self.ckpt.dir,is_best=(
                self.ckpt.log[best[1][1], 0] == epoch))

    def fliphor(self, inputs):
        inv_idx = torch.arange(inputs.size(
            3) - 1, -1, -1).long()  # N x C x H x W
        return inputs.index_select(3, inv_idx)

    # def extract_feature(self, loader):
    #     features = torch.FloatTensor()
    #     for (inputs, labels) in loader:
    #         ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
    #         for i in range(2):
    #             if i == 1:
    #                 inputs = self.fliphor(inputs)
    #             input_img = inputs.to(self.device)
    #             outputs = self.model(input_img)
    #             f = outputs[0].data.cpu()
    #             ff = ff + f

    #         fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    #         ff = ff.div(fnorm.expand_as(ff))

    #         features = torch.cat((features, ff), 0)
    #     return features
    def extract_feature(self, loader, args):
        features = torch.FloatTensor()
        pids, camids = [], []

        for d in loader:
            inputs, pid, camid = self._parse_data_for_eval(d)
            input_img = inputs.to(self.device)
            outputs = self.model(input_img)
            # print(outputs.shape)
            if args.feat_inference == 'after':
                # f1 = outputs[0].data.cpu()
                # # # flip
                # inputs = inputs.index_select(
                #     3, torch.arange(inputs.size(3) - 1, -1, -1))
                # input_img = inputs.to(self.device)
                # outputs = self.model(input_img)
                # f2 = outputs[0].data.cpu()

                f1 = outputs.data.cpu()
                # # flip
                inputs = inputs.index_select(
                    3, torch.arange(inputs.size(3) - 1, -1, -1))
                input_img = inputs.to(self.device)
                outputs = self.model(input_img)
                f2 = outputs.data.cpu()

                # f2=0
                # print('kkkkk')
            else:
                f1 = outputs[-1].data.cpu()
                # flip
                inputs = inputs.index_select(
                    3, torch.arange(inputs.size(3) - 1, -1, -1))
                input_img = inputs.to(self.device)
                outputs = self.model(input_img)
                f2 = outputs[-1].data.cpu()

            ff = f1 + f2
            if ff.dim() == 3:
                fnorm = torch.norm(
                    ff, p=2, dim=1, keepdim=True) * np.sqrt(ff.shape[2])
                ff = ff.div(fnorm.expand_as(ff))
                ff = ff.view(ff.size(0), -1)
                # ff = ff.view(ff.size(0), -1)
                # fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                # ff = ff.div(fnorm.expand_as(ff))

            else:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))
                # pass
            # fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            # ff = ff.div(fnorm.expand_as(ff))
            features = torch.cat((features, ff), 0)
            pids.extend(pid)
            camids.extend(camid)
            # print(features.shape)
        return features, np.asarray(pids), np.asarray(camids)

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1

            return epoch > self.args.epochs

    # tools for reid datamanager
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
        utility.save_checkpoint(
            {
                'state_dict': self.model.state_dict(),
                'epoch': epoch,
                'rank1': rank1,
                'optimizer': self.optimizer.state_dict(),
                # 'scheduler': self.scheduler.state_dict(),
            },
            save_dir,
            is_best=is_best
        )


