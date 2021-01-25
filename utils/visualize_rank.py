from __future__ import print_function, absolute_import
import shutil

import sys
import os
import numpy as np
import os.path as osp
import cv2
import torch
from torch.nn import functional as F

sys.path.append(osp.join(osp.abspath(osp.dirname(__file__)), ".."))

import data_v2
import loss
import optim
from model import Model
from option import args
import utils.utility as utility
from utils.model_complexity import compute_model_complexity
import yaml


__all__ = ['visualize_ranked_results']

GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
BW = 5  # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def visualize_ranked_results(
    distmat, query_loader, gallery_loader, data_type, width=128, height=256, save_dir='', topk=10
):
    """Visualizes ranked results.
    Supports both image-reid and video-reid.
    For image-reid, ranks will be plotted in a single figure. For video-reid, ranks will be
    saved in folders each containing a tracklet.
    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid, dsetid).
        data_type (str): "image" or "video".
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    num_q, num_g = distmat.shape
    print(num_q,num_g)
    def mkdir_if_missing(s_dir):

        if not osp.exists(s_dir):
            os.makedirs(s_dir)

    mkdir_if_missing(save_dir)
    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Visualizing top-{} ranks ...'.format(topk))

    query = query_loader
    gallery = gallery_loader
    print(len(query),len(gallery))

    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(distmat, axis=1)

    def _cp_img_to(src, dst, rank, prefix, matched=False):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
            matched: bool
        """
        if isinstance(src, (tuple, list)):
            if prefix == 'gallery':
                suffix = 'TRUE' if matched else 'FALSE'
                dst = osp.join(
                    dst, prefix + '_top' + str(rank).zfill(3)
                ) + '_' + suffix
            else:
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(
                dst, prefix + '_top' + str(rank).zfill(3) + '_name_' +
                osp.basename(src)
            )
            shutil.copy(src, dst)
    # num_q =200


    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx][:3]
        qimg_path_name = qimg_path[0] if isinstance(
            qimg_path, (tuple, list)
        ) else qimg_path

        if data_type == 'image':
            qimg = cv2.imread(qimg_path)
            qimg = cv2.resize(qimg, (width, height))
            qimg = cv2.copyMakeBorder(
                qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            # resize twice to ensure that the border width is consistent across images
            qimg = cv2.resize(qimg, (width, height))
            num_cols = topk + 1
            grid_img = 255 * np.ones(
                (
                    height,
                    num_cols * width + topk * GRID_SPACING + QUERY_EXTRA_SPACING, 3
                ),
                dtype=np.uint8
            )
            grid_img[:, :width, :] = qimg
        else:
            qdir = osp.join(
                save_dir, osp.basename(osp.splitext(qimg_path_name)[0])
            )
            mkdir_if_missing(qdir)
            _cp_img_to(qimg_path, qdir, rank=0, prefix='query')

        rank_idx = 1
        for g_idx in indices[q_idx, :]:
            gimg_path, gpid, gcamid = gallery[g_idx][:3]
            invalid = (qpid == gpid) & (qcamid == gcamid)

            if not invalid:
                matched = gpid == qpid
                if data_type == 'image':
                    border_color = GREEN if matched else RED
                    gimg = cv2.imread(gimg_path)
                    gimg = cv2.resize(gimg, (width, height))
                    gimg = cv2.copyMakeBorder(
                        gimg,
                        BW,
                        BW,
                        BW,
                        BW,
                        cv2.BORDER_CONSTANT,
                        value=border_color
                    )
                    gimg = cv2.resize(gimg, (width, height))
                    start = rank_idx * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                    end = (
                        rank_idx + 1
                    ) * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                    ggname = osp.basename(osp.splitext(gimg_path)[0])
                    
                    font=cv2.FONT_HERSHEY_SIMPLEX#使用默认字体
                    # cv2.putText(gimg, ggname,(0,20),font,0.6,(255,255,255),2)

                    grid_img[:, start:end, :] = gimg
                else:
                    _cp_img_to(
                        gimg_path,
                        qdir,
                        rank=rank_idx,
                        prefix='gallery',
                        matched=matched
                    )

                rank_idx += 1
                if rank_idx > topk:
                    break

        if data_type == 'image':
            imname = osp.basename(osp.splitext(qimg_path_name)[0])
            cv2.imwrite(osp.join(save_dir, imname + '.jpg'), grid_img)

        if (q_idx + 1) % 100 == 0:
            print('- done {}/{}'.format(q_idx + 1, num_q))

    print('Done. Images have been saved to "{}" ...'.format(save_dir))


# tools for reid datamanager data_v2
def _parse_data_for_train(data):
    imgs = data[0]
    pids = data[1]
    return imgs, pids


def _parse_data_for_eval(data):
    imgs = data[0]
    pids = data[1]
    camids = data[2]
    return imgs, pids, camids


def extract_feature(model, device, loader, args):
    features = torch.FloatTensor()
    pids, camids = [], []

    for d in loader:
        inputs, pid, camid = _parse_data_for_eval(d)
        input_img = inputs.to(device)
        outputs = model(input_img)
        # print(outputs.shape)
        if args.feat_inference == 'after':

            f1 = outputs.data.cpu()
            # flip
            inputs = inputs.index_select(
                3, torch.arange(inputs.size(3) - 1, -1, -1))
            input_img = inputs.to(device)
            outputs = model(input_img)
            f2 = outputs.data.cpu()

        else:
            f1 = outputs[-1].data.cpu()
            # flip
            inputs = inputs.index_select(
                3, torch.arange(inputs.size(3) - 1, -1, -1))
            input_img = inputs.to(device)
            outputs = model(input_img)
            f2 = outputs[-1].data.cpu()

        ff = f1 + f2
        if ff.dim() == 3:
            fnorm = torch.norm(
                ff, p=2, dim=1, keepdim=True)  # * np.sqrt(ff.shape[2])
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


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--root', type=str)
    # parser.add_argument('-d', '--dataset', type=str, default='market1501')
    # parser.add_argument('-m', '--model', type=str, default='osnet_x1_0')
    # parser.add_argument('--weights', type=str)
    # parser.add_argument('--save-dir', type=str, default='log')
    # parser.add_argument('--height', type=int, default=256)
    # parser.add_argument('--width', type=int, default=128)
    # args = parser.parse_args()

    if args.config != '':
        with open(args.config, 'r') as f:
            config = yaml.load(f)
        for op in config:
            setattr(args, op, config[op])

    # loader = data.Data(args)
    ckpt = utility.checkpoint(args)
    loader = data_v2.ImageDataManager(args)
    model = Model(args, ckpt)
    optimzer = optim.make_optimizer(args, model)
    # loss = loss.make_loss(args, ckpt) if not args.test_only else None

    start = -1
    if args.load != '':
        start = ckpt.resume_from_checkpoint(
            osp.join(ckpt.dir, 'model-latest.pth'), model, optimzer) - 1
    if args.pre_train != '':
        ckpt.load_pretrained_weights(model, args.pre_train)

    scheduler = optim.make_scheduler(args, optimzer, start)

    # print('[INFO] System infomation: \n {}'.format(get_pretty_env_info()))
    ckpt.write_log('[INFO] Model parameters: {com[0]} flops: {com[1]}'.format(
        com=compute_model_complexity(model, (1, 3, args.height, args.width))))

    use_gpu = torch.cuda.is_available()

    # datamanager = torchreid.data.ImageDataManager(
    #     root=args.root,
    #     sources=args.dataset,
    #     height=args.height,
    #     width=args.width,
    #     batch_size_train=100,
    #     batch_size_test=100,
    #     transforms=None,
    #     train_sampler='SequentialSampler'
    # )
    # test_loader = loader.testloader
    test_loader = loader.test_loader
    query_loader = loader.query_loader
    query_dataset = loader.queryset.query
    gallery_dataset = loader.galleryset.gallery
    # model = torchreid.models.build_model(
    #     name=args.model,
    #     num_classes=datamanager.num_train_pids,
    #     use_gpu=use_gpu
    # )

    if use_gpu:
        model = model.cuda()

    device = torch.device('cuda' if use_gpu else 'cpu')

    model.eval()

    # self.ckpt.add_log(torch.zeros(1, 6))
    # qf = self.extract_feature(self.query_loader,self.args).numpy()
    # gf = self.extract_feature(self.test_loader,self.args).numpy()
    with torch.no_grad():

        qf, query_ids, query_cams = extract_feature(
            model, device, query_loader, args)
        gf, gallery_ids, gallery_cams = extract_feature(
            model, device, test_loader, args)

    if args.re_rank:
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    else:
        # dist = cdist(qf, gf,metric='cosine')

        # cosine distance
        dist =1- torch.mm(qf, gf.t()).cpu().numpy()

    # if args.weights and check_isfile(args.weights):
    #     load_pretrained_weights(model, args.weights)
    save_dir = ckpt.dir

    visualize_ranked_results(
        dist,
        query_dataset,
        gallery_dataset,
        loader.data_type,
        width=loader.width,
        height=loader.height,
        save_dir=osp.join(save_dir, 'visrank'),
        # topk=visrank_topk)
    )


if __name__ == '__main__':
    main()
