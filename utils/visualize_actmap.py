"""Visualizes CNN activation maps to see where the CNN focuses on to extract features.
Reference:
    - Zagoruyko and Komodakis. Paying more attention to attention: Improving the
      performance of convolutional neural networks via attention transfer. ICLR, 2017
    - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
"""

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


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 10


@torch.no_grad()
def visactmap(
    model,
    test_loader,
    save_dir,
    width,
    height,
    use_gpu,
    img_mean=None,
    img_std=None
):
    if img_mean is None or img_std is None:
        # use imagenet mean and std
        img_mean = IMAGENET_MEAN
        img_std = IMAGENET_STD

    model.eval()

    for target in list(test_loader.keys()):
        data_loader = test_loader[target]['query']  # only process query images
        # original images and activation maps are saved individually
        actmap_dir = osp.join(save_dir, 'actmap_' + target)

        if not osp.exists(actmap_dir):
            os.makedirs(actmap_dir)

        print('Visualizing activation maps for {} ...'.format(target))

        for batch_idx, data in enumerate(data_loader):

            # if batch_idx == 10:
            #     break

            imgs, paths = data[0], data[-1]
            if use_gpu:
                imgs = imgs.cuda()

            # forward to get convolutional feature maps
            try:
                outputs_list = model(imgs)
            except TypeError:
                raise TypeError(
                    'forward() got unexpected keyword argument "return_featuremaps". '
                    'Please add return_featuremaps as an input argument to forward(). When '
                    'return_featuremaps=True, return feature maps only.'
                )
            # print(len(outputs_list))
            # # if batch_idx == 2:
            # print(outputs_list[0].size())

            #     break

            for j, output in enumerate(zip(*outputs_list)):
                # if outputs.dim() != 4:
                #     raise ValueError(
                #         'The model output is supposed to have '
                #         'shape of (b, c, h, w), i.e. 4 dimensions, but got {} dimensions. '
                #         'Please make sure you set the model output at eval mode '
                #         'to be the last convolutional feature maps'.format(
                #             outputs.dim()
                #         )
                #     )
                # print(j)

                n_fmaps = len(output)
                # print(n_fmaps)/
                grid_img = 255 * np.ones(
                    (height, (n_fmaps + 1) * width + n_fmaps * GRID_SPACING, 3), dtype=np.uint8
                )
                # RGB image
                if use_gpu:
                    imgs = imgs.cpu()
                if j>= imgs.size()[0]:
                    break
                img = imgs[j, ...]
                for t, m, s in zip(img, img_mean, img_std):
                    t.mul_(s).add_(m).clamp_(0, 1)
                img_np = np.uint8(np.floor(img.numpy() * 255))
                # (c, h, w) -> (h, w, c)
                img_np = img_np.transpose((1, 2, 0))


                path = paths[j]
                imname = osp.basename(osp.splitext(path)[0])
                # print(imname[:4])
                # print()
                # if imname[:4] not in ['4117','4699','0580','0601']:
                # if imname[:4] not in ['0101']:

                #     continue

                # compute activation maps]
                # print(output)
                for output_number, outputs in enumerate(output):

                    # outputs = (outputs**2).sum(1)
                    # b, h, w = outputs.size()
                    # outputs = outputs.view(b, h * w)
                    # outputs = F.normalize(outputs, p=2, dim=1)
                    # outputs = outputs.view(b, h, w)
                    # print(output)
                    # print(outputs.size())
                    outputs = (outputs**2).sum(0)
                    h, w = outputs.size()
                    outputs = outputs.view(h * w)
                    outputs = F.normalize(outputs, p=2, dim=0)
                    outputs = outputs.view(h, w)

                    if outputs.size()[0]!=24:
                        z=torch.zeros(12,8).cuda()
                        if output_number ==4:
                            outputs=torch.cat((outputs,z),0) 
                        if output_number == 5 :
                            outputs=torch.cat((z,outputs),0) 
                    # if outputs.size()[0]!=24:
                    #     z=torch.zeros(8,8)
                    #     if output_number ==1:
                    #         outputs=torch.cat((outputs,z,z),0) 
                    #     if output_number == 2 :
                    #         outputs=torch.cat((z,outputs,z),0) 
                    #     if output_number == 3 :
                    #         outputs=torch.cat((outputs,z,z),0) 

                    if use_gpu:
                        outputs = outputs.cpu()

                    # for j in range(outputs.size(0)):
                        # get image name
                        # print(j)
                    path = paths[j]
                    imname = osp.basename(osp.splitext(path)[0])



                    # activation map
                    # am = outputs[j, ...].numpy()
                    am = outputs.numpy()
                    # print(output_number)
                    # print(am)
                    try:
                        # if output_number>=1:
                        #     am = cv2.resize(am,(width,height/2))
                        # else:
                        #     am = cv2.resize(am, (width, height))
                        am = cv2.resize(am,(width,height))
                    except:
                        print(path)
                        break

                    am = 255 * (am - np.min(am)) / (
                        np.max(am) - np.min(am) + 1e-12
                    )
                    am = np.uint8(np.floor(am))
                    am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

                    # overlapped
                    overlapped = img_np * 0.45 + am * 0.55
                    overlapped[overlapped > 255] = 255
                    overlapped = overlapped.astype(np.uint8)

                    # save images in a single figure (add white spacing between images)
                    # from left to right: original image, activation map, overlapped image

                    grid_img[:, :width, :] = img_np[:, :, ::-1]
                    grid_img[:,
                             (output_number + 1) * width + (output_number + 1) * GRID_SPACING:(output_number + 2) * width + (output_number + 1) * GRID_SPACING, :] = overlapped
                    # grid_img[:, 2 * width + 2 * GRID_SPACING:, :] = overlapped
                cv2.imwrite(osp.join(actmap_dir, imname  + '.jpg'), grid_img)

            if (batch_idx + 1) % 10 == 0:
                print(
                    '- done batch {}/{}'.format(
                        batch_idx + 1, len(data_loader)
                    )
                )


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
    test_loader = loader.testloader

    # model = torchreid.models.build_model(
    #     name=args.model,
    #     num_classes=datamanager.num_train_pids,
    #     use_gpu=use_gpu
    # )

    if use_gpu:
        model = model.cuda()

    # if args.weights and check_isfile(args.weights):
    #     load_pretrained_weights(model, args.weights)

    save_dir = ckpt.dir

    visactmap(
        model, test_loader, save_dir, args.width, args.height, use_gpu
    )


if __name__ == '__main__':
    main()
