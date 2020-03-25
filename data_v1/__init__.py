from importlib import import_module
from torchvision import transforms
from utils.random_erasing import RandomErasing, Cutout
from .sampler import RandomSampler, RandomIdentitySampler
from torch.utils.data import dataloader


class Data:
    def __init__(self, args):

        # train_list = [
        #     transforms.Resize((args.height, args.width), interpolation=3),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        #                          0.229, 0.224, 0.225])
        # ]

        train_list = [
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.Pad(10),
            transforms.RandomCrop((args.height, args.width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        if args.random_erasing:
            train_list.append(RandomErasing(
                probability=args.probability, mean=[0.485, 0.456, 0.406]))
            print('Using random_erasing augmentation.')
        if args.cutout:
            train_list.append(Cutout(mean=[0.485, 0.456, 0.406]))
            print('Using cutout augmentation.')

        train_transform = transforms.Compose(train_list)

        test_transform = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        if not args.test_only and args.model == 'MGN':
            module_train = import_module('data.' + args.data_train.lower())
            self.trainset = getattr(module_train, args.data_train)(
                args, train_transform, 'train')
            self.train_loader = dataloader.DataLoader(self.trainset,
                                                      sampler=RandomIdentitySampler(
                                                          self.trainset, args.batchid * args.batchimage, args.batchimage),
                                                      # shuffle=True,
                                                      batch_size=args.batchid * args.batchimage,
                                                      num_workers=args.nThread)
        # elif not args.test_only and args.model in ['ResNet50','PCB'] and args.loss.split('*')[1]=='CrossEntropy':
        #     module_train = import_module('data.' + args.data_train.lower())
        #     self.trainset = getattr(module_train, args.data_train)(
        #         args, train_transform, 'train')
        #     self.train_loader = dataloader.DataLoader(self.trainset,
        #                                               shuffle=True,
        #                                               batch_size=args.batchid * args.batchimage,
        #                                               num_workers=args.nThread)
        elif not args.test_only and args.model in ['ResNet50', 'PCB', 'PCB_v', 'PCB_conv', 'BB_2_db','BB', 'MGDB','MGDB_v2','MGDB_v3','BB_2_v3','BB_2', 'PCB_conv_modi_2', 'BB_2_conv','BB_2_cat', 'BB_4_cat','PCB_conv_modi', 'Pyramid','PLR'] and bool(args.sampler):

            module_train = import_module('data.' + args.data_train.lower())
            self.trainset = getattr(module_train, args.data_train)(
                args, train_transform, 'train')
            # self.train_loader = dataloader.DataLoader(self.trainset,
            #                                           sampler=RandomSampler(
            #                                               self.trainset, args.batchid, batch_image=args.batchimage),
            #                                           # shuffle=True,
            #                                           batch_size=args.batchid * args.batchimage,
            #                                           num_workers=args.nThread,
            #                                           drop_last=True)
            self.train_loader = dataloader.DataLoader(self.trainset,
                                                      sampler=RandomIdentitySampler(
                                                          self.trainset, args.batchid * args.batchimage, args.batchimage),
                                                      # shuffle=True,
                                                      batch_size=args.batchid * args.batchimage,
                                                      num_workers=args.nThread)

        elif not args.test_only and args.model not in ['MGN', 'ResNet50', 'PCB','BB_2_db', 'PCB_v', 'PCB_conv','MGDB', 'PCB_conv_modi_2', 'PCB_conv_modi', 'BB', 'BB_2','BB_2_cat','BB_4_cat','PLR']:
            raise Exception(
                'DataLoader for {} not designed'.format(args.model))
        else:
            self.train_loader = None

        if args.data_test in ['Market1501', 'DukeMTMC', 'GTA']:
            module = import_module('data.' + args.data_train.lower())
            self.galleryset = getattr(module, args.data_test)(
                args, test_transform, 'test')
            self.queryset = getattr(module, args.data_test)(
                args, test_transform, 'query')

        else:
            raise Exception()
        # print(len(self.trainset))

        self.test_loader = dataloader.DataLoader(
            self.galleryset, batch_size=args.batchtest, num_workers=args.nThread)
        self.query_loader = dataloader.DataLoader(
            self.queryset, batch_size=args.batchtest, num_workers=args.nThread)
