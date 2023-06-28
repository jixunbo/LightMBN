# Option Description
```
'--nThread': type=int, default=4, number of threads for data loading.

'--cpu', action='store_true', if raise, use cpu only.

'--nGPU', type=int, default=1, number of GPUs.

--config', type=str, default="", config path,if you have config file,use to set options, you don't need to input any option again.

'--datadir', type=str, is the dataset root, which contains folder Market-1501, DukeMTMC-ReID etw..

'--data_train' and '--data_test', type=str, specify the name of train/test dataset, which we can train on single or multiple datasets but test on another datasets, supported options: market1501, dukemtmc, MOT17, cuhk03_spilited(767/700 protocol) or e.g. market1501+dukemtmc.

'--batchid 6' and '--batchimage 8': type=int, indicate that each batch contrains 6 persons, each person has 8 different images, totally 48 images.

'--sampler', type=str,default='True', if 'True', sample batchid persons and batchimage in a batch, else, ramdom selected totally batchid\*batchimage in a batch.

''--batchtest', type=int, default=32, total batch size for evaluation.

'--test_only', action='store_true', if raise, only run the evaluation.

'--save', type=str, default='test', name of the folder to save output, if '', then it will create the name using current time.

'--load', type=str, default='', name of the output folder, if there is a checkpoint file in the folder, it will resume trainning.

'--pre_train', type=str, default='', path of pre-trained model file.

'--epochs', type=int, is the epochs we'd like to train, while '--test_every 10' means evaluation will be excuted in every 10 epochs, the parameters of network and optimizer are updated after every every evaluation. 

'--model', default='LMBN_n', name of model, options: LMBN_n, LMBN_r,  ResNet50, PCB, MGN, etw..

'--loss', type=str, default='0.5\*CrossEntropy+0.5\*Triplet', you can combine different loss functions and corresponding weights, you can use only one loss function or 2 and more functions, e.g. '1\*CrossEntropy', '0.5\*CrossEntropy+0.5\*MSLoss+0.0005\*CenterLoss', options: CrossEntropy, Triplet, MSLoss, CenterLoss, Focal, GroupLoss.

'--margin', type=float, margin for Triplet and MSLoss.

'--if_labelsmooth', action='store_true', if raise, label smoothing on.

'--bnneck', action='store_true', if raise, use BNNeck, only for ResNet and PCB.

'--drop_block', action='store_true', if raise, use Batch Drop Block, and '--h_ratio 0.3 and --w_ratio 1.0' indicate the erased region on the feature maps. 

'--pool', type=str, default='avg', choose pooling method, options: avg, max.

'--feats', type=int, default=256, dimension of feature maps for evaluation.

'--height', type=int, default=384, height of the input image.

''--width', type=int, default=128, width of the input image.

'--num_classes', type=int, default=751, number of classes of train dataset, but normally you don't need to set it, it'll be automatically setted depend on the dataset.

'--lr', type=float, default=6e-4, initial learning rate.

'--gamma', type=float, default=0.1,learning rate decay factor for step decay.

'--warmup', type=str, default='constant', learning rate warmup method, options: linear, constant.

'--w_cosine_annealing', action='store_true', if raise, use warm up cosine annealing learning rate scheduler.

'--pcb_different_lr', type=str, default='True', if 'True', use different lr only for PCB, if lr is 5e-3, then lr for classifier is 5e-3, lr for other part is 5e-4.

'--optimizer, default='ADAM', options: 'SGD','ADAM','NADAM','RMSprop'.

'--momentum', type=float, default=0.9, SGD momentum.

'--nesterov', action='store_true', if raise, SGD nesterov.

'--parts', type=int, default=6, is used to set the number of stripes to be devided, original paper set 6.

'--re_rank', action='store_true', 'if raise, use re-ranking.

'--cutout', action='store_true', if raise, use cutout augmentation.

'--random_erasing', action='store_true', if raise, use random erasing augmentation.

'--probability', type=float, default=0.5, probability of random erasing.

'--T', type=int, default=3, number of iterations of computing group loss.

'--num_anchors', type=int, default=1, number of iterations of computing group loss.
```