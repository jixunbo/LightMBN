import data_v1
import data_v2
from loss import make_loss
from model import make_model
from optim import make_optimizer, make_scheduler
# import engine_v1
# import engine_v2
import engine_v3
import os.path as osp
from option import args
import utils.utility as utility
from utils.model_complexity import compute_model_complexity
from torch.utils.collect_env import get_pretty_env_info
import yaml
import torch


if args.config != '':
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    for op in config:
        setattr(args, op, config[op])
torch.backends.cudnn.benchmark = True

# loader = data.Data(args)
ckpt = utility.checkpoint(args)
loader = data_v2.ImageDataManager(args)
model = make_model(args, ckpt)
optimzer = make_optimizer(args, model)
loss = make_loss(args, ckpt) if not args.test_only else None

start = -1
if args.load != '':
    start, model, optimizer = ckpt.resume_from_checkpoint(
        osp.join(ckpt.dir, 'model-latest.pth'), model, optimzer)
    start = start - 1
if args.pre_train != '':
    ckpt.load_pretrained_weights(model, args.pre_train)

scheduler = make_scheduler(args, optimzer, start)

# print('[INFO] System infomation: \n {}'.format(get_pretty_env_info()))
ckpt.write_log('[INFO] Model parameters: {com[0]} flops: {com[1]}'.format(com=compute_model_complexity(model, (1, 3, args.height, args.width))
                                                                          ))

engine = engine_v3.Engine(args, model, optimzer,
                          scheduler, loss, loader, ckpt)
# engine = engine.Engine(args, model, loss, loader, ckpt)

n = start + 1
while not engine.terminate():

    n += 1
    engine.train()
    if args.test_every != 0 and n % args.test_every == 0:
        engine.test()
    elif n == args.epochs:
        engine.test()
