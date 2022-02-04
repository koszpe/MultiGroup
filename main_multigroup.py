#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import simsiam.loader
import simsiam.builder
from evaluator import SnnEvaluator
from mp_utils import AllReduce, AllGather
from simsiam.logger import TBLogger

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("--runname", default="dev", help="Name of run on tensorboard")
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset', default='/shared_data/imagenet')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--no-multiprocessing-distributed', action='store_false',
                    dest="multiprocessing_distributed",
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--no-fix-pred-lr', action='store_false', dest="fix_pred_lr",
                    help='Fix learning rate for the predictor')

parser.add_argument('--logdir', default="/storage/simsiam/logs", type=str,
                    help='Where to log')
parser.add_argument("--save-frequency", default=5, help="Frequency of checkpoint saving in epochs")

parser.add_argument("--group-sizes", default=[2, 16, 32, 64], type=int, nargs="+", help="Size of the groups to create")
parser.add_argument("--group-nums", default= [16, 8,  4,  4], type=int, nargs="+", help="Num of the grous")
parser.add_argument('--no-cossim', action='store_false', dest="use_cossim",
                    help='Fix learning rate for the predictor')
parser.add_argument('--no-corr', action='store_false', dest="use_corr",
                    help='Fix learning rate for the predictor')
parser.add_argument('--no-memax', action='store_false', dest="use_memax",
                    help='Fix learning rate for the predictor')
parser.add_argument('--no-entropy', action='store_false', dest="use_ent",
                    help='Fix learning rate for the predictor')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = simsiam.builder.MultiGroup(
        args.group_sizes,
        args.group_nums,
        models.__dict__[args.arch],
        args.dim, args.pred_dim)

    evaluator = SnnEvaluator(dim=args.dim, n_classes=1000, max_queue_size=10)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            evaluator.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            evaluator.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        evaluator = evaluator.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model) # print model after SyncBatchNorm

    # define loss function (criterion) and optimizer
    criterion = create_loss(
        use_cossim=args.use_cossim,
        use_corr=args.use_corr,
        use_memax=args.use_memax,
        use_ent=args.use_ent,
    )

    if args.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
        # optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = datasets.ImageFolder(
        traindir,
        simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    global_step = len(train_loader) * args.start_epoch
    folder = os.path.join(args.logdir, args.runname)
    os.makedirs(folder, exist_ok=True)
    tb_logger = TBLogger(log_dir=folder,
                         global_step=global_step,
                         batch_size=args.batch_size,
                         world_size=args.world_size if args.distributed else 1)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        # with torch.autograd.set_detect_anomaly(True):
        train(train_loader, model, criterion, optimizer, epoch, tb_logger, evaluator, args)

        if (epoch + 1) % args.save_frequency == 0 or epoch == 0:
            filename = os.path.join(folder, 'checkpoint_{:04d}.pt'.format(epoch+1))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename=filename)


def sharpen(p, T=0.25):
    sharp_p = p**(1./T)
    sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
    return sharp_p

sm = torch.nn.Softmax(dim=1)
def softmax(x):
    return sm(x - x.max(dim=1, keepdim=True)[0])

def crossentropy(ps, qs):
    loss_count = 0
    sum_loss = 0
    for ss_ps, ss_qs in zip(ps, qs):
        for p, q in zip(ss_ps, ss_qs):
            p = softmax(p)
            # p[p<1-4] = 0
            # p = sharpen(p)
            q = softmax(q)
            sum_loss -= torch.sum(p * torch.log(q+1e-6)) / len(p)
            loss_count += 1
    return sum_loss / loss_count

def entropy(ps):
    loss_count = 0
    sum_loss = 0
    for ss_ps in ps:
        for p in ss_ps:
            p = softmax(p)
            sum_loss -= torch.sum(p * torch.log(p+1e-6)) / len(p)
            loss_count += 1
    return sum_loss / loss_count

def me_max(qs):
    loss_count = 0
    sum_loss = 0
    for ss_qs in qs:
        for q in ss_qs:
            avg_probs = torch.mean(softmax(q), dim=0)
            sum_loss -= torch.sum(avg_probs * torch.log(avg_probs + 1e-6))
            loss_count += 1
    return sum_loss / loss_count

def create_loss(use_cossim, use_ent, use_corr, use_memax):
    cossim = nn.CosineSimilarity(dim=1)
    def compute_all_loss(p1, p2, z1, z2, gs):
        losses = dict()
        losses["cossim"] = (cossim(p1, z2).mean() + cossim(p2, z1).mean()) * 0.5
        losses["memax"] = me_max(gs)
        losses["corr"] = correlation(gs, apply_softmax=False, corr=True)
        losses["ent"] = entropy(gs)
        if not use_cossim:
            losses["cossim"] = losses["cossim"].detach()
        if not use_ent:
            losses["ent"] = losses["ent"].detach()
        if not use_corr:
            losses["corr"] = losses["corr"].detach()
        if not use_memax:
            losses["memax"] = losses["memax"].detach()
        return losses
    return compute_all_loss

softm = torch.nn.Softmax(dim=-1)
def correlation(qs, apply_softmax=True, corr=True):
    sum_corr = 0
    for q in qs:
        q = q.clone().double()
        if apply_softmax:
            q = softm(q)
        q = q - q.mean(dim=1, keepdim=True)
        if corr:
            q = q / q.std(dim=1, keepdim=True)
        tril_indices = torch.tril_indices(q.shape[0], q.shape[1], offset=-1)
        corr_m = torch.einsum("nbi,mbj->nmij", q, q) / (q.shape[1] - 1)
        sum_corr += corr_m[tril_indices[0], tril_indices[1], :, :].abs().mean()
    sum_corr /= len(qs)

    return sum_corr

def train(train_loader, model, criterion, optimizer, epoch, tb_logger, evaluator, args):
    log_per_step = 1000
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, ys) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            ys = ys.cuda(args.gpu, non_blocking=True)

        # compute output and loss
        p1, p2, z1, z2, gs = model(x1=images[0], x2=images[1])
        loss_dict = criterion(p1, p2, z1, z2, gs)
        loss = - loss_dict["cossim"] + loss_dict["corr"] - loss_dict["memax"] - loss_dict["ent"]
        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluate
        ys = ys.repeat(2)
        evaluate = tb_logger.need_log(log_per_step)
        metrics = evaluator(torch.cat([z1, z2]), ys, evaluate=evaluate)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        if tb_logger.need_log(log_per_step):
            tb_logger.add_scalar(tag="train/loss", scalar_value=loss.item())
            tb_logger.add_scalar(tag="train/sim", scalar_value=loss_dict["cossim"].item())
            tb_logger.add_scalar(tag="train/memax", scalar_value=loss_dict["memax"].item())
            tb_logger.add_scalar(tag="train/corr", scalar_value=loss_dict["corr"].item())
            tb_logger.add_scalar(tag="train/ent", scalar_value=loss_dict["ent"].item())
            for key, value in metrics.items():
                tb_logger.add_scalar(tag=f"train/{key}", scalar_value=value.item())
        tb_logger.step()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()
