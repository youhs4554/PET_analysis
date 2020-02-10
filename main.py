import argparse
from collections import Counter
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import os
from losses import FocalLoss
from datasets import init_dataset
from train import train_epoch, evaluate
from models import init_net

import multiprocessing
multiprocessing.set_start_method('spawn', True)


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode',
        default='max',
        type=str,
        help='Slice MODE ( min | max )',
    )
    parser.add_argument(
        '--model_idx',
        default=None,
        type=str,
        help='Model idx to train ( 1 | 2 | 4 | 5 )',
    )
    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help='BatchSize',
    )
    parser.add_argument(
        '--single_channel',
        action='store_true',
        help='If true, use single-channel image(gray) as input')
    parser.set_defaults(single_channel=False)
    parser.add_argument(
        '--flatten',
        action='store_true',
        help='If true, use flatten 3d-image as input')
    parser.set_defaults(flatten=False)
    parser.add_argument(
        '--rnn',
        action='store_true',
        help='If true, use rnn')
    parser.set_defaults(rnn=False)

    args = parser.parse_args()

    return args


def run():
    args = parse_opts()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "9"

    # GLOBAL VARS #
    MODE = args.mode
    CLASS_WEIGHT = False
    N_EP = 20
    FLATTEN = args.flatten
    RNN = args.rnn
    BATCH_SIZE = args.batch_size
    ####

    datasets, dataloaders = init_dataset(
        BATCH_SIZE, single_channel=args.single_channel)

    print('[Train] class counts', np.unique(
        datasets['train'].target_vals, return_counts=True))
    print('[Test] class counts', np.unique(
        datasets['test'].target_vals, return_counts=True))

    n_ch = 1 if args.single_channel else 3

    if MODE == 'min':
        in_channels = datasets['train'].min_depth*n_ch
    elif MODE == 'max':
        in_channels = datasets['train'].max_depth*n_ch

    torch.manual_seed(0)

    # init net
    net = init_net(opt=args.model_idx, in_channels=in_channels)

    class_weight = None
    if CLASS_WEIGHT:
        cnts = Counter(datasets['train'].target_vals)
        n = len(datasets['train'])
        class_weight = [max(cnts.values())/cnts['0'],
                        max(cnts.values())/cnts['1']]
        class_weight = torch.FloatTensor(class_weight)

    cross_entrp_loss = nn.CrossEntropyLoss(weight=class_weight).cuda()
    focal_loss = FocalLoss().cuda()

    optimizer = optim.Adam(net.parameters(), lr=0.000027)

    criterion = cross_entrp_loss

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'min', verbose=True, patience=7)

    for ep in range(N_EP):
        train_epoch(net, dataloaders['train'], optimizer,
                    criterion, ep, scheduler=None, flatten=FLATTEN, MODE=MODE, rnn=RNN)
        valid_loss = evaluate(net, dataloaders['test'], criterion,
                              ep, flatten=FLATTEN, MODE=MODE, rnn=RNN)
        # scheduler.step(valid_loss)


if __name__ == '__main__':
    run()
