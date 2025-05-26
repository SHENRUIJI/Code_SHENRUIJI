import os
import sys
import copy
import argparse
import shutil
import time
import numpy as np
import random
import csv
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

from utils.history import History
from utils.dataloader import Mydataset, collate
from utils.train_utils import train, validation, print_info, init_random_seed, set_random_seed, resume_model
from utils.inference import init_model
from core.optimizers import *
from build import BuildNet

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--device', help='device used for training. (Deprecated)')
    parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use')
    parser.add_argument('--split-validation', action='store_true', help='whether to split validation set from training set.')
    parser.add_argument('--ratio', type=float, default=0.2, help='the proportion of the validation set to the training set.')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

# 配置部分
model_cfg = dict(
    backbone=dict(type='SEResNet', depth=34, num_stages=4, out_indices=(3,), style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=11,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, label_smoothing=0.1),
        topk=(1, 5),
        dropout_ratio=0.5
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CenterCrop', crop_size=224, efficientnet_style=True, interpolation='bicubic', backend='pillow'),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data_cfg = dict(
    batch_size=16,
    num_workers=4,
    train=dict(
        pretrained_flag=False,
        pretrained_weights='',
        freeze_flag=False,
        freeze_layers=(),
        epoches=100,
    ),
    test=dict(
        ckpt='',
        metrics=['accuracy', 'precision', 'recall', 'f1_score', 'confusion'],
        metric_options=dict(topk=(1, 5), thrs=None, average_mode='none')
    )
)

optimizer_cfg = dict(type='AdamW', lr=0.0005, weight_decay=0.05)
lr_config = dict(type='CosineAnnealingLrUpdater', min_lr=1e-6, warmup='linear', warmup_iters=1000, warmup_ratio=1e-6)

#-主函数-
def main():
    args = parse_args()
    print_info(model_cfg)

    meta = dict()
    dirname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_dir = os.path.join('logs', model_cfg.get('backbone').get('type'), dirname)
    meta['save_dir'] = save_dir

    seed = init_random_seed(args.seed)
    set_random_seed(seed, deterministic=args.deterministic)
    meta['seed'] = seed

    total_annotations = "datas/train.txt"
    with open(total_annotations, encoding='utf-8') as f:
        total_datas = f.readlines()
    if args.split_validation:
        total_nums = len(total_datas)
        if isinstance(seed, int):
            rng = np.random.default_rng(seed)
            rng.shuffle(total_datas)
        val_nums = int(total_nums * args.ratio)
        folds = list(range(int(1.0 / args.ratio)))
        fold = random.choice(folds)
        val_start = val_nums * fold
        val_end = val_nums * (fold + 1)
        train_datas = total_datas[:val_start] + total_datas[val_end:]
        val_datas = total_datas[val_start:val_end]
    else:
        train_datas = total_datas.copy()
        with open('datas/test.txt', encoding='utf-8') as f:
            val_datas = f.readlines()

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Initialize the weights.')
    model = BuildNet(model_cfg)
    if not data_cfg['train']['pretrained_flag']:
        model.init_weights()
    if data_cfg['train']['freeze_flag'] and data_cfg['train']['freeze_layers']:
        freeze_layers = ' '.join(list(data_cfg['train']['freeze_layers']))
        print('Freeze layers : ' + freeze_layers)
        model.freeze_layers(data_cfg['train']['freeze_layers'])

    if device != torch.device('cpu'):
        model = DataParallel(model, device_ids=[args.gpu_id])

    optimizer = eval('optim.' + optimizer_cfg.pop('type'))(params=model.parameters(), **optimizer_cfg)
    lr_update_func = eval(lr_config.pop('type'))(**lr_config)

    train_dataset = Mydataset(train_datas, train_pipeline)
    val_dataset = Mydataset(val_datas, copy.deepcopy(train_pipeline))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=data_cfg['batch_size'],
                              num_workers=data_cfg['num_workers'], pin_memory=True, drop_last=True,
                              collate_fn=collate)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=data_cfg['batch_size'],
                            num_workers=data_cfg['num_workers'], pin_memory=True, drop_last=True,
                            collate_fn=collate)

    runner = dict(
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        iter=0,
        epoch=0,
        max_epochs=data_cfg['train']['epoches'],
        max_iters=data_cfg['train']['epoches'] * len(train_loader),
        best_train_loss=float('INF'),
        best_val_acc=float(0),
        best_train_weight='',
        best_val_weight='',
        last_weight=''
    )
    meta['train_info'] = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[])

    if args.resume_from:
        model, runner, meta = resume_model(model, runner, args.resume_from, meta)
    else:
        os.makedirs(save_dir)
        model = init_model(model, data_cfg, device=device, mode='train')

        from ptflops import get_model_complexity_info

        def clear_ptflops_attrs(model):
            for module in model.modules():
                if hasattr(module, '__flops__'):
                    delattr(module, '__flops__')
                if hasattr(module, '__params__'):
                    delattr(module, '__params__')

        model_for_analysis = copy.deepcopy(model.module if isinstance(model, torch.nn.DataParallel) else model)

        def forward_dummy(self, x):
            return self.backbone(x)

        model_for_analysis.forward = forward_dummy.__get__(model_for_analysis, model_for_analysis.__class__)
        model_for_analysis = model_for_analysis.to('cpu')

        with torch.no_grad():
            macs, params = get_model_complexity_info(
                model_for_analysis, (3, 224, 224), as_strings=True,
                print_per_layer_stat=False, verbose=False
            )

        tqdm.write(f"Model FLOPs (MACs): {macs}")
        tqdm.write(f"Model Params: {params}")

        model = model.to(device)

    train_history = History(meta['save_dir'])
    time_csv_path = os.path.join(meta['save_dir'], 'epoch_times.csv')
    with open(time_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Time(seconds)'])

    lr_update_func.before_run(runner)

    for epoch in range(runner['epoch'], runner['max_epochs']):
        start_time = time.time()

        lr_update_func.before_train_epoch(runner)
        train(model, runner, lr_update_func, device, epoch, data_cfg['train']['epoches'], data_cfg['test'], meta)
        validation(model, runner, data_cfg['test'], device, epoch, data_cfg['train']['epoches'], meta)
        train_history.after_epoch(meta)

        end_time = time.time()
        with open(time_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, round(end_time - start_time, 4)])

if __name__ == "__main__":
    main()
