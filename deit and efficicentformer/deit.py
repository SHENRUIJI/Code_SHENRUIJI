import os
import sys
sys.path.insert(0,os.getcwd())
import copy
import argparse
import shutil
import time
import numpy as np
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

from utils.history import History
from utils.dataloader import Mydataset, collate
from utils.train_utils import train, validation, print_info, file2dict, init_random_seed, set_random_seed, resume_model
from utils.inference import init_model
from core.optimizers import *
from build import BuildNet

import torch, gc

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

#模型配置
model_cfg = dict(
    backbone=dict(
        type='VisionTransformer',
        arch='deit-tiny',
        img_size=224,
        patch_size=16,
        use_distillation=True,
        drop_path_rate=0.1,
        init_cfg=[
            dict(type='Kaiming', layer='Conv2d', mode='fan_in', nonlinearity='linear')
        ]
    ),
    neck=None,
    head=dict(
        type='DeiTClsHead',
        num_classes=11,
        in_channels=192,
        hidden_dim=3072,
        loss=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0
        ),
        distill_loss=dict(
            type='CrossEntropyLoss',
            loss_weight=0.5
        )
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow', interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow', interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
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
        freeze_layers=('backbone',),
        epoches=100,
    ),
    test=dict(
        ckpt='',
        metrics=['accuracy', 'precision', 'recall', 'f1_score', 'confusion'],
        metric_options=dict(
            topk=(1, 5),
            thrs=None,
            average_mode='none'
        )
    )
)

optimizer_cfg = dict(
    type='AdamW',
    lr=0.0005,
    weight_decay=0.05,
    betas=(0.9, 0.999)
)

lr_config = dict(
    type='CosineAnnealingLrUpdater',
    min_lr_ratio=1e-2,
    by_epoch=False,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=3,
    warmup_by_epoch=True
)

# 主函数main

def before_each_epoch():
    gc.collect()
    torch.cuda.empty_cache()
    print("显存清空完成")


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--device', help='device used for training. (Deprecated)')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--split-validation', action='store_true')
    parser.add_argument('--ratio', type=float, default=0.2)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


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

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Initialize the weights.')
    model = BuildNet(model_cfg)
    if not data_cfg.get('train').get('pretrained_flag'):
        model.init_weights()
    if data_cfg.get('train').get('freeze_flag') and data_cfg.get('train').get('freeze_layers'):
        model.freeze_layers(data_cfg.get('train').get('freeze_layers'))

    if device != torch.device('cpu'):
        model = DataParallel(model, device_ids=[args.gpu_id])

    from fvcore.nn import FlopCountAnalysis, parameter_count
    backbone = model.module.backbone if isinstance(model, torch.nn.DataParallel) else model.backbone
    backbone.eval()
    backbone = backbone.to(device)

    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    try:
        flops = FlopCountAnalysis(backbone, dummy_input)
        params = parameter_count(backbone)
        print(f"[Backbone] 参数量: {params[''] / 1e6:.2f} M")
        print(f"[Backbone] 总计算量: {flops.total() / 1e9:.2f} GFLOPs")
    except Exception as e:
        print(f"无法统计 FLOPs/参数量：{e}")

    optimizer = eval('optim.' + optimizer_cfg.pop('type'))(params=model.parameters(), **optimizer_cfg)
    lr_update_func = eval(lr_config.pop('type'))(**lr_config)

    train_dataset = Mydataset(train_datas, train_pipeline)
    val_dataset = Mydataset(val_datas, copy.deepcopy(train_pipeline))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=data_cfg.get('batch_size'), num_workers=data_cfg.get('num_workers'), pin_memory=True, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=data_cfg.get('batch_size'), num_workers=data_cfg.get('num_workers'), pin_memory=True, drop_last=True, collate_fn=collate)

    runner = dict(
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        iter=0,
        epoch=0,
        max_epochs=data_cfg.get('train').get('epoches'),
        max_iters=data_cfg.get('train').get('epoches') * len(train_loader),
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

    epoch_times = []
    train_history = History(meta['save_dir'])
    lr_update_func.before_run(runner)

    for epoch in range(runner.get('epoch'), runner.get('max_epochs')):
        before_each_epoch()
        epoch_start = time.time()

        lr_update_func.before_train_epoch(runner)
        train(model, runner, lr_update_func, device, epoch, data_cfg.get('train').get('epoches'), data_cfg.get('test'), meta)
        validation(model, runner, data_cfg.get('test'), device, epoch, data_cfg.get('train').get('epoches'), meta)

        epoch_duration = time.time() - epoch_start
        epoch_times.append({'epoch': epoch + 1, 'time_sec': round(epoch_duration, 2)})
        print(f"Epoch {epoch + 1} 耗时: {epoch_duration:.2f} 秒")

        train_history.after_epoch(meta)

    import csv
    csv_path = os.path.join(meta['save_dir'], 'epoch_times.csv')
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'time_sec'])
        writer.writeheader()
        writer.writerows(epoch_times)

    print(f"每轮训练时间已保存至: {csv_path}")


if __name__ == "__main__":
    main()
