# from configs.common import *
from configs.backbones import *
from configs.necks import *
from configs.heads import *
from configs.common import BaseModule,Sequential

import torch.nn as nn
import torch

import functools
from inspect import getfullargspec
from collections import abc
import numpy as np

def build_model(cfg):
    if isinstance(cfg, list):
        modules = [
            eval(cfg_.pop("type"))(**cfg_) for cfg_ in cfg
        ]
        return Sequential(*modules)
    else:
        return eval(cfg.pop("type"))(**cfg)
    

class BuildNet(BaseModule):
    def __init__(self,cfg):
        super(BuildNet, self).__init__()
        self.neck_cfg = cfg.get("neck")
        self.head_cfg = cfg.get("head")
        self.backbone = build_model(cfg.get("backbone"))
        if self.neck_cfg is not None:
            self.neck = build_model(cfg.get("neck"))
        
        if self.head_cfg is not None:
            self.head = build_model(cfg.get("head"))

    def freeze_layers(self,names):
        assert isinstance(names,tuple)
        for name in names:
            layers = getattr(self, name)
            # layers.eval()
            for param in layers.parameters():
                param.requires_grad = False
    
    def extract_feat(self, img, stage='neck'):
       # noqa: E501
        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')

        x = self.backbone(img)

        if stage == 'backbone':
            return x

        if hasattr(self, 'neck') and self.neck is not None:
            x = self.neck(x)
        if stage == 'neck':
            return x
    
    def forward(self, x, return_loss=True, train_statu=False, **kwargs):
        x = self.extract_feat(x)
        
        if not train_statu:
            if return_loss:
                return self.forward_train(x, **kwargs)
            else:
                return self.forward_test(x, **kwargs)
        else:
            return self.forward_test(x), self.forward_train(x, **kwargs)
        
    def forward_train(self, x, targets, **kwargs):
         
        losses = dict()
        loss = self.head.forward_train(x, targets, **kwargs)
        losses.update(loss)
        return losses
        
    def forward_test(self, x, **kwargs):
        
        out = self.head.simple_test(x,**kwargs)
        return out