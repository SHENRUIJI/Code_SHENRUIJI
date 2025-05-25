# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from .vision_transformer_head import VisionTransformerClsHead


class DeiTClsHead(VisionTransformerClsHead):

    def __init__(self, distill_loss=None, *args, **kwargs):
        # ✅ 提前弹出 distill_loss，防止传到父类报错
        self.distill_loss_cfg = distill_loss
        if 'distill_loss' in kwargs:
            kwargs.pop('distill_loss')

        super(DeiTClsHead, self).__init__(*args, **kwargs)

        # ✅ 构造用于蒸馏分支的头
        if self.hidden_dim is None:
            head_dist = nn.Linear(self.in_channels, self.num_classes)
        else:
            head_dist = nn.Linear(self.hidden_dim, self.num_classes)
        self.layers.add_module('head_dist', head_dist)

        # ✅ 使用主 loss 作为 distill loss
        self.distill_loss = self.loss
        self.distill_loss_weight = self.distill_loss_cfg.get("loss_weight", 0.5) if self.distill_loss_cfg else 0.0

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        _, cls_token, dist_token = x

        if self.hidden_dim is None:
            return cls_token, dist_token
        else:
            cls_token = self.layers.act(self.layers.pre_logits(cls_token))
            dist_token = self.layers.act(self.layers.pre_logits(dist_token))
            return cls_token, dist_token

    def simple_test(self, x, softmax=True, post_process=False):
        cls_token, dist_token = self.pre_logits(x)
        cls_score = (self.layers.head(cls_token) +
                     self.layers.head_dist(dist_token)) / 2

        if softmax:
            pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x, gt_label):
        cls_token, dist_token = self.pre_logits(x)
        cls_score = self.layers.head(cls_token)
        dist_score = self.layers.head_dist(dist_token)

        cls_loss_dict = self.loss(cls_score, gt_label)
        distill_loss_dict = self.distill_loss(dist_score, gt_label)

        cls_loss_val = cls_loss_dict.get('loss', 0.0)
        distill_loss_val = distill_loss_dict.get('loss', 0.0)

        total_loss = cls_loss_val + self.distill_loss_weight * distill_loss_val

        return {
            'loss_cls': cls_loss_val,
            'loss_dist': self.distill_loss_weight * distill_loss_val,
            'loss': total_loss
        }
