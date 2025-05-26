import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses import *
from core.evaluations import Accuracy
from ..common.base_module import BaseModule


class ClsHead(BaseModule):

    def __init__(self,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, ),
                 cal_acc=False,
                 init_cfg=None):
        super(ClsHead, self).__init__(init_cfg=init_cfg)

        assert isinstance(loss, dict)
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        loss_type = loss.pop('type')
        self.loss_weight = loss.pop('loss_weight', 1.0)

        if loss_type == 'CrossEntropyLoss':
            # 获取 label_smoothing（如果支持）
            label_smoothing = loss.pop('label_smoothing', 0.0)
            try:
                self.compute_loss = nn.CrossEntropyLoss(
                    reduction='mean',
                    label_smoothing=label_smoothing,
                    **loss
                )
            except TypeError:
                # PyTorch < 1.10 不支持 label_smoothing
                print("[Warning] label_smoothing not supported in this PyTorch version, using default CrossEntropyLoss.")
                self.compute_loss = nn.CrossEntropyLoss(reduction='mean')
        else:
            # 使用你自定义的 loss
            self.compute_loss = eval(loss_type)(**loss)

        self.compute_accuracy = Accuracy(topk=self.topk)
        self.cal_acc = cal_acc

    def loss(self, cls_score, gt_label, **kwargs):
        num_samples = len(cls_score)
        losses = dict()

        # compute loss (手动乘 loss_weight)
        loss = self.compute_loss(cls_score, gt_label) * self.loss_weight

        if self.cal_acc:
            # compute accuracy
            with torch.no_grad():
                acc = self.compute_accuracy(cls_score, gt_label)
                assert len(acc) == len(self.topk)
                losses['accuracy'] = {
                    f'top-{k}': a
                    for k, a in zip(self.topk, acc)
                }
        losses['loss'] = loss

        return losses

    def forward_train(self, cls_score, gt_label, **kwargs):
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]

        warnings.warn(
            'The input of ClsHead should be already logits. '
            'Please modify the backbone if you want to get pre-logits feature.'
        )
        return x

    def simple_test(self, cls_score, softmax=True, post_process=False):
        """Inference without augmentation."""
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred
