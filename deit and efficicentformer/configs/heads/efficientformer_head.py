# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from .cls_head import ClsHead


class EfficientFormerClsHead(ClsHead):
    """EfficientFormer classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        distillation (bool): Whether use a additional distilled head.
            Defaults to True.
        init_cfg (dict): The extra initialization configs. Defaults to
            ``dict(type='Normal', layer='Linear', std=0.01)``.
    """
    
    def post_process(self, preds):
        return preds


    def __init__(self,
                 num_classes,
                 in_channels,
                 distillation=True,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(EfficientFormerClsHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dist = distillation

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.head = nn.Linear(self.in_channels, self.num_classes)
        if self.dist:
            self.dist_head = nn.Linear(self.in_channels, self.num_classes)

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[tuple[tensor, tensor]]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. Every item should be a tuple which
                includes patch token and cls token. The cls token will be used
                to classify and the shape of it should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(x)
        cls_score = self.head(x)
        if self.dist:
            cls_score = (cls_score + self.dist_head(x)) / 2

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        cls_score = self.head(x)

        if self.dist:
            dist_score = self.dist_head(x)
            # 两个 head 的输出都用于计算 loss
            cls_loss = self.loss(cls_score, gt_label, **kwargs)
            distill_loss = self.loss(dist_score, gt_label, **kwargs)

            # 自定义加权（λ = 0.5 可调）
            loss = {
                'loss_cls': cls_loss['loss'],
                'loss_distill': 0.5 * distill_loss['loss']  # 蒸馏分支损失
            }
            return loss
        else:
            losses = self.loss(cls_score, gt_label, **kwargs)
            return losses

