# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from configs.basic.build_layer import build_norm_layer
from ..common.transformer import FFN, PatchEmbed
from core.initialize.weight_init import trunc_normal_
from ..common.base_module import BaseModule, ModuleList
from ..common import MultiheadAttention, resize_pos_embed, to_2tuple


class TransformerEncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(TransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias)

        self.norm2_name, norm2 = build_norm_layer(norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        super(TransformerEncoderLayer, self).init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = self.ffn(self.norm2(x), identity=x)
        return x



class VisionTransformer(BaseModule):
    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 768,
                'num_layers': 8,
                'num_heads': 8,
                'feedforward_channels': 768 * 3,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }),
        **dict.fromkeys(
            ['h', 'huge'],
            {
                # The same as the implementation in MAE
                # <https://arxiv.org/abs/2111.06377>
                'embed_dims': 1280,
                'num_layers': 32,
                'num_heads': 16,
                'feedforward_channels': 5120
            }),
        **dict.fromkeys(
            ['eva-g', 'eva-giant'],
            {
                # The implementation in EVA
                # <https://arxiv.org/abs/2211.07636>
                'embed_dims': 1408,
                'num_layers': 40,
                'num_heads': 16,
                'feedforward_channels': 6144
            }),
        **dict.fromkeys(
            ['deit-t', 'deit-tiny'], {
                'embed_dims': 192,
                'num_layers': 12,
                'num_heads': 3,
                'feedforward_channels': 192 * 4
            }),
        **dict.fromkeys(
            ['deit-s', 'deit-small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 6,
                'feedforward_channels': 384 * 4
            }),
        **dict.fromkeys(
            ['deit-b', 'deit-base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 768 * 4
            }),
    }
    # Some structures have multiple extra tokens, like DeiT.
    num_extra_tokens = 1  # cls_token

    def __init__(self,
                 arch='deit-tiny',
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 with_cls_token=True,
                 avg_token=False,
                 frozen_stages=-1,
                 output_cls_token=True,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 pre_norm=False,
                 use_distillation=False,
                 init_cfg=None):
        super(VisionTransformer, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.img_size = to_2tuple(img_size)

        self.use_distillation = use_distillation
        self.num_extra_tokens = 2 if self.use_distillation else 1

        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=not pre_norm,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        if self.use_distillation:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_extra_tokens, self.embed_dims))
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)
        self.drop_after_pos = nn.Dropout(p=drop_rate)

                # ✅ 插入这里 👇
        if pre_norm:
            _, norm_layer = build_norm_layer(norm_cfg, self.embed_dims, postfix=1)
        else:
            norm_layer = nn.Identity()
        self.pre_norm = norm_layer
        
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        self.out_indices = [i if i >= 0 else self.num_layers + i for i in out_indices]

        dpr = np.linspace(0, drop_path_rate, self.num_layers)
        self.layers = ModuleList()
        layer_cfgs = [layer_cfgs] * self.num_layers if isinstance(layer_cfgs, dict) else layer_cfgs
        for i in range(self.num_layers):
            layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg
            )
            layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayer(**layer_cfg))

        self.frozen_stages = frozen_stages
        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        self.avg_token = avg_token
        if avg_token:
            self.norm2_name, norm2 = build_norm_layer(norm_cfg, self.embed_dims, postfix=2)
            self.add_module(self.norm2_name, norm2)

        if self.frozen_stages > 0:
            self._freeze_stages()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        super(VisionTransformer, self).init_weights()
        if not (isinstance(self.init_cfg, dict) and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name in state_dict and self.pos_embed.shape != state_dict[name].shape:
            ckpt_pos_embed_shape = state_dict[name].shape
            print(f'Resize the pos_embed shape from {ckpt_pos_embed_shape} to {self.pos_embed.shape}.')
            ckpt_pos_embed_shape = to_2tuple(int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
            pos_embed_shape = self.patch_embed.init_out_size
            state_dict[name] = resize_pos_embed(state_dict[name], ckpt_pos_embed_shape, pos_embed_shape, self.interpolate_mode, self.num_extra_tokens)

    def _freeze_stages(self):
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        self.drop_after_pos.eval()
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        self.cls_token.requires_grad = False
        if self.use_distillation:
            self.dist_token.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        if self.frozen_stages == len(self.layers) and self.final_norm:
            self.norm1.eval()
            for param in self.norm1.parameters():
                param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.use_distillation:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)
        x = self.pre_norm(x)
        if not self.with_cls_token:
            x = x[:, 1:]

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)
            if i in self.out_indices:
                B, _, C = x.shape
                if self.use_distillation:
                    cls_token = x[:, 0]
                    dist_token = x[:, 1]
                    patch_token = x[:, 2:].reshape(B, *patch_resolution, C)
                elif self.with_cls_token:
                    cls_token = x[:, 0]
                    patch_token = x[:, 1:].reshape(B, *patch_resolution, C)
                else:
                    cls_token = None
                    patch_token = x.reshape(B, *patch_resolution, C)
                patch_token = patch_token.permute(0, 3, 1, 2)
                if self.avg_token:
                    patch_token = patch_token.permute(0, 2, 3, 1)
                    patch_token = patch_token.reshape(B, patch_resolution[0] * patch_resolution[1], C).mean(dim=1)
                    patch_token = self.norm2(patch_token)
                if self.use_distillation:
                    out = [patch_token, cls_token, dist_token]
                elif self.output_cls_token:
                    out = [patch_token, cls_token]
                else:
                    out = patch_token
                outs.append(out)

        return tuple(outs)