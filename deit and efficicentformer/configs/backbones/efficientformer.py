# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from typing import Optional, Sequence

import torch
import torch.nn as nn
from ..common import ConvModule, DropPath, BaseModule, ModuleList, Sequential, LayerScale
from ..basic.build_layer import build_activation_layer, build_norm_layer

from .poolformer import Pooling


class AttentionWithBias(BaseModule):


    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 key_dim=32,
                 attn_ratio=4.,
                 resolution=7,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.attn_ratio = attn_ratio
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        h = self.dh + self.nh_kd * 2
        self.qkv = nn.Linear(embed_dims, h)
        self.proj = nn.Linear(self.dh, embed_dims)

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        """change the mode of model."""
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):

        B, N, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.d], dim=-1)

        attn = ((q @ k.transpose(-2, -1)) * self.scale +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class Flat(nn.Module):
    """Flat the input from (B, C, H, W) to (B, H*W, C)."""

    def __init__(self, ):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = x.flatten(2).transpose(1, 2)
        return x


class LinearMlp(BaseModule):


    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_cfg=dict(type='GELU'),
                 drop=0.,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):

        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x


class ConvMlp(BaseModule):


    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 drop=0.,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.norm1 = build_norm_layer(norm_cfg, hidden_features)[1]
        self.norm2 = build_norm_layer(norm_cfg, out_features)[1]

        self.drop = nn.Dropout(drop)

    def forward(self, x):


        x = self.act(self.norm1(self.fc1(x)))
        x = self.drop(x)
        x = self.norm2(self.fc2(x))
        x = self.drop(x)
        return x


class Meta3D(BaseModule):


    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 drop=0.,
                 drop_path=0.,
                 use_layer_scale=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.token_mixer = AttentionWithBias(dim)
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LinearMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        if use_layer_scale:
            self.ls1 = LayerScale(dim)
            self.ls2 = LayerScale(dim)
        else:
            self.ls1, self.ls2 = nn.Identity(), nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.token_mixer(self.norm1(x))))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class Meta4D(BaseModule):


    def __init__(self,
                 dim,
                 pool_size=3,
                 mlp_ratio=4.,
                 act_cfg=dict(type='GELU'),
                 drop=0.,
                 drop_path=0.,
                 use_layer_scale=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.token_mixer = Pooling(pool_size=pool_size)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        if use_layer_scale:
            self.ls1 = LayerScale(dim, data_format='channels_first')
            self.ls2 = LayerScale(dim, data_format='channels_first')
        else:
            self.ls1, self.ls2 = nn.Identity(), nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.token_mixer(x)))
        x = x + self.drop_path(self.ls2(self.mlp(x)))
        return x


def basic_blocks(in_channels,
                 out_channels,
                 index,
                 layers,
                 pool_size=3,
                 mlp_ratio=4.,
                 act_cfg=dict(type='GELU'),
                 drop_rate=.0,
                 drop_path_rate=0.,
                 use_layer_scale=True,
                 vit_num=1,
                 has_downsamper=False):
    """generate EfficientFormer blocks for a stage."""
    blocks = []
    if has_downsamper:
        blocks.append(
            ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
                norm_cfg=dict(type='BN'),
                act_cfg=None))
    if index == 3 and vit_num == layers[index]:
        blocks.append(Flat())
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (
            sum(layers) - 1)
        if index == 3 and layers[index] - block_idx <= vit_num:
            blocks.append(
                Meta3D(
                    out_channels,
                    mlp_ratio=mlp_ratio,
                    act_cfg=act_cfg,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                ))
        else:
            blocks.append(
                Meta4D(
                    out_channels,
                    pool_size=pool_size,
                    act_cfg=act_cfg,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale))
            if index == 3 and layers[index] - block_idx - 1 == vit_num:
                blocks.append(Flat())
    blocks = nn.Sequential(*blocks)
    return blocks


class EfficientFormer(BaseModule):
 
   # noqa: E501

    # --layers: [x,x,x,x], numbers of layers for the four stages
    # --embed_dims: [x,x,x,x], embedding dims for the four stages
    # --downsamples: [x,x,x,x], has downsample or not in the four stages
    # --vit_num：(int), the num of vit blocks in the last stage
    arch_settings = {
        'l1': {
            'layers': [3, 2, 6, 4],
            'embed_dims': [48, 96, 224, 448],
            'downsamples': [False, True, True, True],
            'vit_num': 1,
        },
        'l3': {
            'layers': [4, 4, 12, 6],
            'embed_dims': [64, 128, 320, 512],
            'downsamples': [False, True, True, True],
            'vit_num': 4,
        },
        'l7': {
            'layers': [6, 6, 18, 8],
            'embed_dims': [96, 192, 384, 768],
            'downsamples': [False, True, True, True],
            'vit_num': 8,
        },
    }

    def __init__(self,
                 arch='l1',
                 in_channels=3,
                 pool_size=3,
                 mlp_ratios=4,
                 reshape_last_feat=False,
                 out_indices=-1,
                 frozen_stages=-1,
                 act_cfg=dict(type='GELU'),
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_layer_scale=True,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)
        self.num_extra_tokens = 0  # no cls_token, no dist_token

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            default_keys = set(self.arch_settings['l1'].keys())
            assert set(arch.keys()) == default_keys, \
                f'The arch dict must have {default_keys}, ' \
                f'but got {list(arch.keys())}.'

        self.layers = arch['layers']
        self.embed_dims = arch['embed_dims']
        self.downsamples = arch['downsamples']
        assert isinstance(self.layers, list) and isinstance(
            self.embed_dims, list) and isinstance(self.downsamples, list)
        assert len(self.layers) == len(self.embed_dims) == len(
            self.downsamples)

        self.vit_num = arch['vit_num']
        self.reshape_last_feat = reshape_last_feat

        assert self.vit_num >= 0, "'vit_num' must be an integer " \
                                  'greater than or equal to 0.'
        assert self.vit_num <= self.layers[-1], (
            "'vit_num' must be an integer smaller than layer number")

        self._make_stem(in_channels, self.embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(self.layers)):
            if i != 0:
                in_channels = self.embed_dims[i - 1]
            else:
                in_channels = self.embed_dims[i]
            out_channels = self.embed_dims[i]
            stage = basic_blocks(
                in_channels,
                out_channels,
                i,
                self.layers,
                pool_size=pool_size,
                mlp_ratio=mlp_ratios,
                act_cfg=act_cfg,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                vit_num=self.vit_num,
                use_layer_scale=use_layer_scale,
                has_downsamper=self.downsamples[i])
            network.append(stage)

        self.network = ModuleList(network)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'

        self.out_indices = out_indices
        for i_layer in self.out_indices:
            if not self.reshape_last_feat and \
                    i_layer == 3 and self.vit_num > 0:
                layer = build_norm_layer(
                    dict(type='LN'), self.embed_dims[i_layer])[1]
            else:
                # use GN with 1 group as channel-first LN2D
                layer = build_norm_layer(
                    dict(type='GN', num_groups=1), self.embed_dims[i_layer])[1]

            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _make_stem(self, in_channels: int, stem_channels: int):
        """make 2-ConvBNReLu stem layer."""
        self.patch_embed = Sequential(
            ConvModule(
                in_channels,
                stem_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
                conv_cfg=None,
                norm_cfg=dict(type='BN'),
                inplace=True),
            ConvModule(
                stem_channels // 2,
                stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
                conv_cfg=None,
                norm_cfg=dict(type='BN'),
                inplace=True))

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            if idx == len(self.network) - 1:
                N, _, H, W = x.shape
                if self.downsamples[idx]:
                    H, W = H // 2, W // 2
            x = block(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')

                if idx == len(self.network) - 1 and x.dim() == 3:
                    # when ``vit-num`` > 0 and in the last stage,
                    # if `self.reshape_last_feat`` is True, reshape the
                    # features to `BCHW` format before the final normalization.
                    # if `self.reshape_last_feat`` is False, do
                    # normalization directly and permute the features to `BCN`.
                    if self.reshape_last_feat:
                        x = x.permute((0, 2, 1)).reshape(N, -1, H, W)
                        x_out = norm_layer(x)
                    else:
                        x_out = norm_layer(x).permute((0, 2, 1))
                else:
                    x_out = norm_layer(x)

                outs.append(x_out.contiguous())
        return tuple(outs)

    def forward(self, x):
        # input embedding
        x = self.patch_embed(x)
        # through stages
        x = self.forward_tokens(x)
        return x

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            # Include both block and downsample layer.
            module = self.network[i]
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(EfficientFormer, self).train(mode)
        self._freeze_stages()
