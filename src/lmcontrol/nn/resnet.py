from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

from torchvision.utils import _log_api_usage_once
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1, conv3x3

import torch
import torch.nn as nn
from torch import Tensor


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        planes: List[int],
        num_outputs: int,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        return_embeddings: bool = False
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = planes[0]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.return_embeddings = return_embeddings

        if planes is None:
            planes = [64, 128, 256, 512]

        self.layer1 = self._make_layer(block, planes[0], layers[0])
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        idx = max(i for i in range(len(layers)) if layers[i] != 0)

        if block == BasicBlock:
            expansion = [1, 1, 1, 1]
        elif block == Bottleneck:
            expansion = [4, 4, 4, 4]


        idx = max(i for i in range(len(layers)) if layers[i] != 0)
        self.n_features = planes[idx] * expansion[idx]

        self.fc = nn.Linear(self.n_features, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        if blocks == 0:
            return nn.Identity()
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )

        self.inplanes = planes * block.expansion  # Not good to change self

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if not self.return_embeddings:
            x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def get_block(block_type):
    """This is the type of sub-block to use to build a resnet block"""
    return {
        'BasicBlock': BasicBlock,
        'Bottleneck': Bottleneck
    }[block_type]


def get_planes(plane_cmd):
    """This is the size (i.e. number of planes/channels) of each layer in a resnet block"""
    p = int(plane_cmd)
    return [2**p, 2**(p+1), 2**(p+2), 2**(p+3)]


def get_layers(layers_cmd):
    """This is actually the number of resnet blocks to use, where we fix the size of each resnet block to be 1 layer."""
    l = int(layers_cmd)
    layers = [1 if i < l else 0 for i in range(4)]
    return layers


def add_args(parser):
    parser.add_argument("--block", type=get_block, choices=['BasicBlock', 'Bottleneck'], help="type of block to use in the model", default='Bottleneck')
    parser.add_argument("--planes", type=get_planes, choices=['3', '4'], help="list of number of planes for each layer", default='4')
    parser.add_argument("--layers", type=get_layers, choices=['1', '2', '3', '4'], help="list of number of layers in each stage", default='4')
