from functools import partial

import torch
import numpy as np
import pretrainedmodels
from torch import nn
from torch.nn import functional as F
import torchvision.models as M
# from efficientnet_pytorch import EfficientNet

from .utils import ON_KAGGLE


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def create_net(net_cls, pretrained: bool):
    if ON_KAGGLE and pretrained:
        net = net_cls()
        model_name = net_cls.__name__
        weights_path = f'../input/{model_name}/{model_name}.pth'
        net.load_state_dict(torch.load(weights_path))
    else:
        net = net_cls(pretrained=pretrained)
    return net


def get_head(nf: int, n_classes):
    model = nn.Sequential(
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        Flatten(),
        # nn.BatchNorm1d(nf),
        nn.Dropout(p=0.25),
        nn.Linear(nf, n_classes)
        # nn.BatchNorm1d(nf),
        # nn.Dropout(p=0.25),
        # nn.Linear(nf, 1024),
        # nn.BatchNorm1d(1024),
        # nn.Dropout(p=0.25),
        # nn.Linear(1024, n_classes)
    )
    for i, module in enumerate(model):
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        if isinstance(module, nn.Linear):
            if getattr(module, "weight_v", None) is not None:
                print("Initing linear with weight normalization")
                assert model[i].weight_g is not None
            else:
                nn.init.kaiming_normal_(module.weight)
                print("Initing linear")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    return model


def get_seresnet_model(arch: str = "se_resnext101_32x4d", n_classes: int = 1103, pretrained=True):
    full = pretrainedmodels.__dict__[arch](
        pretrained='imagenet' if pretrained else None)
    model = nn.Sequential(
        nn.Sequential(full.layer0, full.layer1, full.layer2, full.layer3[:3]),
        nn.Sequential(full.layer3[3:], full.layer4),
        get_head(2048, n_classes))
    print(" | ".join([
        "{:,d}".format(np.sum([p.numel() for p in x.parameters()])) for x in model]))
    return model


def get_densenet_model(arch: str = "densenet169", n_classes: int = 1103, pretrained=True):
    full = pretrainedmodels.__dict__[arch](
        pretrained='imagenet' if pretrained else None)
    print(len(full.features))
    model = nn.Sequential(
        nn.Sequential(*full.features[:8]),
        nn.Sequential(*full.features[8:]),
        get_head(full.features[-1].num_features, n_classes))
    print(" | ".join([
        "{:,d}".format(np.sum([p.numel() for p in x.parameters()])) for x in model]))
    return model


class Swish(nn.Module):
    def forward(self, x):
        """ Swish activation function """
        return x * torch.sigmoid(x)


# def get_efficientnet(arch: str = "efficientnet-b3", n_classes: int = 1103, pretrained=True):
#     if pretrained == True:
#         base_model = EfficientNet.from_pretrained(arch)
#     else:
#         base_model = EfficientNet.from_name(arch)
#     # print(base_model)
#     print(len(base_model._blocks))
#     model = nn.Sequential(
#         nn.Sequential(
#             base_model._conv_stem,
#             base_model._bn0,
#             Swish(),
#             *base_model._blocks[:20]
#         ),
#         nn.Sequential(*base_model._blocks[20:]),
#         nn.Sequential(
#             base_model._conv_head,
#             base_model._bn1,
#             Swish(),
#             *get_head(base_model._fc.in_features, n_classes)[1:],
#         )
#     )
#     print(" | ".join([
#         "{:,d}".format(np.sum([p.numel() for p in x.parameters()])) for x in model]))
#     return model
