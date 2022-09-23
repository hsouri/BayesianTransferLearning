"""Some helper functions for PyTorch
"""
import os
from pathlib import Path
import torch.nn as nn
import torch
import shutil
import datetime
from ...solo_learn.methods.custom_resnet import get_resnet
from ...solo_learn.methods.wide_resnet import resnet50x1
from torchvision.models import resnet18, resnet50
from ...swag.swag import SWAG
from torch.utils.data import DataLoader
from collections import OrderedDict

from .eval import eval_fn, ImagenetValidationDataset
from ...solo_learn.utils.backbones import (
    swin_base,
    swin_large,
    swin_small,
    swin_tiny,
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
)


def get_backbone(args):
    backbone_model = {
        "resnet18": resnet18,
        "resnet50": resnet50,
        "resnet50_c": get_resnet,
        "vit_tiny": vit_tiny,
        "vit_small": vit_small,
        "vit_base": vit_base,
        "vit_large": vit_large,
        "swin_tiny": swin_tiny,
        "swin_small": swin_small,
        "swin_base": swin_base,
        "swin_large": swin_large,
    }[args['encoder']]

    # initialize encoder
    if args['encoder'] == 'resnet50_c':
        backbone = resnet50x1()
    try:
        kwargs = args['backbone_args']
        backbone = backbone_model(pretrained=args['pytorch_pretrain'], progress=args['pytorch_pretrain'], **kwargs)
    except:
        kwargs = args['backbone_args']
        backbone = backbone_model(**kwargs)

    if "resnet" in args['encoder']:
        # remove fc layer
        backbone.fc = nn.Identity()
    backbone.train()
    return backbone


def get_sample_dir(samples_dir):
    samples_dir = samples_dir + datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    samples_dir = Path(samples_dir or '.') / '.samples'
    if os.path.exists(samples_dir):
        shutil.rmtree(samples_dir)
    os.makedirs(samples_dir, exist_ok=False)
    return samples_dir

def load_weights2(net: torch.nn.Module, path: str, remove_classifier: bool = False, eval_model: bool = True,
                 val_path: str = '/path/to/your/dir'):
    """Load network weights (encoder and classifier) and test it on imagenet"""
    state = torch.load(path)
    if 'state_dict' in state:
        state = state["state_dict"]
    if 'resnet' in state:
        if not remove_classifier:
            classifier_state = OrderedDict()
            if 'fc.weight' in state['resnet']:
                classifier_state['weight'] = state['resnet']['fc.weight']
                classifier_state['bias'] = state['resnet']['fc.bias']
                net.classifier.load_state_dict(classifier_state, strict=True)
        del state['resnet']['fc.weight']
        del state['resnet']['fc.bias']
        net.encoder.load_state_dict(state['resnet'], strict=True)
    if 'head' in state:
        net.projector.load_state_dict(state['head'], strict=True)
    if eval_model:
        dataset = ImagenetValidationDataset(val_path)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=8)
        eval_fn(net, data_loader)
    return state


def load_weights_simclr(net: torch.nn.Module, path: str, remove_classifier: bool = False, eval_model: bool = True,
                 val_path: str = '/path/to/your/dir'):
    """Load network weights (encoder and classifier) and test it on imagenet"""
    state = torch.load(path)
    if 'state_dict' in state:
        state = state["state_dict"]
    if 'resnet' in state:
        net.load_state_dict(state['resnet'], strict=True)
    if eval_model:
        dataset = ImagenetValidationDataset(val_path)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=8)
        eval_fn(net, data_loader)
    return state


def load_weights(net, load_classifier=False, path: str = None):

    checkpoint = torch.load(path + '_model.pt', map_location=torch.device('cpu'))
    net.backbone.load_state_dict(checkpoint)

    # TODO: fix loading from ssl_prior
    # state = torch.load(path)
    # #if state
    # if isinstance(state, SWAG):
    #     state.sample(0.)
    #     state = state.base_model.state_dict()
    # if 'state_dict' in state:
    #     state = state['state_dict']
    # for k in list(state.keys()):
    #     if "base_model.encoder" in k:
    #         state[k.replace("base_model.encoder.", "")] = state[k]
    #         del state[k]
    # if 'resnet' in state:
    #     state = state['resnet']
    # if load_classifier:
    #     for k in list(state.keys()):
    #         if "module.linear" in k:
    #             state[k.replace("module.linear.", "")] = state[k]
    #             del state[k]
    #     net.backbone.load_state_dict(state, strict=True)
    # else:
    #     ss = net.backbone.load_state_dict(state, strict=False)
    #     print(ss)
