"""
 * Copyright (C) 2019 Zhonghui You
 * If you are using this code in your research, please cite the paper:
 * Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks, in NeurIPS 2019.
"""

import torch
import torchvision
from torchvision import transforms
import os

from config import cfg

def _get_loaders(root):
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    val_dataset = torchvision.datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle, 
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.data.test_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

def get_imagenet():
    return _get_loaders('./data/imagenet12')
