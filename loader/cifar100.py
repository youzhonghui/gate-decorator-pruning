"""
 * Copyright (C) 2019 Zhonghui You
 * If you are using this code in your research, please cite the paper:
 * Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks, in NeurIPS 2019.
"""

import torch
import torchvision
from torchvision import transforms

from config import cfg

def get_cifar100():
    print('==> Preparing Cifar100 data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data/pytorch', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.data.batch_size, shuffle=cfg.data.shuffle, num_workers=cfg.data.num_workers)

    testset = torchvision.datasets.CIFAR100(root='./data/pytorch', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.data.test_batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    return train_loader, test_loader
