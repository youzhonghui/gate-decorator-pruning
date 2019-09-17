import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
import math

from loader import get_loader
from models import get_model
from trainer import get_trainer
from loss import get_criterion

from utils import dotdict
from config import cfg
from logger import logger


def _sgdr(epoch):
    lr_min, lr_max = cfg.train.sgdr.lr_min, cfg.train.sgdr.lr_max
    restart_period = cfg.train.sgdr.restart_period
    _epoch = epoch - cfg.train.sgdr.warm_up

    while _epoch/restart_period > 1.:
        _epoch = _epoch - restart_period
        restart_period = restart_period * 2.

    radians = math.pi*(_epoch/restart_period)
    return lr_min + (lr_max - lr_min) *  0.5*(1.0 + math.cos(radians))

def _step_lr(epoch):
    v = 0.0
    for max_e, lr_v in cfg.train.steplr:
        v = lr_v
        if epoch <= max_e:
            break
    return v

def get_lr_func():
    if cfg.train.steplr is not None:
        return _step_lr
    elif cfg.train.sgdr is not None:
        return _sgdr
    else:
        assert False

def adjust_learning_rate(epoch, pack):
    if pack.optimizer is None:
        if cfg.train.optim == 'sgd' or cfg.train.optim is None:
            pack.optimizer = optim.SGD(
                pack.net.parameters(),
                lr=1,
                momentum=cfg.train.momentum,
                weight_decay=cfg.train.weight_decay,
                nesterov=cfg.train.nesterov
            )
        else:
            print('WRONG OPTIM SETTING!')
            assert False
        pack.lr_scheduler = optim.lr_scheduler.LambdaLR(pack.optimizer, get_lr_func())

    pack.lr_scheduler.step(epoch)
    return pack.lr_scheduler.get_lr()

def recover_pack():
    train_loader, test_loader = get_loader()

    pack = dotdict({
        'net': get_model(),
        'train_loader': train_loader,
        'test_loader': test_loader,
        'trainer': get_trainer(),
        'criterion': get_criterion(),
        'optimizer': None,
        'lr_scheduler': None
    })

    adjust_learning_rate(cfg.base.epoch, pack)
    return pack

def set_seeds():
    torch.manual_seed(cfg.base.seed)
    if cfg.base.cuda:
        torch.cuda.manual_seed_all(cfg.base.seed)
        torch.backends.cudnn.deterministic = True
        if cfg.base.fp16:
            torch.backends.cudnn.enabled = True
            # torch.backends.cudnn.benchmark = True
    np.random.seed(cfg.base.seed)
    random.seed(cfg.base.seed)


def main():
    set_seeds()
    pack = recover_pack()

    for epoch in range(cfg.base.epoch + 1, cfg.train.max_epoch + 1):
        lr = adjust_learning_rate(epoch, pack)
        info = pack.trainer.train(pack)
        info.update(pack.trainer.test(pack))
        info.update({'LR': lr})
        print(epoch, info)
        logger.save_record(epoch, info)
        if epoch % cfg.base.model_saving_interval == 0:
            logger.save_network(epoch, pack.net)

if __name__ == '__main__':
    main()
