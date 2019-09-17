import torch
import torch.nn as nn

import os, contextlib
from thop import profile

def analyse_model(net, inputs):
    # silence
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            flops, params = profile(net, (inputs, ))
    return flops, params


def finetune(pack, lr_min, lr_max, T, mute=False):
    logs = []
    epoch = 0

    def iter_hook(curr_iter, total_iter):
        total = T * total_iter
        half = total / 2
        itered = epoch * total_iter + curr_iter
        if itered < half:
            _iter = epoch * total_iter + curr_iter
            _lr = (1- _iter / half) * lr_min + (_iter / half) * lr_max
        else:
            _iter = (epoch - T/2) * total_iter + curr_iter
            _lr = (1- _iter / half) * lr_max + (_iter / half) * lr_min

        for g in pack.optimizer.param_groups:
            g['lr'] = max(_lr, 0)
            g['momentum'] = 0.0

    for i in range(T):
        info = pack.trainer.train(pack, iter_hook = iter_hook)
        info.update(pack.trainer.test(pack))
        info.update({'LR': pack.optimizer.param_groups[0]['lr']})
        epoch += 1
        if not mute:
            print(info)
        logs.append(info)

    return logs
