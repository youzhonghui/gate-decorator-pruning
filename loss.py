"""
 * Copyright (C) 2019 Zhonghui You
 * If you are using this code in your research, please cite the paper:
 * Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks, in NeurIPS 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from config import cfg

def get_criterion():
    pair = {
        'softmax': nn.CrossEntropyLoss()
    }

    assert (cfg.loss.criterion in pair)
    criterion = pair[cfg.loss.criterion]
    return criterion
