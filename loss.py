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
