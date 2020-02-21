"""
 * Copyright (C) 2019 Zhonghui You
 * If you are using this code in your research, please cite the paper:
 * Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks, in NeurIPS 2019.
"""

import torch

from config import cfg
import os
import json
import numpy as np


class MetricsRecorder():
    def __init__(self):
        self.rec = {}

    def add(self, pairs):
        for key, val in pairs.items():
            if key not in self.rec:
                self.rec[key] = []
            self.rec[key].append(val)

    def mean(self):
        r = {}
        for key, val in self.rec.items():
            r[key] = np.mean(val)
        return r

class Logger():
    def __init__(self):
        self.base_path = './logs/' + cfg.base.task_name
        self.logfile = self.base_path + '/log.json'
        self.cfgfile = self.base_path + '/cfg.json'

        if not os.path.isdir(self.base_path):
            os.makedirs(self.base_path, exist_ok=True)
            with open(self.logfile, 'w') as fp:
                json.dump({}, fp)
            with open(self.cfgfile, 'w') as fp:
                json.dump(cfg.raw(), fp)

    def save_record(self, epoch, record):
        with open(self.logfile) as fp:
            log = json.load(fp)

        log[str(epoch)] = record
        with open(self.logfile, 'w') as fp:
            json.dump(log, fp)

    def save_network(self, epoch, network):
        saving_path = self.base_path + '/ckp.%d.torch' % epoch
        print('saving model ...')
        if type(network) is torch.nn.DataParallel:
            torch.save(network.module.state_dict(), saving_path)
        else:
            torch.save(network.state_dict(), saving_path)

        cfg.base.epoch = epoch
        cfg.base.checkpoint_path = saving_path
        with open(self.cfgfile, 'w') as fp:
            json.dump(cfg.raw(), fp)

logger = None
if logger is None:
    logger = Logger()
