"""
 * Copyright (C) 2019 Zhonghui You
 * If you are using this code in your research, please cite the paper:
 * Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks, in NeurIPS 2019.
"""

import argparse
import json
from utils import dotdict

def make_as_dotdict(obj):
    if type(obj) is dict:
        obj = dotdict(obj)
        for key in obj:
            if type(obj[key]) is dict:
                obj[key] = make_as_dotdict(obj[key])
    return obj

def parse():
    print('Parsing config file...')
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.json",
        help="Configuration file to use"
    )
    cli_args = parser.parse_args()

    with open(cli_args.config) as fp:
        config = make_as_dotdict(json.loads(fp.read()))
    print(json.dumps(config, indent=4, sort_keys=True))
    return config

class Singleton(object):
    _instance = None
    def __new__(cls, *args, **kw):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kw)  
        return cls._instance 

class Config(Singleton):
    def __init__(self):
        self._cfg = dotdict({})
        try:
            self._cfg = parse()
        except:
            pass

    def __getattr__(self, name):
        if name == '_cfg':
            super().__setattr__(name)
        else:
            return self._cfg.__getattr__(name)

    def __setattr__(self, name, val):
        if name == '_cfg':
            super().__setattr__(name, val)
        else:
            self._cfg.__setattr__(name, val)

    def __delattr__(self, name):
        return self._cfg.__delitem__(name)

    def copy(self, new_config):
        self._cfg = make_as_dotdict(new_config)

cfg = Config()

def parse_from_dict(d):
    global cfg
    assert type(d) == dict
    cfg.copy(d)
