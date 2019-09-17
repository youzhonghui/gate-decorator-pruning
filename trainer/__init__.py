from trainer.normal import NormalTrainer
from config import cfg

def get_trainer():
    pair = {
        'normal': NormalTrainer
    }
    assert (cfg.train.trainer in pair)

    return pair[cfg.train.trainer]()
