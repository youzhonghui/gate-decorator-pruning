import torch
from config import cfg

def get_vgg16_for_cifar():
    from models.cifar.vgg import VGG
    return VGG('VGG16', cfg.model.num_class)

def get_resnet50_for_imagenet():
    from models.imagenet.resnet50 import Resnet50
    return Resnet50(cfg.model.num_class)

def get_resnet56():
    from models.cifar.resnet56 import resnet56
    return resnet56(cfg.model.num_class)

def get_model():
    pair = {
        'cifar.vgg16': get_vgg16_for_cifar,
        'resnet50': get_resnet50_for_imagenet,
        'cifar.resnet56': get_resnet56
    }

    model = pair[cfg.model.name]()

    if cfg.base.checkpoint_path != '':
        print('restore checkpoint: ' + cfg.base.checkpoint_path)
        model.load_state_dict(torch.load(cfg.base.checkpoint_path, map_location='cpu' if not cfg.base.cuda else 'cuda'))

    if cfg.base.cuda:
        model = model.cuda()

    if cfg.base.multi_gpus:
        model = torch.nn.DataParallel(model)
    return model
