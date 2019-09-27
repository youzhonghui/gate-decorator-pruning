# Gate Decorator (NeurIPS 2019)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/NifTK/NiftyNet/blob/dev/LICENSE)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

This repo contains required scripts to reproduce results from paper:

_Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks_

### Requirements

python 3.6+ and PyTorch 1.0+

### Installation

1. clone the code
2. pip install --upgrade git+https://github.com/youzhonghui/pytorch-OpCounter.git
3. pip install tqdm

### How to use

In the `run/resnet-56` folder, we provide an example which **reduces the FLOPs of resnet-56 by 70%**, but still maintains **93.15%** accuracy on CIFAR-10:
1. The `run/resnet-56/resnet56_prune.ipynb` prunes the network with Tick-Tock framework.
2. The `run/resnet-56/finetune.ipynb` shows how to finetune the pruned network to get better results.

If you want to run the demo code, you may need to install [jupyter notebook](https://jupyter.org/)

### Todo

- [x] Basic running example.
- [x] PyTorch 1.2 compatibility test.
- [x] The command-line execution demo.
- [ ] ResNet-50 pruned model.

### Citation

If you use this code for your research, please cite our paper:
```
@inproceedings{zhonghui2019gate,
  title={Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks},
  author={Zhonghui You and
          Kun Yan and
          Jinmian Ye and
          Meng Ma and
          Ping Wang},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```