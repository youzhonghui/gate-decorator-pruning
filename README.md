# Gate Decorator (NeurIPS 2019)

[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

This repo contains required scripts to reproduce results from paper:

Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks, in NeurIPS 2019.

### Requirements

python 3.6+ and PyTorch 1.0

The code has been tested only with PyTorch 1.0. We will test it with newer version later.

### Installation

1. clone the code
2. pip install --upgrade git+https://github.com/youzhonghui/pytorch-OpCounter.git
3. pip install tqdm
4. mkdir data

### How to use

In the `run/resnet-56` folder we provide an example to show how to use the code.

If you want to run the code, you may need install [jupyter notebook](https://jupyter.org/)

### TODO

- [x] Basic running example.
- [ ] PyTorch 1.2 compatibility test.
- [ ] ResNet-50 pruned model.

### Citation

If you use this code for your research, please cite our papers.
```
@inproceedings{zhonghui2019gate,
  title={Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks},
  author={Zhonghui You and
          Kun Yan and
          Jinmian Ye and
          Meng Ma and
          Ping Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```