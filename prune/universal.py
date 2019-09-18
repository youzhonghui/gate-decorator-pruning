"""
 * Copyright (C) 2019 Zhonghui You
 * If you are using this code in your research, please cite the paper:
 * Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks, in NeurIPS 2019.
"""

import torch
import torch.nn as nn

import numpy as np
import uuid

OBSERVE_TIMES = 5
FINISH_SIGNAL = 'finish'

class Meltable(nn.Module):
    def __init__(self):
        super(Meltable, self).__init__()

    @classmethod
    def melt_all(cls, net):
        def _melt(modules):
            keys = modules.keys()
            for k in keys:
                if len(modules[k]._modules) > 0:
                    _melt(modules[k]._modules)
                if isinstance(modules[k], Meltable):
                    modules[k] = modules[k].melt()

        _melt(net._modules)

    @classmethod
    def observe(cls, pack, lr):
        tmp = pack.train_loader
        if pack.tick_trainset is not None:
            pack.train_loader = pack.tick_trainset

        for m in pack.net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.abs_().add_(1e-3)

        def replace_relu(modules):
            keys = modules.keys()
            for k in keys:
                if len(modules[k]._modules) > 0:
                    replace_relu(modules[k]._modules)
                if isinstance(modules[k], nn.ReLU):
                    modules[k] = nn.LeakyReLU(inplace=True)
        replace_relu(pack.net._modules)

        count = 0
        def _freeze_bn(curr_iter, total_iter):
            for m in pack.net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            nonlocal count
            count += 1
            if count == OBSERVE_TIMES:
                return FINISH_SIGNAL
        info = pack.trainer.train(pack, iter_hook=_freeze_bn, update=False, mute=True)

        def recover_relu(modules):
            keys = modules.keys()
            for k in keys:
                if len(modules[k]._modules) > 0:
                    recover_relu(modules[k]._modules)
                if isinstance(modules[k], nn.LeakyReLU):
                    modules[k] = nn.ReLU(inplace=True)
        recover_relu(pack.net._modules)

        for m in pack.net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.abs_().add_(-1e-3)

        pack.train_loader = tmp


class GatedBatchNorm2d(Meltable):
    def __init__(self, bn, minimal_ratio = 0.1):
        super(GatedBatchNorm2d, self).__init__()
        assert isinstance(bn, nn.BatchNorm2d)
        self.bn = bn
        self.group_id = uuid.uuid1()

        self.channel_size = bn.weight.shape[0]
        self.minimal_filter = max(1, int(self.channel_size * minimal_ratio))
        self.device = bn.weight.device
        self._hook = None

        self.g = nn.Parameter(torch.ones(1, self.channel_size, 1, 1).to(self.device), requires_grad=True)
        self.register_buffer('area', torch.zeros(1).to(self.device))
        self.register_buffer('score', torch.zeros(1, self.channel_size, 1, 1).to(self.device))
        self.register_buffer('bn_mask', torch.ones(1, self.channel_size, 1, 1).to(self.device))
        
        self.extract_from_bn()

    def set_groupid(self, new_id):
        self.group_id = new_id

    def extra_repr(self):
        return '%d -> %d | ID: %s' % (self.channel_size, int(self.bn_mask.sum()), self.group_id)

    def extract_from_bn(self):
        # freeze bn weight
        with torch.no_grad():
            self.bn.bias.set_(torch.clamp(self.bn.bias / self.bn.weight, -10, 10))
            self.g.set_(self.g * self.bn.weight.view(1, -1, 1, 1))
            self.bn.weight.set_(torch.ones_like(self.bn.weight))
            self.bn.weight.requires_grad = False

    def reset_score(self):
        self.score.zero_()

    def cal_score(self, grad):
        # used for hook
        self.score += (grad * self.g).abs()

    def start_collecting_scores(self):
        if self._hook is not None:
            self._hook.remove()

        self._hook = self.g.register_hook(self.cal_score)

    def stop_collecting_scores(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
    
    def get_score(self, eta=0.0):
        # use self.bn_mask.sum() to calculate the number of input channel. eta should had been normed
        flops_reg = eta * int(self.area[0]) * self.bn_mask.sum()
        return ((self.score - flops_reg) * self.bn_mask).view(-1)

    def forward(self, x):
        x = self.bn(x) * self.g

        self.area[0] = x.shape[-1] * x.shape[-2]

        if self.bn_mask is not None:
            return x * self.bn_mask
        return x

    def melt(self):
        with torch.no_grad():
            mask = self.bn_mask.view(-1)
            replacer = nn.BatchNorm2d(int(self.bn_mask.sum())).to(self.bn.weight.device)
            replacer.running_var.set_(self.bn.running_var[mask != 0])
            replacer.running_mean.set_(self.bn.running_mean[mask != 0])
            replacer.weight.set_((self.bn.weight * self.g.view(-1))[mask != 0])
            replacer.bias.set_((self.bn.bias * self.g.view(-1))[mask != 0])
        return replacer

    @classmethod
    def transform(cls, net, minimal_ratio=0.1):
        r = []
        def _inject(modules):
            keys = modules.keys()
            for k in keys:
                if len(modules[k]._modules) > 0:
                    _inject(modules[k]._modules)
                if isinstance(modules[k], nn.BatchNorm2d):
                    modules[k] = GatedBatchNorm2d(modules[k], minimal_ratio)
                    r.append(modules[k])
        _inject(net._modules)
        return r


class FinalLinearObserver(Meltable):
    ''' assert was in the last layer. only input was masked '''
    def __init__(self, linear):
        super(FinalLinearObserver, self).__init__()
        assert isinstance(linear, nn.Linear)
        self.linear = linear
        self.in_mask = torch.zeros(linear.weight.shape[1]).to('cpu')
        self.f_hook = linear.register_forward_hook(self._forward_hook)
    
    def extra_repr(self):
        return '(%d, %d) -> (%d, %d)' % (
            int(self.linear.weight.shape[1]),
            int(self.linear.weight.shape[0]),
            int((self.in_mask != 0).sum()),
            int(self.linear.weight.shape[0]))

    def _forward_hook(self, m, _in, _out):
        x = _in[0]
        self.in_mask += x.data.abs().cpu().sum(0, keepdim=True).view(-1)

    def forward(self, x):
        return self.linear(x)

    def melt(self):
        with torch.no_grad():
            replacer = nn.Linear(int((self.in_mask != 0).sum()), self.linear.weight.shape[0]).to(self.linear.weight.device)
            replacer.weight.set_(self.linear.weight[:, self.in_mask != 0])
            replacer.bias.set_(self.linear.bias)
        return replacer


class Conv2dObserver(Meltable):
    def __init__(self, conv):
        super(Conv2dObserver, self).__init__()
        assert isinstance(conv, nn.Conv2d)
        self.conv = conv
        self.in_mask = torch.zeros(conv.in_channels).to('cpu')
        self.out_mask = torch.zeros(conv.out_channels).to('cpu')
        self.f_hook = conv.register_forward_hook(self._forward_hook)

    def extra_repr(self):
        return '(%d, %d) -> (%d, %d)' % (self.conv.in_channels, self.conv.out_channels, int((self.in_mask != 0).sum()), int((self.out_mask != 0).sum()))
    
    def _forward_hook(self, m, _in, _out):
        x = _in[0]
        self.in_mask += x.data.abs().sum(2, keepdim=True).sum(3, keepdim=True).cpu().sum(0, keepdim=True).view(-1)

    def _backward_hook(self, grad):
        self.out_mask += grad.data.abs().sum(2, keepdim=True).sum(3, keepdim=True).cpu().sum(0, keepdim=True).view(-1)
        new_grad = torch.ones_like(grad)
        return new_grad

    def forward(self, x):
        output = self.conv(x)
        noise = torch.zeros_like(output).normal_()
        output = output + noise
        if self.training:
            output.register_hook(self._backward_hook)
        return output

    def melt(self):
        if self.conv.groups == 1:
            groups = 1
        elif self.conv.groups == self.conv.out_channels:
            groups = int((self.out_mask != 0).sum())
        else:
            assert False

        replacer = nn.Conv2d(
            in_channels = int((self.in_mask != 0).sum()),
            out_channels = int((self.out_mask != 0).sum()),
            kernel_size = self.conv.kernel_size,
            stride = self.conv.stride,
            padding = self.conv.padding,
            dilation = self.conv.dilation,
            groups = groups,
            bias = (self.conv.bias is not None)
        ).to(self.conv.weight.device)

        with torch.no_grad():
            if self.conv.groups == 1:
                replacer.weight.set_(self.conv.weight[self.out_mask != 0][:, self.in_mask != 0])
            else:
                replacer.weight.set_(self.conv.weight[self.out_mask != 0])
            if self.conv.bias is not None:
                replacer.bias.set_(self.conv.bias[self.out_mask != 0])
        return replacer
    
    @classmethod
    def transform(cls, net):
        r = []
        def _inject(modules):
            keys = modules.keys()
            for k in keys:
                if len(modules[k]._modules) > 0:
                    _inject(modules[k]._modules)
                if isinstance(modules[k], nn.Conv2d):
                    modules[k] = Conv2dObserver(modules[k])
                    r.append(modules[k])
        _inject(net._modules)
        return r

# -------------------------------------------------------------------------------------------------------

def get_gate_sparse_loss(masks, sparse_lambda):
    def _loss_hook(data, label, logits):
        loss = 0.0
        for gbn in masks:
            if isinstance(gbn, GatedBatchNorm2d):
                loss += gbn.g.abs().sum()
        return sparse_lambda * loss

    return _loss_hook

class IterRecoverFramework():
    def __init__(self, pack, masks, sparse_lambda=1e-5, flops_eta=0.0, minium_filter=10):
        self.pack = pack
        self.masks = masks
        self.sparse_loss_hook = get_gate_sparse_loss(masks, sparse_lambda)
        self.logs = []
        # minium_filter would be delete
        # self.minium_filter = minium_filter
        self.sparse_lambda = sparse_lambda
        self.flops_eta = flops_eta
        self.eta_scale_factor = 1.0

        self.total_filters = sum([m.bn.weight.shape[0] for m in masks])
        self.pruned_filters = 0

    def recover(self, lr, test):
        for gbn in self.masks:
            if isinstance(gbn, GatedBatchNorm2d):
                gbn.reset_score()
                gbn.start_collecting_scores()

        for g in self.pack.optimizer.param_groups:
            g['lr'] = lr

        tmp = self.pack.train_loader
        self.pack.train_loader = self.pack.tick_trainset
        info = self.pack.trainer.train(self.pack)
        self.pack.train_loader = tmp

        if test:
            info.update(self.pack.trainer.test(self.pack))

        info.update({'LR': lr})

        for gbn in self.masks:
            if isinstance(gbn, GatedBatchNorm2d):
                gbn.stop_collecting_scores()
        
        return info

    def get_threshold(self, status, num):
        '''
            input score list from layers, and the number of filter to prune
        '''
        total_filters, left_filters = 0, 0
        filtered_score_list = []
        
        for group_id, v in status.items():
            total_filters += len(v['score']) * v['count']
            left_filters += int((v['score'] != 0).sum()) * v['count']

            sorted_score = np.sort(v['score'])[:-v['minimal']]
            filtered_score = sorted_score[sorted_score != 0]
            for i in range(v['count']):
                filtered_score_list.append(filtered_score)

        scores = np.concatenate(filtered_score_list)
        threshold = np.sort(scores)[num]
        to_prune = int((scores <= threshold).sum())

        info = {'left': left_filters, 'to_prune': to_prune, 'total_pruned_ratio': (total_filters - left_filters + to_prune) / total_filters}
        return threshold, info

    def set_mask(self, status, threshold):
        for group_id, v in status.items():
            hard_threshold = float(np.sort(v['score'])[-v['minimal']])
            hard_mask = v['score'] >= hard_threshold
            soft_mask = v['score'] > threshold
            v['mask'] = (hard_mask + soft_mask)

        with torch.no_grad():
            for g in self.masks:
                if g.group_id in status:
                    mask = torch.from_numpy(status[g.group_id]['mask'].astype('float32')).to(g.device).view(1, -1, 1, 1)
                    g.bn_mask.set_(mask * g.bn_mask)

    def freeze_conv(self):
        self._status = {}
        for m in self.pack.net.modules():
            if isinstance(m, nn.Conv2d):
                for p in m.parameters():
                    self._status[id(p)] = p.requires_grad
                    p.requires_grad = False

    def restore_conv(self):
        for m in self.pack.net.modules():
            if isinstance(m, nn.Conv2d):
                for p in m.parameters():
                    p.requires_grad = self._status[id(p)]

    def tock(self, lr_min=0.001, lr_max=0.01, tock_epoch = 20, mute=False, acc_step=1):
        logs = []
        epoch = 0
        T = tock_epoch
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
            
            for g in self.pack.optimizer.param_groups:
                g['lr'] = max(_lr, 0)
                # g['momentum'] = 0.9
        
        for i in range(T):
            info = self.pack.trainer.train(self.pack, loss_hook = self.sparse_loss_hook, iter_hook = iter_hook, acc_step=acc_step)
            info.update(self.pack.trainer.test(self.pack))
            info.update({'LR': self.pack.optimizer.param_groups[0]['lr']})
            epoch += 1
            if not mute:
                print('Tock - %d,\t Test Loss: %.4f,\t Test Acc: %.2f, Final LR: %.5f' % (i, info['test_loss'], info['acc@1'], info['LR']))
            logs.append(info)
        return logs

    def tick(self, lr, test):
        ''' Do Prune '''
        self.freeze_conv()
        info = self.recover(lr, test)
        self.restore_conv()
        return info

    def prune(self, num, tick=False, lr=0.01, test=True):
        info = {}
        if tick:
            info = self.tick(lr, test)

            area = []
            for g in self.masks:
                area.append(int(g.area[0]))
            self.eta_scale_factor = min(area)

        status = {}
        for g in self.masks:
            if g.group_id in status:
                # assert the gbn in same group has the same channel size
                status[g.group_id]['score'] += g.get_score(self.flops_eta / self.eta_scale_factor).cpu().data.numpy()
                status[g.group_id]['count'] += 1
            else:
                status[g.group_id] = {
                    'score': g.get_score(self.flops_eta / self.eta_scale_factor).cpu().data.numpy(),
                    'minimal': g.minimal_filter,
                    'count': 1,
                    'mask': None
                }
        
        threshold, r = self.get_threshold(status, num)
        info.update(r)
        threshold = float(threshold)
        self.set_mask(status, threshold)
        if test:
            info.update({'after_prune_test_acc': self.pack.trainer.test(self.pack)['acc@1']})
        self.logs.append(info)
        self.pruned_filters = self.total_filters - info['left']
        info['total'] = self.total_filters
        return info
