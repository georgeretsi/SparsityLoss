'''
created on Dec 28, 2019
@author: georgeretsi
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SparseSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, w):
        w = F.hardshrink(w, 1)
        return w

    @staticmethod
    def backward(ctx, g):
        return g


class SparseConv(nn.Module):
    def __init__(self, conv, r=.5, trainable=False):
        super(SparseConv, self).__init__()
        self.conv = conv
        self.trainable = trainable
        self.r = nn.Parameter(torch.Tensor([r]))

    def forward(self, x):


        w = self.conv.weight
        b = self.conv.bias
        stride = self.conv.stride
        padding = self.conv.padding
        groups = self.conv.groups

        # impose zero mean
        self.conv.weight.data = self.conv.weight.data - self.conv.weight.data.mean()
        m = 0.0  # w.mean().detach()

        rr = torch.clamp(self.r, 0, 5)
        if self.trainable:
            a = rr
        else:
            a = rr.detach()

        l = a * (w-m).std().detach()
        nw = (w - m) / (l + 1e-6)
        sw = SparseSTE.apply(nw)

        w = m + (l + 1e-6) * sw
        out = F.conv2d(x, w, bias=b, padding=padding, stride=stride, groups=groups)

        return out


class SparseFc(nn.Module):
    def __init__(self, fc, r=.5, trainable=False):
        super(SparseFc, self).__init__()
        self.fc = fc
        self.trainable = trainable
        self.r = nn.Parameter(torch.Tensor([r]))

    def forward(self, x):

        w = self.fc.weight
        b = self.fc.bias

        # impose zero mean
        self.fc.weight.data = self.fc.weight.data - self.fc.weight.data.mean()
        m = 0.0  # w.mean().detach()

        rr = torch.clamp(self.r, 0, 5)
        if self.trainable:
            a = rr
        else:
            a = rr.detach()

        l = a * (w - m).std().detach()
        nw = (w - m) / (l + 1e-6)
        sw = SparseSTE.apply(nw)

        w = m + (l + 1e-6) * sw
        out = F.linear(x, w, bias=b)
        return out

def iter_sparsify(m, r, trainable=True, pthres=1000):
    for name, child in m.named_children():
        iter_sparsify(child, r, trainable, pthres)
        if type(child) == nn.Conv2d:
            if (child.in_channels * child.out_channels * child.kernel_size[0] * child.kernel_size[1]) >= pthres:
                slayer = SparseConv(child, r, trainable)
                m.__setattr__(name, slayer)
        if type(child) == nn.Linear:
            if (child.in_features * child.out_features) >= pthres:
                slayer = SparseFc(child, r, trainable)
                m.__setattr__(name, slayer)

def iter_desparsify(m):
    for name, child in m.named_children():
        #print(name)
        iter_desparsify(child)
        if type(child) == SparseConv:
            conv = child.conv
            w = conv.weight.data

            mean = 0.0 #w.mean()
            s = (w-mean).std()
            r = (s * torch.clamp(child.r, 0, 5)).item()
            w = F.hardshrink(w-mean, r + 1e-6)
            conv.weight.data = w

            m.__setattr__(name, conv)

        if type(child) == SparseFc:
            fc= child.fc
            w = fc.weight.data

            mean = 0.0  # w.mean()
            s = (w - mean).std()
            r = (s * torch.clamp(child.r, 0, 5)).item()
            w = F.hardshrink(w - mean, r + 1e-6)
            fc.weight.data = w

            m.__setattr__(name, fc)

def sparsity(model, print_per_layer=False):
    zeros_cnt = 0
    cnt = 0
    for name, layer in model.named_modules():
        if ('Sparse' in layer.__class__.__name__):
            if 'Conv' in layer.__class__.__name__ :
                w = layer.conv.weight
            elif 'Fc' in layer.__class__.__name__:
                w = layer.fc.weight
            else:
                print(" Not Recognized Sparse Module ")

            m = 0.0 #w.mean()
            a = torch.clamp(layer.r, 0, 5)
            l = a * (w-m).std()
            nw = F.hardshrink((w - m) / (l+1e-6), 1)
            tsparsity = (nw == 0).float().sum().item()
            tnum = nw.numel()
            zeros_cnt += tsparsity
            cnt += tnum

            if print_per_layer:
                print("{} sparsity: {}%".format(name, round(100.0 * tsparsity / tnum, 2)))

    return 100 * float(zeros_cnt) / float(cnt), zeros_cnt

def change_sparsity(model, r):
    for name, layer in model.named_modules():
        if ('Sparse' in layer.__class__.__name__):
            layer.r = r

def adaptive_loss(model, reduce=True):
    eloss = []
    nweights = []
    for name, layer in model.named_modules():
        if ('Sparse' in layer.__class__.__name__):
            if 'Conv' in layer.__class__.__name__ :
                nw = layer.conv.weight.numel()
            if 'Fc' in layer.__class__.__name__:
                nw = layer.fc.weight.numel()
            eloss += [1 - torch.erf(layer.r / math.sqrt(2))]
            nweights += [nw]

    if reduce:
        eloss = sum([e * n for e, n in zip(eloss, nweights)]) / sum(nweights)
    else:
        eloss = torch.cat(eloss)

    return eloss

