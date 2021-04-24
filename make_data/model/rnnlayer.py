import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import functools
import numpy as np
import math
from torch.autograd import Variable
from collections import OrderedDict
from .functions import binarize
from .model_config import DEBUG_QUANTIZE

#############################################################################################################
################################################ Binary Linear ##############################################
#############################################################################################################
class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim N*T*D to (N*T)*D, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t = 0
        n = 0
        z = 0
        dim = len(x.size())
        if dim > 2:
            n, t, z = x.size(0), x.size(1), x.size(2)   # (batch, seq_len, input_size)
            if not x.is_contiguous():
                x = x.contiguous()
            x = x.view(n * t, -1)          # (batch * seq_len, input_size)

        x = self.module(x)

        if dim > 2:
            x = x.view(t, n, -1) # (batch, seq_len, input_size)

        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class RNNLayer(nn.Module):
    def __init__(self, input_size, output_size, rnn_type = nn.LSTM, bidirectional = False, batch_norm = True, bias = True, dropout = 0.0):
        super(RNNLayer, self).__init__()
        self.input_size     = input_size
        self.output_size    = output_size
        self.bidirectional  = bidirectional
        self.dropout        = dropout
        self.batch_norm     = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn            = rnn_type(input_size = input_size, hidden_size = output_size, bias = bias, batch_first = True, dropout = dropout, bidirectional = bidirectional)
        self.num_directions = 2 if bidirectional else 1

        if DEBUG_QUANTIZE:
            self.bn_min       = 100000000
            self.bn_max       = -100000000
            self.bn_mean      = 0.0
            self.bn_std       = 0.0

            self.act_min       = 100000000
            self.act_max       = -100000000
            self.act_mean      = 0.0
            self.act_std       = 0.0

    def forward(self, x):
        # x: (N, T, in_size)
        # y: (N, T, out_size)
        if self.batch_norm is not None:
            x = self.batch_norm(x)

        if DEBUG_QUANTIZE:
            with torch.no_grad():
                self.bn_min  = min(x.min(), self.bn_min)
                self.bn_max  = max(x.max(), self.bn_max)
                self.bn_mean = (x.mean() + self.bn_mean) / 2.0
                self.bn_std  = (x.std() + self.bn_std) / 2.0

        x, _ = self.rnn(x)
        if DEBUG_QUANTIZE:
            with torch.no_grad():
                self.act_min = min(x.min(), self.act_min)
                self.act_max = max(x.max(), self.act_max)
                self.act_mean = (x.mean() +  self.act_mean) / 2.0
                self.act_std = (x.std() +  self.act_std) / 2.0

        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (NxTxH*2) -> (NxTxH) by sum
        self.rnn.flatten_parameters()
        return x