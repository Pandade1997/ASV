import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np
import math
import os
from torch.autograd import Variable
from collections import OrderedDict
from .base_model import supported_acts, supported_norms

class CConv2d(nn.Module):
    """
    n_time_out = floor((n_time_in + 2*padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True):
        super(CConv2d, self).__init__()
        
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation
        self.bias         = bias

        self.conv_r      = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups, bias = bias)
        self.conv_i      = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups, bias = bias)

    def forward(self, x_r, x_i):
        # input: (num_block, num_channe, height, width)
        # r = weight_r * x_r - weight_i * x_i
        # i = weight_r * x_i + weight_i * x_r
        o_r = self.conv_r(x_r) - self.conv_i(x_i)
        o_i = self.conv_r(x_i) + self.conv_i(x_r)
        return o_r, o_i 

class CConv2dLayer(nn.Module):
    """
    n_time_out = floor((n_time_in + 2*padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, batch_norm = None, act_type = None, dropout = 0.0):
        super(CConv2dLayer, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation
        self.bias         = bias

        if act_type is not None:
            self.act_type = act_type.lower()
            assert self.act_type in supported_acts, "act_type should be either relu, sigmoid, softmax or tanh"
            self.act_func = supported_acts[self.act_type]
        else:
            self.act_func = None

        self.cconv2d = CConv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups, bias = bias)

        #self.batch_norm = nn.LayerNorm(out_channels) if batch_norm else None
        self.batch_norm  = nn.BatchNorm2d(out_channels) if batch_norm else None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x_r, x_i, take_power = False):

        # input: (num_block, num_channe, height, width)
        # r = weight_r * x_r - weight_i * x_i
        # i = weight_r * x_i + weight_i * x_r
        
        o_r, o_i = self.cconv2d(x_r, x_i)
        
        if self.batch_norm is not None:
            o_r = self.batch_norm(o_r)
            o_i = self.batch_norm(o_i)
        
        if self.act_func is not None:
            o_r = self.act_func(o_r)
            o_i = self.act_func(o_i)

        if self.dropout is not None:
            o_r = self.dropout(o_r)
            o_i = self.dropout(o_i)

        if take_power:
            out = o_r ** 2 + o_i ** 2
            return out
        else:
            return o_r, o_i 
