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
from .model_config import DEBUG_QUANTIZE

#############################################################################################################
############################################# TDNN Layer ####################################################
#############################################################################################################
def conv_relu_bn(n_in, n_out, kernel_size=3, stride=1, padding=0, dilation=1, bias = True):
    """
    n_time_out = floor((n_time_in + 2*padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    """
    return nn.Sequential(
        nn.Conv1d(in_channels = n_in, out_channels = n_out, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = bias),
        nn.ReLU(),
        nn.BatchNorm1d(n_out, eps=1e-3, affine=True)       
    )
def conv_relu(n_in, n_out, kernel_size=3, stride=1, padding=0, dilation=1, bias = True):
    """
    n_time_out = floor((n_time_in + 2*padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    """
    return nn.Sequential(
        nn.Conv1d(in_channels = n_in, out_channels = n_out, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = bias),
        nn.ReLU()       
    )

class TDNNLayer(nn.Module):
    """
    n_time_out = floor((n_time_in + 2*padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    """
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=0, dilation=1, bias = True, batch_norm = True, binary = False, weight_init = None, bias_init = None, dropout = 0.0):
        super(TDNNLayer, self).__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.binary = binary
        self.input_size  = n_in
        self.output_size = n_out
        
        if self.binary:
            self.tdnn = conv_relu(n_in, n_out, kernel_size, stride, padding, dilation, bias)
        else:
            self.tdnn = conv_relu(n_in, n_out, kernel_size, stride, padding, dilation, bias)

        if weight_init is not None:
            self.tdnn.weight.data.copy_(weight_init)
        if bias_init is not None:
            self.tdnn.bias.data.copy_(bias_init)
        
        #self.batch_norm = nn.LayerNorm(n_out) if self.batch_norm else None
        self.batch_norm = nn.BatchNorm1d(n_out, eps=1e-3, affine=True) if self.batch_norm else None

        self.dropout = nn.Dropout(dropout) if self.dropout > 0.0 else None
        
        if DEBUG_QUANTIZE:
            self.bn_min        = 100000000
            self.bn_max        = -100000000
            self.bn_mean       = 0.0
            self.bn_std        = 0.0
            self.act_min       = 100000000
            self.act_max       = -100000000
            self.act_mean      = 0.0
            self.act_std       = 0.0

    def forward(self, x):
        # input: (num_block, num_frame, input_size)
        x = self.tdnn(x)
        
        if DEBUG_QUANTIZE:
            with torch.no_grad():
                self.act_min = min(x.min(), self.act_min)
                self.act_max = max(x.max(), self.act_max)
                self.act_mean = (x.mean() +  self.act_mean) / 2.0
                self.act_std = (x.std() +  self.act_std) / 2.0
        
        if self.batch_norm is not None:
            x = self.batch_norm(x)
            if DEBUG_QUANTIZE:
                with torch.no_grad():
                    self.bn_min  = min(x.min(), self.bn_min)
                    self.bn_max  = max(x.max(), self.bn_max)
                    self.bn_mean = (x.mean() + self.bn_mean) / 2.0
                    self.bn_std  = (x.std() + self.bn_std) / 2.0

        if self.dropout is not None:
            x = self.dropout(x)

        return x
        