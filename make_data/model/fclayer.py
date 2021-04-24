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
class BinaryLinear(nn.Linear):
    def forward(self, input):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = binarize(self.weight.org)

        output = nn.functional.linear(input, self.weight)

        if self.bias is not None:
            self.bias.org = self.bias.data.clone()
            output += self.bias.view(1, -1).expand_as(output)

        return output

#############################################################################################################
############################################# Deep FeedForward Layer ########################################
#############################################################################################################
class FCLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        act_func = nn.ReLU,
        batch_norm = False,
        dropout = 0.0,
        bias = True,
        binary = False,
        weight_init = None,
        bias_init = None):
        
        super(FCLayer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.binary = binary

        if self.binary:
            self.fc = BinaryLinear(input_size, output_size, bias = bias)
        else:
            self.fc = nn.Linear(input_size, output_size, bias = bias)

        if weight_init is not None:
            self.fc.weight.data.copy_(weight_init)

        if bias_init is not None:
            self.fc.bias.data.copy_(bias_init)

        #self.batch_norm = nn.LayerNorm(output_size) if batch_norm else None
        self.batch_norm = nn.BatchNorm1d(output_size) if batch_norm else None
        self.act_func = act_func
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.bias = bias

        if DEBUG_QUANTIZE:
            self.fc_min        = 100000000
            self.fc_max        = -100000000
            self.fc_mean       = 0.0
            self.fc_std        = 0.0
            self.bn_min        = 100000000
            self.bn_max        = -100000000
            self.bn_mean       = 0.0
            self.bn_std        = 0.0
            self.act_min       = 100000000
            self.act_max       = -100000000
            self.act_mean      = 0.0
            self.act_std       = 0.0

    def forward(self, x):
        if len(x.size()) > 2:
            t, n, z = x.size(0), x.size(1), x.size(2)   # (num_block, num_frame, input_size)
            if not x.is_contiguous():
                x = x.contiguous()
            x = x.view(t * n, -1)                       # (num_block * num_frame, input_size)

        x = self.fc(x)

        if DEBUG_QUANTIZE:
            with torch.no_grad():
                self.fc_min  = min(x.min(), self.fc_min)
                self.fc_max  = max(x.max(), self.fc_max)
                self.fc_mean = (x.mean() +  self.fc_mean) / 2.0
                self.fc_std  = (x.std() +  self.fc_std) / 2.0

        if self.batch_norm is not None:
            x = self.batch_norm(x)

            if DEBUG_QUANTIZE:
                with torch.no_grad():
                    self.bn_min  = min(x.min(), self.bn_min)
                    self.bn_max  = max(x.max(), self.bn_max)
                    self.bn_mean = (x.mean() + self.bn_mean) / 2.0
                    self.bn_std  = (x.std() + self.bn_std) / 2.0

        if self.act_func is not None:
            x = self.act_func(x)

        if DEBUG_QUANTIZE:
            with torch.no_grad():
                self.act_min = min(x.min(), self.act_min)
                self.act_max = max(x.max(), self.act_max)
                self.act_mean = (x.mean() +  self.act_mean) / 2.0
                self.act_std = (x.std() +  self.act_std) / 2.0

        if self.dropout is not None:
            x = self.dropout(x)

        return x