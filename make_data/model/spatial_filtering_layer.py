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
from torch.nn.parameter import Parameter
from .functions import to_cuda
from .complex_conv_layer import CConv2dLayer
from .model_config import DEBUG_QUANTIZE

#############################################################################################################
##################################### Complex multi-channel filter Layer ####################################
#############################################################################################################

class Filter(nn.Module):
    r"""Applies a complex filter to the incoming data: :math:`y = sum(x .* A, dim = -1) + b`

    Args:
        num_bin: number of frequency bin
        num_channel: number of channels
        bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``
    """
    __constants__ = ['bias', 'num_bin', 'num_channel']

    def __init__(self, num_bin, num_channel, init_weight = None, init_bias = None, bias = True, fix = False):
        super(Filter, self).__init__()

        self.num_bin     = num_bin
        self.num_channel = num_channel
        self.intialized  = False
        self.fix         = fix
        
        self.weight = Parameter(torch.Tensor(1, 2, num_bin, num_channel), requires_grad = (not fix)) # (1, 2, num_bin, num_channel)
        if init_weight is not None:
            self.weight.data.copy_(init_weight)
            self.intialized = True
        if bias:
            self.bias = Parameter(torch.Tensor(1, 2, num_bin), requires_grad = (not fix))            # (1, 2, num_bin)
            if init_bias is not None:
                self.bias.data.copy_(init_bias)
        else:
            self.bias = None
            #self.register_parameter('bias', None)
        if init_weight is None:
            self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a = math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # input =  ( num_frame, 2, num_bin, num_channel )
        # weight = ( 1, 2, num_bin, num_channel )
        num_frame, num_dim, num_bin, num_channel = input.size()
        assert num_frame > 0 and num_bin == self.num_bin and num_channel == self.num_channel and num_dim == 2, "illegal shape for input, the required input shape is (%d, 2, %d, %d), but got (%d, %s, %d, %d)" % (num_frame, self.num_bin, self.num_channel, num_frame, num_dim, num_bin, num_channel)
        
        filter_out_r = self.weight[:,0,:,:] * input[:,0,:,:] - self.weight[:,1,:,:] * input[:,1,:,:] # (num_frame, num_bin, num_channel)
        filter_out_i = self.weight[:,0,:,:] * input[:,1,:,:] + self.weight[:,1,:,:] * input[:,0,:,:] # (num_frame, num_bin, num_channel)

        filter_out_r = filter_out_r.sum(dim = 2) # (num_frame, num_bin)
        filter_out_i = filter_out_i.sum(dim = 2) # (num_frame, num_bin)

        filter_out = torch.cat((torch.unsqueeze(filter_out_r, 1), torch.unsqueeze(filter_out_i, 1)), 1) # (num_frame, 2, num_bin)

        if self.bias is not None:
            filter_out = filter_out + self.bias.expand_as(filter_out) # 
        return filter_out # (num_frame, 2, num_bin)

    def extra_repr(self):
        return 'num_bin={}, num_channel={}, bias={}'.format(
            self.num_bin, self.num_channel, self.bias is not None
        )

class SFLayer(nn.Module):
    r"""Implement spatial filtering multi-channel audio to get beamformed output signals, all operators on complex fft coefficients of audios
    Args:
        num_beam: number of beamforming filters
        num_bin: number of frequency bin
        num_channel: number of channels
        bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``
    """
    def __init__(self, num_beam, num_bin, num_channel, bias = True, weight_init = None, bias_init = None, fix = False):
        super(SFLayer, self).__init__()
        
        self.num_beam    = num_beam
        self.num_bin     = num_bin
        self.num_channel = num_channel
        self.bias        = bias
        self.fix         = fix
        
        filters = []
        for n in range(num_beam):
            if weight_init is not None:
                init_weight = torch.from_numpy(weight_init[n][np.newaxis, :, :, :]) # (1, 2, num_bin, num_channel)
            else:
                init_weight = None
            if bias_init is not None:
                init_bias   = torch.from_numpy(bias_init[n])
            else:
                init_bias   = None
            filter = Filter(num_bin = self.num_bin, num_channel = self.num_channel, init_weight = init_weight, init_bias = init_bias, bias = self.bias, fix = self.fix)
            filters.append(filter)
        self.filter = nn.ModuleList(filters)
        
        if DEBUG_QUANTIZE:
            self.filter_min        = 100000000
            self.filter_max        = -100000000
            self.filter_mean       = 0.0
            self.filter_std        = 0.0

    def forward(self, input):
        # fft_input          = ( num_block * num_channel, num_bin * 2, num_frame)
        # fft_input          = ( num_block, num_channel, num_bin * 2, num_frame)

        # fft_input_r        = ( num_block, num_channel, num_bin, num_frame)
        # fft_input_i        = ( num_block, num_channel, num_bin, num_frame)

        # fft_input_r        = ( num_block, num_frame, num_bin, num_channel)
        # fft_input_i        = ( num_block, num_frame, num_bin, num_channel)

        # fft_input_r        = ( num_block * num_frame, num_bin, num_channel)
        # fft_input_i        = ( num_block * num_frame, num_bin, num_channel)
        
        # filter_input       = ( num_frame, 2, num_bin, num_channel )
        # filter_output      = ( num_frame, 2, num_bin, num_beam)

        # align_fft_input    = ( num_frame, 2, num_bin, num_beam)
        # align_fft_output   = ( num_frame, 2, num_bin)

        # phasen_fft_input   = ( num_block, 2, num_bin, num_frame )

        # input      = ( num_frame, 2, num_bin, num_channel )
        # output     = ( num_frame, 2, num_bin, num_beam )

        output_list = []
        for beamforming in self.filter:
            output = beamforming(input)     # (num_frame, 2, num_bin)
            output = output.unsqueeze(3)    # (num_frame, 2, num_bin, 1)
            output_list.append(output)      
        output = torch.cat(output_list, 3)

        if DEBUG_QUANTIZE:
            with torch.no_grad():
                self.filter_min = min(output.min(), self.filter_min)
                self.filter_max = max(output.max(), self.filter_max)
                self.filter_mean = (output.mean() +  self.filter_mean) / 2.0
                self.filter_std = (output.std() +  self.filter_std) / 2.0
        
        return output # (num_frame, 2, num_bin, num_beam)

class CConv2DSFLayer(nn.Module):
    r"""Implement spatial filtering multi-channel audio to get beamformed output signals, all operators on complex fft coefficients of audios
    Args:
        num_beam: number of beamforming filters
        num_bin: number of frequency bin
        num_channel: number of channels
        bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``
    """
    def __init__(self, num_beam, num_bin, num_channel, bias = True):
        super(CConv2DSFLayer, self).__init__()
        
        self.num_beam    = num_beam
        self.num_bin     = num_bin
        self.num_channel = num_channel
        self.bias        = bias
        
        kernel_size = [3, num_channel]
        stride      = [1, num_channel]
        padding     = [1, 0]
        self.cconv_filter = CConv2dLayer(in_channels = 1, out_channels = num_beam, kernel_size = kernel_size, stride = stride, padding = padding, dilation = 1, bias= bias)

    def forward(self, input_r, input_i):
        # input_r = ( num_block, 1, num_bin, num_frame * num_channel )
        # input_i = ( num_block, 1, num_bin, num_frame * num_channel )

        output_r, output_i = self.cconv_filter(input_r, input_i)

        return output_r, output_i # (num_block, num_beam, num_bin, num_frame)

