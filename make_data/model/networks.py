import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import math

from collections import OrderedDict

from .base_model import clones, supported_rnns, supported_acts, supported_loss, init_net, get_non_pad_mask, get_attn_pad_mask
from .fbanklayer import KaldiFbankModel, FbankModel
from .tdnnlayer import TDNNLayer
from .fclayer import FCLayer
from .rnnlayer import RNNLayer
from .transformerlayer import TransformerLayer, TransformerFFLayer

from .spatial_filtering_layer import SFLayer, CConv2DSFLayer
from .frequency_align_layer import Conv2DFALayer, FCFALayer, DeepConv2DFALayer
from .dcf import DCF
from .gsc import GSC
from .dpr_dsnr import DPR_DSNR
from .ipd import IPD
from .df import DF
from .functions import to_cuda

from .regularization import Regularization

from .model_config import DEBUG_QUANTIZE

class InferenceBatchLogSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            batch_size = input_.size()[0]
            return torch.stack([torch.log_softmax(input_[i]) for i in range(batch_size)], 0)
        else:
            return input_

class InferenceBatchSigmoid(nn.Module):
    def forward(self, input_):
        batch_size = input_.size()[0]
        return torch.stack([torch.sigmoid(input_[i]) for i in range(batch_size)], 0)

class InferenceBatchRelu(nn.Module):
    def forward(self, input_):
        batch_size = input_.size()[0]
        return torch.stack([torch.relu(input_[i]) for i in range(batch_size)], 0)

class DeepFFNet(nn.Module):
    "Define deep forward network to extract deep feature or solve classification or regression problem ."
    def __init__(
        self,
        layer_size,
        binary = False,
        hidden_act_type='relu',
        out_act_type = None,
        batch_norm = False,
        dropout = 0.0,
        bias = True):

        super(DeepFFNet, self).__init__()

        self.layer_size  = layer_size
        self.input_size  = layer_size[0]
        self.output_size = layer_size[-1]
        self.num_layer   = len(layer_size)
        self.batch_norm  = batch_norm
        self.bias        = bias
        self.dropout     = dropout
        self.binary      = binary

        if hidden_act_type is not None:
            self.hidden_act_type = hidden_act_type.lower()
            assert self.hidden_act_type in supported_acts, "act_type should be either relu, sigmoid, softmax or tanh"
            self.hidden_act = supported_acts[self.hidden_act_type]
        else:
            self.hidden_act = None

        if out_act_type is not None:
            self.out_act_type = out_act_type.lower()
            assert self.out_act_type in supported_acts, "act_type should be either relu, sigmoid, softmax or tanh"
            self.out_act = supported_acts[self.out_act_type]
        else:
            self.out_act = None

        layers = []

        for i in range(0, self.num_layer - 2):
            fc = FCLayer(
                    layer_size[i],
                    layer_size[i + 1],
                    act_func = self.hidden_act,
                    batch_norm = self.batch_norm,
                    dropout = self.dropout,
                    bias = self.bias,
                    binary = self.binary)
            layers.append(fc)

        fc = FCLayer(
                layer_size[-2],
                layer_size[-1],
                act_func = self.out_act,
                batch_norm = self.batch_norm,
                dropout = self.dropout,
                bias = self.bias,
                binary = self.binary)
        layers.append(fc)
        
        if DEBUG_QUANTIZE:
            self.input_min        = 100000000
            self.input_max        = -100000000
            self.input_mean       = 0.0
            self.input_std        = 0.0
        
        self.NNet = nn.Sequential(*layers)
        self.num_layer   = len(self.NNet)

    def forward(self, input):
        if DEBUG_QUANTIZE:
            with torch.no_grad():
                self.input_min  = min(input.min(), self.input_min)
                self.input_max  = max(input.max(), self.input_max)
                self.input_mean = (input.mean() +  self.input_mean) / 2.0
                self.input_std  = (input.std() +  self.input_std) / 2.0
    
        y = self.NNet(input)
        if len(input.size()) > 2:   # input: (num_block, num_frame, input_size)
            if not y.is_contiguous():
                y = y.contiguous()
            y = y.view(input.size()[0], input.size()[1], -1)
        return y

class Outputer(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(
        self,
        in_size,
        out_size,
        binary = False,
        out_act_type = None,
        batch_norm = False,
        bias = False,
        vectorized_type = None):
        super(Outputer, self).__init__()

        self.binary = binary
        self.bias = bias
        self.batch_norm = batch_norm
        self.vectorized_type = vectorized_type
        self.input_size  = in_size
        self.output_size = out_size

        if out_act_type is not None:
            self.out_act_type = out_act_type.lower()
            assert self.out_act_type in supported_acts, "act_type should be either relu, sigmoid, softmax or tanh"
            self.out_act = supported_acts[self.out_act_type]
        else:
            self.out_act = None

        layers = []
        fc = FCLayer(
                input_size = in_size,
                output_size = out_size,
                act_func = self.out_act,
                batch_norm = self.batch_norm,
                dropout = 0.0,
                bias = self.bias,
                binary = self.binary)
        layers.append(fc)
        
        if DEBUG_QUANTIZE:
            self.input_min        = 100000000
            self.input_max        = -100000000
            self.input_mean       = 0.0
            self.input_std        = 0.0
            
            self.vectorized_min        = 100000000
            self.vectorized_max        = -100000000
            self.vectorized_mean       = 0.0
            self.vectorized_std        = 0.0
        
        self.NNet = nn.Sequential(*layers)
        self.num_layer   = len(self.NNet)

    def forward(self, x):
    
        if DEBUG_QUANTIZE:
            with torch.no_grad():
                self.input_min  = min(x.min(), self.input_min)
                self.input_max  = max(x.max(), self.input_max)
                self.input_mean = (x.mean() +  self.input_mean) / 2.0
                self.input_std  = (x.std() +  self.input_std) / 2.0
        
        # x: (num_block, num_frame, output_size)
        if self.vectorized_type is not None:
            if self.vectorized_type.lower() == 'concat': 
                x = x.contiguous().view(x.size()[0], -1)
            elif self.vectorized_type.lower() == 'gap':
                x = torch.mean(x, 1)
                
            if DEBUG_QUANTIZE:
                with torch.no_grad():
                    self.vectorized_min  = min(x.min(), self.vectorized_min)
                    self.vectorized_max  = max(x.max(), self.vectorized_max)
                    self.vectorized_mean = (x.mean() +  self.vectorized_mean) / 2.0
                    self.vectorized_std  = (x.std() +  self.vectorized_std) / 2.0
            
        y = self.NNet(x)        # (num_block, num_frame, output_size)
        if len(x.size()) > 2:   # input: (num_block, num_frame, input_size)
            if not y.is_contiguous():
                y = y.contiguous()
            y = y.view(x.size(0), x.size(1), -1)
        return y

class TDNNNet(nn.Module):
    def __init__(self, layers_config, binary = False):
        """
        layers_config is a list that contains the information of each layer
        layers_config[:]
            in_size: 128
            out_size: 128
            kernel_size: 3
            stride: 1
            padding: 0
            dilation: 1
            batch_norm: False
            bias: True
        """
        super().__init__()
        
        self.layers_config = layers_config
        self.num_layer     = len(layers_config)
        self.binary        = binary
        self.input_size  = self.layers_config[0]['in_size']
        self.output_size = self.layers_config[-1]['out_size']
        
        tdnn_layers = []
        for i in range(0, self.num_layer):
            in_size     = self.layers_config[i]['in_size']
            out_size    = self.layers_config[i]['out_size']
            kernel_size = self.layers_config[i]['kernel_size']
            stride      = self.layers_config[i]['stride']
            padding     = self.layers_config[i]['padding']
            dilation    = self.layers_config[i]['dilation']
            batch_norm  = self.layers_config[i]['batch_norm']
            bias        = self.layers_config[i]['bias']
            
            tdnn_layers.append(TDNNLayer(n_in = in_size, n_out = out_size, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = bias, batch_norm = batch_norm))
        
        if DEBUG_QUANTIZE:
            self.input_min        = 100000000
            self.input_max        = -100000000
            self.input_mean       = 0.0
            self.input_std        = 0.0
        
        self.NNet = nn.Sequential(*tdnn_layers)
        self.num_layer   = len(self.NNet)
        
    def forward(self, input):
        if DEBUG_QUANTIZE:
            with torch.no_grad():
                self.input_min  = min(input.min(), self.input_min)
                self.input_max  = max(input.max(), self.input_max)
                self.input_mean = (input.mean() +  self.input_mean) / 2.0
                self.input_std  = (input.std() +  self.input_std) / 2.0
    
        # input: (num_block, num_frame, input_size)
        input = input.permute([0, 2, 1]) # input: (num_block, input_size, num_frame)
        y = self.NNet(input)   # y: (num_block, output_size, num_frame)
        y = y.permute([0, 2, 1]) # y: (num_block, num_frame, output_size)
        return y

class DeepRNNNet(nn.Module):
    "Define deep rnn network to extract deep feature or solve classification or regression problem ."
    def __init__(self, layer_size, rnn_type = 'gru', bidirectional = False, batch_norm = False, bias = True, dropout = 0.0):
        super(DeepRNNNet, self).__init__()

        self.layer_size    = layer_size
        self.input_size    = layer_size[0]
        self.output_size   = layer_size[-1]
        self.num_layer     = len(layer_size) - 1
        self.batch_norm    = batch_norm
        self.bias          = bias
        self.dropout       = dropout
        self.bidirectional = bidirectional
        self.rnn_type      = rnn_type

        if rnn_type is not None:
            rnn_type = rnn_type.lower()
            assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn, gru"
            self.rnn_type = supported_rnns[rnn_type]
        else:
            self.rnn_type = nn.GRU

        # input_size, output_size, rnn_type = nn.LSTM, bidirectional = False, batch_norm = True, bias = True, dropout = 0.0
        layers = []
        for i in range(0, self.num_layer):
            rnn = RNNLayer(input_size = layer_size[i], 
                            output_size = layer_size[i + 1], 
                            rnn_type =self.rnn_type, 
                            bidirectional = self.bidirectional, 
                            batch_norm = self.batch_norm, 
                            bias = self.bias, 
                            dropout = self.dropout)
            layers.append(rnn)

        if DEBUG_QUANTIZE:
            self.input_min        = 100000000
            self.input_max        = -100000000
            self.input_mean       = 0.0
            self.input_std        = 0.0
        
        self.NNet = nn.Sequential(*layers)
        self.num_layer   = len(self.NNet)

    def forward(self, input):
        if DEBUG_QUANTIZE:
            with torch.no_grad():
                self.input_min  = min(input.min(), self.input_min)
                self.input_max  = max(input.max(), self.input_max)
                self.input_mean = (input.mean() +  self.input_mean) / 2.0
                self.input_std  = (input.std() +  self.input_std) / 2.0

        # input: (num_block, num_frame, input_size)
        y = self.NNet(input)
        return y   # (num_block, num_frame, output_size)


class MultiEnhNet(nn.Module):
    def __init__(self, spatial_filter_config, frequency_align_config):
        r"""
        Implement multi-channel filter by spatial filtering layer and frequency align layer, 
        the input is complex FFT coefficients of multi-channel audios with shape [num_frame, 2, num_bin, num_channel], 
        and the output is complex FFT coefficients of the enhanced speech with shape [num_frame, 2, num_bin]

        spatial_filter_config:
            num_bin: number of frequency bin
            num_channel: number of channels
            num_beam: number of beamforming filters
            bias: If set to False, the layer will not learn an additive bias. Default: True
            fixed: If set to True, the spatial_filter layer will not be learn and updated. Default: True
            weight_init: None
            bias_init: None
            regularization_weight: regularization weight of spatial filter layer, Default: 0.1
            regularization_p: p-norm of spatial filter layer, Default: 1.0 or 2.0

        frequency_align_config:
            type: type of frequency alignment layer, fc or conv
            num_align: number of frequency alignment ways
            num_bin: number of frequency bin
            num_beam: number of beamforming filters
            pooling_type: type of alignment pooling, agv or max
            num_freq_kernel: if the type of frequency alignment layer is conv, num_freq_kernel MUST be given, it means that the size along frequency axis
            bias: If set to False, the layer will not learn an additive bias. Default: True
            batch_norm: Default: False
            dropout: rate of dropout, Default 0.0
            fixed: If set to True, the spatial_filter layer will not be learn and updated. Default: True
            weight_init: None
            bias_init: None
            regularization_weight: regularization weight of spatial filter layer, Default: 0.1
            regularization_p: p-norm of spatial filter layer, Default: 1.0 or 2.0
        """
        super(MultiEnhNet, self).__init__()
        
        self.spatial_filter_config  = spatial_filter_config
        self.frequency_align_config = frequency_align_config

        ## construct spatial_filter
        num_bin     = spatial_filter_config['num_bin']
        num_channel = spatial_filter_config['num_channel']
        num_beam    = spatial_filter_config['num_beam']
        bias        = spatial_filter_config['bias']
        weight_init = spatial_filter_config['weight_init']
        bias_init   = spatial_filter_config['bias_init']
        self.spatial_filter = SFLayer(num_beam = num_beam, num_bin = num_bin, num_channel = num_channel, bias = bias, weight_init = weight_init, bias_init = bias_init)

        regularization_weight = spatial_filter_config['regularization_weight']
        regularization_p      = spatial_filter_config['regularization_p']
        fixed                 = spatial_filter_config['fixed']
        if regularization_weight > 0.0 and not fixed:
            self.spatial_filter_regularization = Regularization(weight_decay = regularization_weight, p = regularization_p)
        else:
            self.spatial_filter_regularization = None

        if frequency_align_config['type'].lower() == 'fc':
            num_align    = frequency_align_config['num_align']
            num_bin      = frequency_align_config['num_bin']
            num_beam     = frequency_align_config['num_beam']
            pooling_type = frequency_align_config['pooling_type']
            bias         = frequency_align_config['bias']
            batch_norm   = frequency_align_config['batch_norm']
            dropout      = frequency_align_config['dropout']
            weight_init  = frequency_align_config['weight_init']
            bias_init    = frequency_align_config['bias_init']

            self.frequency_align = FCFALayer(num_beam = num_beam, num_align = num_align, num_bin = num_bin, bias = bias, pooling_type = pooling_type, batch_norm = batch_norm, dropout = dropout, weight_init = weight_init, bias_init = bias_init)
        elif frequency_align_config['type'].lower() == 'conv':
            num_align           = frequency_align_config['num_align']
            num_bin             = frequency_align_config['num_bin']
            num_freq_kernel     = frequency_align_config['num_freq_kernel']
            num_beam_kernel     = frequency_align_config['num_beam']
            pooling_type        = frequency_align_config['pooling_type']
            bias                = frequency_align_config['bias']
            batch_norm          = frequency_align_config['batch_norm']
            dropout             = frequency_align_config['dropout']
            weight_init         = frequency_align_config['weight_init']
            bias_init           = frequency_align_config['bias_init']

            self.frequency_align = Conv2DFALayer(num_freq_kernel = num_freq_kernel, num_beam_kernel = num_beam_kernel, num_align = num_align, num_bin = num_bin, bias = bias, pooling_type = pooling_type, batch_norm = batch_norm, dropout = dropout, weight_init = weight_init, bias_init = bias_init)
        else:
            num_align           = frequency_align_config['num_align']
            num_bin             = frequency_align_config['num_bin']
            num_freq_kernel     = frequency_align_config['num_freq_kernel']
            num_beam_kernel     = frequency_align_config['num_beam']
            pooling_type        = frequency_align_config['pooling_type']
            bias                = frequency_align_config['bias']
            batch_norm          = frequency_align_config['batch_norm']
            dropout             = frequency_align_config['dropout']
            self.frequency_align = DeepConv2DFALayer(num_freq_kernel = num_freq_kernel, num_beam_kernel = num_beam_kernel, num_align = num_align, num_bin = num_bin, bias = bias, pooling_type = pooling_type, batch_norm = batch_norm, dropout = dropout)

        regularization_weight = frequency_align_config['regularization_weight']
        regularization_p      = frequency_align_config['regularization_p']
        fixed                 = frequency_align_config['fixed']
        if regularization_weight > 0.0 and not fixed:
            self.frequency_align_regularization = Regularization(weight_decay = regularization_weight, p = regularization_p)
        else:
            self.frequency_align_regularization = None

        if DEBUG_QUANTIZE:
            self.input_min        = 100000000
            self.input_max        = -100000000
            self.input_mean       = 0.0
            self.input_std        = 0.0
    
    def forward(self, input, guid_info = None, num_block = 1):
        # input: ( num_frame, 2, num_bin, num_channel )
        if DEBUG_QUANTIZE:
            with torch.no_grad():
                self.input_min  = min(input.min(), self.input_min)
                self.input_max  = max(input.max(), self.input_max)
                self.input_mean = (input.mean() +  self.input_mean) / 2.0
                self.input_std  = (input.std() +  self.input_std) / 2.0

        if self.spatial_filter_config['fixed']:
            with torch.no_grad():
                spatial_filter_out = self.spatial_filter(input) # (num_frame, 2, num_bin, num_beam)
        else:
            spatial_filter_out = self.spatial_filter(input)     # (num_frame, 2, num_bin, num_beam)
        
        if self.frequency_align_config['fixed']:
            with torch.no_grad():
                frequency_align_out = self.frequency_align(spatial_filter_out, guid_info = guid_info, num_block = num_block)  # ( num_frame, 2, num_bin )
        else:
            frequency_align_out = self.frequency_align(spatial_filter_out, guid_info = guid_info, num_block = num_block)      # ( num_frame, 2, num_bin )
        
        return frequency_align_out # ( num_frame, 2, num_bin )
    
    def get_trainable_params(self, print_info = False):
        if print_info:
            print("####### Trainable Parames of MultiEnhNet")
        weights, biases = [], []
        if not self.spatial_filter_config['fixed']:
            if print_info:
                print("spatial_filter = ")
            spatial_filter_param_names = []
            for name, param in self.spatial_filter.named_parameters():
                spatial_filter_param_names.append(name)
                if 'bias' in name:
                    biases += [param]
                else:
                    weights += [param]
            if print_info:
                print(spatial_filter_param_names)
        
        if not self.frequency_align_config['fixed']:
            if print_info:
                print("frequency_align = ")
            frequency_align_param_names = []
            for name, param in self.frequency_align.named_parameters():
                frequency_align_param_names.append(name)
                if 'bias' in name:
                    biases += [param]
                else:
                    weights += [param]
            if print_info:
                print(frequency_align_param_names)
            
        if len(weights) < 1:
            return None
        params = [{'params': weights, }, {'params': biases, }]
        return params
    
    def get_regularization_loss(self):
        
        reg_loss = 0.0

        if self.spatial_filter_regularization is not None:
            reg_loss = reg_loss + self.spatial_filter_regularization(self.spatial_filter)
        
        if self.frequency_align_regularization is not None:
            reg_loss = reg_loss + self.frequency_align_regularization(self.frequency_align)

        return reg_loss

class DCFGSCEnhNet(nn.Module):
    def __init__(self, dcf_config, gsc_config, net_config):
        r"""
        Implement multi-channel enhancor by dcf layer and gsc layer, 
        the input is complex FFT coefficients of multi-channel audios with shape [num_frame, 2, num_bin, num_channel], 
        and the output is complex FFT coefficients of the enhanced speech with shape [num_frame, 2, num_bin]

        dcf_config:
            num_bin: number of frequency bin
            num_channel: number of channels
            num_beam: number of beamformer for enhance directions
            num_null: number of beamformer for null directions
            targ_bf_weight: beamformer filter coefficients for enhance directions
            null_bf_weight: beamformer filter coefficients for null directions
            fix: If set to True, the beamformer layer will not be learn and updated. Default: True
            alpha: smooth factor of dcf 
        gsc_config:
            num_bin: number of frequency bin
            num_channel: number of channels
            num_beam: number of beamformer for enhance directions
            num_null: number of beamformer for null directions
            targ_bf_weight: beamformer filter coefficients for enhance directions
            block_bf_weight: beamformer filter coefficients for bolck directions
            fix: If set to True, the beamformer layer will not be learn and updated. Default: True
            alpha_v: smooth factor of gsc 

        net_config:
            rnn_layer_size: size of rnn hidden layer, Default: [40, 128, 128, 128, 257]
            rnn_type: type of rnn, gru | lstm | rnn | None
            bidirectional: If set to True, RNN is bidirectional, otherwise False, Default: False

            cnn_layer_size: size of cnn hidden layer, Default: [40, 128, 128, 128, 257]
            cnn_type: type of cnn | resnet | u-net | dense-net | None
            
            mask_layer_size: size of mask net
            mask_act_type: type of mask activation, None|relu|sigmoid|softmax|tanh|softplus|prelu|leakyrelu

            batch_norm: Default: False
            bias: Default: True
            dropout: Default: 0.0
        """
        super(DCFGSCEnhNet, self).__init__()
        
        self.dcf_config = dcf_config
        self.gsc_config = gsc_config
        self.net_config = net_config

        ## construct dcf
        num_bin        = dcf_config['num_bin']
        num_channel    = dcf_config['num_channel']
        dcf_num_beam   = dcf_config['num_beam']
        dcf_num_null   = dcf_config['num_null']
        targ_bf_weight = dcf_config['targ_bf_weight']
        null_bf_weight = dcf_config['null_bf_weight']
        fix            = dcf_config['fix']
        self.dcf = DCF(num_beam = dcf_num_beam, num_null = dcf_num_null, num_bin = num_bin, num_channel = num_channel, targ_bf_weight = targ_bf_weight, null_bf_weight = null_bf_weight, fix = fix)

        ## construct gsc
        num_bin         = gsc_config['num_bin']
        num_channel     = gsc_config['num_channel']
        gsc_num_beam    = gsc_config['num_beam']
        gsc_num_null    = gsc_config['num_null']
        targ_bf_weight  = gsc_config['targ_bf_weight']
        block_bf_weight = gsc_config['block_bf_weight']
        fix             = gsc_config['fix']
        self.gsc = GSC(num_beam = gsc_num_beam, num_null = gsc_num_null, num_bin = num_bin, num_channel = num_channel, targ_bf_weight = targ_bf_weight, block_bf_weight = block_bf_weight, fix = fix)

        self.spect_feat_size = net_config['spect_feat_size']
        rnn_input_size  = dcf_num_null * num_bin + self.spect_feat_size
        mask_input_size = dcf_num_null * num_bin + self.spect_feat_size
        ## define and initialize the cnnE
        if net_config['cnn_type'] is not None:
            '''
            layer_size  = [dcf_config['num_null']]
            layer_size  = layer_size + net_config['cnn_layer_size']
            rnn_type    = net_config['cnn_type']
            batch_norm  = net_config['batch_norm']
            bias        = net_config['bias']
            dropout     = net_config['dropout']
            self.cnnE   = DeepRNNNet(layer_size = layer_size, rnn_type = rnn_type, batch_norm = batch_norm, bias = bias, dropout = dropout)
            cnn_input       = to_cuda(self, torch.randn(1, dcf_num_null, 1, num_bin)) # (num_block, num_null, num_frame, num_bin)
            cnn_output      = self.cnnE(cnn_input)                                    # (num_block, out_channel, num_frame, num_dim)
            rnn_input_size  = cnn_output.size(-1) + self.spect_feat_size
            mask_input_size = cnn_output.size(-1) + self.spect_feat_size
            '''
            self.cnnE       = None
        else:
            self.cnnE       = None
        
        ## define and initialize the rnnE
        if net_config['rnn_type'] is not None:
            layer_size      = [rnn_input_size]
            layer_size      = layer_size + net_config['rnn_layer_size']
            rnn_type        = net_config['rnn_type']
            bidirectional   = net_config['bidirectional']
            batch_norm      = net_config['batch_norm']
            bias            = net_config['bias']
            dropout         = net_config['dropout']
            self.rnnE       = DeepRNNNet(layer_size = layer_size, rnn_type = rnn_type, bidirectional = bidirectional, batch_norm = batch_norm, bias = bias, dropout = dropout)
            mask_input_size = layer_size[-1]
        else:
            self.rnnE       = None

        ## define and initialize the mask_prj
        layer_size    = [mask_input_size]
        layer_size    = layer_size + net_config['mask_layer_size']
        mask_act_type = net_config['mask_act_type']
        bias          = net_config['bias']
        dropout       = net_config['dropout']
        self.mask = DeepFFNet(layer_size = layer_size, hidden_act_type = 'relu', out_act_type = mask_act_type, batch_norm = False, dropout = dropout, bias = bias, binary = False)

    def dcf_enhance(self, mFFT, tdoa):
        with torch.no_grad():
            dcf_mask, dcf_targ_bf_out = self.dcf(input = mFFT, beam_id = tdoa, alpha = self.dcf_config['alpha']) # ( num_block, num_frame, num_bin, num_null ), ( num_block, num_frame, 2, num_bin )

            return dcf_targ_bf_out # ( num_block, num_frame, 2, num_bin )

    def forward(self, mFFT, tdoa, cmvn = None, fbank_extractor = None, DCF_Ratio = 0.5):
        # mFFT:         ( num_block * num_frame, 2, num_bin, num_channel )
        # spect_feats:  ( num_block, num_frame, feat_size )
        feats, ehFFT = self.dcf(input = mFFT, beam_id = tdoa, alpha = self.dcf_config['alpha']) # ( num_block, num_frame, num_bin, num_null ), ( num_block, num_frame, 2, num_bin )
        
        if self.spect_feat_size > 0:
            spect_feats   = torch.unsqueeze(ehFFT[:, :, 0, :] ** 2 + ehFFT[:, :, 1, :] ** 2, 1) # ( num_block, 1, num_frame, num_bin )
            spect_feats   = torch.clamp(spect_feats, min=1.0e-13)                               # ( num_block, 1, num_frame, num_bin )

            if fbank_extractor is not None:
                #print("using fbank spect_feats")
                spect_feats = fbank_extractor(torch.sqrt(spect_feats))  # ( num_block, 1, num_frame, spect_feat_size )
            else:
                #print("using log spect_feats")
                spect_feats = torch.log(spect_feats)                    # ( num_block, 1, num_frame, spect_feat_size )
            
            spect_feats = spect_feats.squeeze(1)                        # ( num_block, num_frame, spect_feat_size )

            # apply cmvn to mFFT
            if cmvn is not None:
                add_shift = cmvn[0, :].squeeze()                # (spect_feat_size)
                add_shift = add_shift.unsqueeze(0).unsqueeze(0) # (1, 1, spect_feat_size)
                add_shift = add_shift.expand_as(spect_feats)    # (num_block, num_frame, spect_feat_size)
    
                add_scale = cmvn[1, :].squeeze()                # (spect_feat_size)
                add_scale = add_scale.unsqueeze(0).unsqueeze(0) # (1, 1, spect_feat_size)
                add_scale = add_scale.expand_as(spect_feats)    # (num_block, num_frame, spect_feat_size)
    
                spect_feats = ( spect_feats + add_shift ) * add_scale # (num_block, num_frame, spect_feat_size)
        else:
            spect_feats = None

        if self.cnnE is not None:
            feats = feats.permute([0, 3, 1, 2]).contiguous() # (num_block, num_frame, num_bin, num_null) --> (num_block, num_null, num_frame, num_bin)
            feats = self.cnnE(feats).squeeze(1)              # (num_block, 1, num_frame, num_dim) --> (num_block, num_frame, num_dim)
        else:
            feats = torch.flatten(feats, start_dim=-2, end_dim=-1) # ( num_block, num_frame, num_dim )

        if spect_feats is not None:
            feats = torch.cat((feats, spect_feats), dim = -1)

        if self.rnnE is not None:
            feats = self.rnnE(feats)                               # ( num_block, num_frame, num_dim )
        
        mask = self.mask(feats) # ( num_block, num_frame, num_bin )

        enh_out  = self.gsc(input = mFFT, mask = mask, beam_id = tdoa, alpha_v = self.gsc_config['alpha_v'], DCF_Ratio = DCF_Ratio) # ( num_block, num_frame, 2, num_bin)
        
        return enh_out, mask # ( num_block, num_frame, 2, num_bin), ( num_block, num_frame, num_bin )
    
    def get_trainable_params(self, print_info = False):
        if print_info:
            print("####### Trainable Parames of DCFGSCEnhNet")
        weights, biases = [], []
        if not self.dcf_config['fix']:
            if print_info:
                print("dcf = ")
            dcf_param_names = []
            for name, param in self.dcf.named_parameters():
                dcf_param_names.append(name)
                if 'bias' in name:
                    biases += [param]
                else:
                    weights += [param]
            if print_info:
                print(dcf_param_names)
        if not self.gsc_config['fix']:
            if print_info:
                print("gsc = ")
            gsc_param_names = []
            for name, param in self.gsc.named_parameters():
                gsc_param_names.append(name)
                if 'bias' in name:
                    biases += [param]
                else:
                    weights += [param]
            if print_info:
                print(gsc_param_names)
        
        if self.cnnE is not None:
            if print_info:
                print("cnnE = ")
            cnnE_param_names = []
            for name, param in self.cnnE.named_parameters():
                cnnE_param_names.append(name)
                if 'bias' in name:
                    biases += [param]
                else:
                    weights += [param]
            if print_info:
                print(cnnE_param_names)
        
        if self.rnnE is not None:
            if print_info:
                print("rnnE = ")
            rnnE_param_names = []
            for name, param in self.rnnE.named_parameters():
                rnnE_param_names.append(name)
                if 'bias' in name:
                    biases += [param]
                else:
                    weights += [param]
            if print_info:
                print(rnnE_param_names)

        if print_info:
            print("mask = ")
        mask_param_names = []
        for name, param in self.mask.named_parameters():
            mask_param_names.append(name)
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        if print_info:
            print(mask_param_names)

        if len(weights) < 1:
            return None
        params = [{'params': weights, }, {'params': biases, }]
        return params

class Neural_Spatial_Filter(nn.Module):
    def __init__(self, dpr_config, df_config, net_config):
        r"""
        Implement Neural Spatial Filter proposed by paper "Neural Spatial Filter: Target Speaker Speech Separation Assisted with Directional Information", 
        the input is complex FFT coefficients of multi-channel audios with shape [num_frame, 2, num_bin, num_channel], 
        and the output is complex FFT coefficients of the enhanced speech with shape [num_frame, 2, num_bin]

        dpr_config:
            num_bin: number of frequency bin
            num_channel: number of channels
            num_beam: number of beamformer for enhance directions
            targ_bf_weight: beamformer filter coefficients for enhance directions
            fix: If set to True, the beamformer layer will not be learn and updated. Default: True

        df_config:
            num_bin: number of frequency bin
            num_channel: number of channels
            mic_pos: position of microphone below fomat, egs: [(x1, y1), (x2, y2), ...] 
            sound_speed: speed of sound wave in the air
            sample_rate: sample rate
            pair_id: pair id for computing directionary features
            do_IPD: If set to True, using IPD feature, otherwise False, Default: True
            do_cosIPD: If set to True, using cosIPD feature, otherwise False, Default: True
            do_sinIPD: If set to True, using sinIPD feature, otherwise False, Default: True
            do_ILD: If set to True, using ILD feature, otherwise False, Default: True

        net_config:
            rnn_layer_size: size of rnn hidden layer, Default: [40, 128, 128, 128, 257]
            rnn_type: type of rnn, gru | lstm | rnn | None
            bidirectional: If set to True, RNN is bidirectional, otherwise False, Default: False

            cnn_layer_size: size of cnn hidden layer, Default: [40, 128, 128, 128, 257]
            cnn_type: type of cnn | resnet | u-net | dense-net | None
            
            mask_layer_size: size of mask net
            mask_act_type: type of mask activation, None|relu|sigmoid|softmax|tanh|softplus|prelu|leakyrelu

            batch_norm: Default: False
            bias: Default: True
            dropout: Default: 0.0
        """
        super(Neural_Spatial_Filter, self).__init__()
        
        self.dpr_config = dpr_config
        self.df_config  = df_config
        self.net_config = net_config

        ## construct DPR_DSNR
        num_bin            = dpr_config['num_bin']
        num_channel        = dpr_config['num_channel']
        num_beam           = dpr_config['num_beam']
        targ_bf_weight     = dpr_config['targ_bf_weight']
        fix                = dpr_config['fix']
        self.dpr_sdnr      = DPR_DSNR( num_beam = num_beam, num_bin = num_bin, num_channel = num_channel, targ_bf_weight = targ_bf_weight, fix = fix )
        self.dpr_sdnr_size = 2 * num_bin

        self.num_tdoa      = num_beam

        ## construct DF
        num_bin         = df_config['num_bin']
        num_channel     = df_config['num_channel']
        mic_pos         = df_config['mic_pos']
        sound_speed     = df_config['sound_speed']
        sample_rate     = df_config['sample_rate']
        self.df         = DF( num_bin = num_bin, num_channel = num_channel, mic_pos = mic_pos, sound_speed = sound_speed, sample_rate = sample_rate )
        self.df_size    = num_bin

        ## construct IPD
        self.pair_id   = df_config['pair_id']
        num_bin        = df_config['num_bin']
        do_IPD         = df_config['do_IPD']
        do_cosIPD      = df_config['do_cosIPD']
        do_sinIPD      = df_config['do_sinIPD']
        do_ILD         = df_config['do_ILD']
        self.ipd       = IPD(num_bin = num_bin, do_IPD = do_IPD, do_cosIPD = do_cosIPD, do_sinIPD = do_sinIPD, do_ILD = do_ILD)
        self.ipd_size  = self.ipd.get_size(pair_id = self.pair_id)

        self.spect_feat_size = net_config['spect_feat_size']
        rnn_input_size       = self.spect_feat_size + self.ipd_size  + self.df_size + self.dpr_sdnr_size
        mask_input_size      = self.spect_feat_size + self.ipd_size  + self.df_size + self.dpr_sdnr_size
        
        ## define and initialize the rnnE
        if net_config['rnn_type'] is not None:
            layer_size      = [rnn_input_size]
            layer_size      = layer_size + net_config['rnn_layer_size']
            rnn_type        = net_config['rnn_type']
            bidirectional   = net_config['bidirectional']
            batch_norm      = net_config['batch_norm']
            bias            = net_config['bias']
            dropout         = net_config['dropout']
            self.rnnE       = DeepRNNNet(layer_size = layer_size, rnn_type = rnn_type, bidirectional = bidirectional, batch_norm = batch_norm, bias = bias, dropout = dropout)
            mask_input_size = layer_size[-1]
        else:
            self.rnnE       = None

        ## define and initialize the mask_prj
        layer_size    = [mask_input_size]
        layer_size    = layer_size + net_config['mask_layer_size']
        mask_act_type = net_config['mask_act_type']
        bias          = net_config['bias']
        dropout       = net_config['dropout']
        self.mask = DeepFFNet(layer_size = layer_size, hidden_act_type = 'relu', out_act_type = mask_act_type, batch_norm = False, dropout = dropout, bias = bias, binary = False)

    def fix_bf_enhance(self, mFFT, tdoa):
        with torch.no_grad():
            if tdoa is not None and self.num_tdoa > 0:
                beam_id = tdoa / ( 360.0 / self.num_tdoa ) + 0.5
                beam_id = beam_id.int() % self.num_tdoa
            else:
                beam_id = None
                
            _, _, targ_bf_out = self.dpr_sdnr(input = mFFT, beam_id = beam_id) # (num_block, num_frame, num_bin), (num_block, num_frame, num_bin), (num_block, num_frame, 2, num_bin)

            return targ_bf_out # ( num_block, num_frame, 2, num_bin )
    
    def forward(self, mFFT, tdoa, cmvn = None, fbank_extractor = None):
        # mFFT:         ( num_block * num_frame, 2, num_bin, num_channel )
        # spect_feats:  ( num_block, num_frame, feat_size )
        
        num_block = len(tdoa)
        num_pair  = len(self.pair_id)
        
        if tdoa is not None and self.num_tdoa > 0:
            beam_id = tdoa / ( 360.0 / self.num_tdoa ) + 0.5
            beam_id = beam_id.int() % self.num_tdoa
        else:
            beam_id = None

        dpr, dsnr, ehFFT = self.dpr_sdnr(input = mFFT, beam_id = beam_id)                 # (num_block, num_frame, num_bin), (num_block, num_frame, 2, num_bin)
        df               = self.df(input = mFFT, beam_id = tdoa, pair_id = self.pair_id)  # (num_block, num_frame, num_bin)

        ipd              = self.ipd(input = mFFT, pair_id = self.pair_id)                 # (num_pair, num_block * num_frame, ipd_size)
        ipd              = ipd.view([num_pair, num_block, -1, self.ipd_size])             # (num_pair, num_block, num_frame, ipd_size)
        ipd              = ipd.permute([1, 2, 0, 3]).contiguous()                         # (num_block, num_frame, num_pair, ipd_size)
        ipd              = torch.flatten(ipd, start_dim=-2, end_dim=-1)                   # (num_block, num_frame, num_pair * ipd_size)

        feats            = torch.cat((dpr, dsnr, df, ipd), -1)                            # (num_block, num_frame, size)
        
        if self.spect_feat_size > 0:
            spect_feats   = torch.unsqueeze(ehFFT[:, :, 0, :] ** 2 + ehFFT[:, :, 1, :] ** 2, 1) # ( num_block, 1, num_frame, num_bin )
            spect_feats   = torch.clamp(spect_feats, min=1.0e-13)                               # ( num_block, 1, num_frame, num_bin )

            if fbank_extractor is not None:
                spect_feats = fbank_extractor(torch.sqrt(spect_feats))  # ( num_block, 1, num_frame, spect_feat_size )
            else:
                spect_feats = torch.log(spect_feats)                    # ( num_block, 1, num_frame, spect_feat_size )

            spect_feats = spect_feats.squeeze(1)                        # ( num_block, num_frame, spect_feat_size )

            # apply cmvn to mFFT
            if cmvn is not None:
                add_shift = cmvn[0, :].squeeze()                # (spect_feat_size)
                add_shift = add_shift.unsqueeze(0).unsqueeze(0) # (1, 1, spect_feat_size)
                add_shift = add_shift.expand_as(spect_feats)    # (num_block, num_frame, spect_feat_size)
    
                add_scale = cmvn[1, :].squeeze()                # (spect_feat_size)
                add_scale = add_scale.unsqueeze(0).unsqueeze(0) # (1, 1, spect_feat_size)
                add_scale = add_scale.expand_as(spect_feats)    # (num_block, num_frame, spect_feat_size)
    
                spect_feats = ( spect_feats + add_shift ) * add_scale # (num_block, num_frame, spect_feat_size)
        else:
            spect_feats = None
        
        if spect_feats is not None:
            feats = torch.cat((feats, spect_feats), dim = -1) # (num_block, num_frame, spect_feat_size + dpr_dsnr_size)

        if self.rnnE is not None:
            feats = self.rnnE(feats)                          # ( num_block, num_frame, num_dim )
        
        mask = self.mask(feats) # ( num_block, num_frame, num_bin )

        enh_out = mask.unsqueeze(2) * ehFFT # (num_block, num_frame, 2, num_bin)
        
        return enh_out, mask # ( num_block, num_frame, 2, num_bin), ( num_block, num_frame, num_bin )

    def get_trainable_params(self, print_info = False):
        if print_info:
            print("####### Trainable Parames of Neural_Spatial_Filter")
        
        weights, biases = [], []
        if not self.dpr_config['fix']:
            if print_info:
                print("DPR_DSNR = ")
            dpr_sdnr_param_names = []
            for name, param in self.dpr_sdnr.named_parameters():
                dpr_sdnr_param_names.append(name)
                if 'bias' in name:
                    biases += [param]
                else:
                    weights += [param]
            if print_info:
                print(dpr_sdnr_param_names)

        if self.rnnE is not None:
            if print_info:
                print("rnnE = ")
            rnnE_param_names = []
            for name, param in self.rnnE.named_parameters():
                rnnE_param_names.append(name)
                if 'bias' in name:
                    biases += [param]
                else:
                    weights += [param]
            if print_info:
                print(rnnE_param_names)

        if print_info:
            print("mask = ")
        mask_param_names = []
        for name, param in self.mask.named_parameters():
            mask_param_names.append(name)
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        if print_info:
            print(mask_param_names)

        if len(weights) < 1:
            return None
        params = [{'params': weights, }, {'params': biases, }]
        
        return params

class CConv2DMultiEnhNet(nn.Module):
    def __init__(self, spatial_filter_config, frequency_align_config):
        r"""
        Implement multi-channel filter by spatial filtering layer and frequency align layer, 
        the input is complex FFT coefficients of multi-channel audios with shape [num_frame, 2, num_bin, num_channel], 
        and the output is complex FFT coefficients of the enhanced speech with shape [num_frame, 2, num_bin]

        spatial_filter_config:
            num_bin: number of frequency bin
            num_channel: number of channels
            num_beam: number of beamforming filters
            bias: If set to False, the layer will not learn an additive bias. Default: True
            fixed: If set to True, the spatial_filter layer will not be learn and updated. Default: True
            weight_init: None
            bias_init: None
            regularization_weight: regularization weight of spatial filter layer, Default: 0.1
            regularization_p: p-norm of spatial filter layer, Default: 1.0 or 2.0

        frequency_align_config:
            type: type of frequency alignment layer, fc or conv
            num_align: number of frequency alignment ways
            num_bin: number of frequency bin
            num_beam: number of beamforming filters
            pooling_type: type of alignment pooling, agv or max
            num_freq_kernel: if the type of frequency alignment layer is conv, num_freq_kernel MUST be given, it means that the size along frequency axis
            bias: If set to False, the layer will not learn an additive bias. Default: True
            batch_norm: Default: False
            dropout: rate of dropout, Default 0.0
            fixed: If set to True, the spatial_filter layer will not be learn and updated. Default: True
            weight_init: None
            bias_init: None
            regularization_weight: regularization weight of spatial filter layer, Default: 0.1
            regularization_p: p-norm of spatial filter layer, Default: 1.0 or 2.0
        """
        super(CConv2DMultiEnhNet, self).__init__()
        
        self.spatial_filter_config  = spatial_filter_config
        self.frequency_align_config = frequency_align_config

        self.num_bin     = spatial_filter_config['num_bin']
        self.num_channel = spatial_filter_config['num_channel']
        self.num_beam    = spatial_filter_config['num_beam']

        ## construct spatial_filter
        num_bin     = spatial_filter_config['num_bin']
        num_channel = spatial_filter_config['num_channel']
        num_beam    = spatial_filter_config['num_beam']
        bias        = spatial_filter_config['bias']
        self.spatial_filter = CConv2DSFLayer(num_beam = num_beam, num_bin = num_bin, num_channel = num_channel, bias = bias)

        regularization_weight = spatial_filter_config['regularization_weight']
        regularization_p      = spatial_filter_config['regularization_p']
        fixed                 = spatial_filter_config['fixed']
        if regularization_weight > 0.0 and not fixed:
            self.spatial_filter_regularization = Regularization(weight_decay = regularization_weight, p = regularization_p)
        else:
            self.spatial_filter_regularization = None

        if frequency_align_config['type'].lower() == 'fc':
            num_align    = frequency_align_config['num_align']
            num_bin      = frequency_align_config['num_bin']
            num_beam     = frequency_align_config['num_beam']
            pooling_type = frequency_align_config['pooling_type']
            bias         = frequency_align_config['bias']
            batch_norm   = frequency_align_config['batch_norm']
            dropout      = frequency_align_config['dropout']
            weight_init  = frequency_align_config['weight_init']
            bias_init    = frequency_align_config['bias_init']

            self.frequency_align = FCFALayer(num_beam = num_beam, num_align = num_align, num_bin = num_bin, bias = bias, pooling_type = pooling_type, batch_norm = batch_norm, dropout = dropout, weight_init = weight_init, bias_init = bias_init)
        elif frequency_align_config['type'].lower() == 'conv':
            num_align           = frequency_align_config['num_align']
            num_bin             = frequency_align_config['num_bin']
            num_freq_kernel     = frequency_align_config['num_freq_kernel']
            num_beam_kernel     = frequency_align_config['num_beam']
            pooling_type        = frequency_align_config['pooling_type']
            bias                = frequency_align_config['bias']
            batch_norm          = frequency_align_config['batch_norm']
            dropout             = frequency_align_config['dropout']
            weight_init         = frequency_align_config['weight_init']
            bias_init           = frequency_align_config['bias_init']

            self.frequency_align = Conv2DFALayer(num_freq_kernel = num_freq_kernel, num_beam_kernel = num_beam_kernel, num_align = num_align, num_bin = num_bin, bias = bias, pooling_type = pooling_type, batch_norm = batch_norm, dropout = dropout, weight_init = weight_init, bias_init = bias_init)
        else:
            num_align           = frequency_align_config['num_align']
            num_bin             = frequency_align_config['num_bin']
            num_freq_kernel     = frequency_align_config['num_freq_kernel']
            num_beam_kernel     = frequency_align_config['num_beam']
            pooling_type        = frequency_align_config['pooling_type']
            bias                = frequency_align_config['bias']
            batch_norm          = frequency_align_config['batch_norm']
            dropout             = frequency_align_config['dropout']
            self.frequency_align = DeepConv2DFALayer(num_freq_kernel = num_freq_kernel, num_beam_kernel = num_beam_kernel, num_align = num_align, num_bin = num_bin, bias = bias, pooling_type = pooling_type, batch_norm = batch_norm, dropout = dropout)

        regularization_weight = frequency_align_config['regularization_weight']
        regularization_p      = frequency_align_config['regularization_p']
        fixed                 = frequency_align_config['fixed']
        if regularization_weight > 0.0 and not fixed:
            self.frequency_align_regularization = Regularization(weight_decay = regularization_weight, p = regularization_p)
        else:
            self.frequency_align_regularization = None

    def forward(self, input, guid_info = None, num_block = 1):
        # input: ( num_frame, 2, num_bin, num_channel )

        # input_r = ( num_block, 1, num_bin, num_frame * num_channel )
        # input_i = ( num_block, 1, num_bin, num_frame * num_channel )
        num_bin     = input.size(2)
        num_channel = input.size(3)
        input_r     = input[:, 0, :, :].squeeze() # ( num_frame, num_bin, num_channel )
        input_i     = input[:, 1, :, :].squeeze() # ( num_frame, num_bin, num_channel )

        input_r = input_r.view(num_block, -1, num_bin, num_channel) # (num_block, num_frame, num_bin, num_channel)
        input_i = input_i.view(num_block, -1, num_bin, num_channel) # (num_block, num_frame, num_bin, num_channel)

        input_r = input_r.permute([0, 2, 1, 3]).contiguous() # (num_block, num_bin, num_frame, num_channel)
        input_i = input_i.permute([0, 2, 1, 3]).contiguous() # (num_block, num_bin, num_frame, num_channel)

        input_r = input_r.view(num_block, num_bin, -1).unsqueeze(1) # (num_block, 1, num_bin, num_frame * num_channel)
        input_i = input_i.view(num_block, num_bin, -1).unsqueeze(1) # (num_block, 1, num_bin, num_frame * num_channel)

        if self.spatial_filter_config['fixed']:
            with torch.no_grad():
                spatial_filter_out_r, spatial_filter_out_i = self.spatial_filter(input_r, input_i) # (num_block, num_beam, num_bin, num_frame)

                spatial_filter_out_r = spatial_filter_out_r.permute([0, 3, 2, 1]).contiguous() # (num_block, num_frame, num_bin, num_beam)
                spatial_filter_out_i = spatial_filter_out_i.permute([0, 3, 2, 1]).contiguous() # (num_block, num_frame, num_bin, num_beam)

                spatial_filter_out_r = spatial_filter_out_r.view(-1, num_bin, self.num_beam).unsqueeze(1)   # (num_frame, 1, num_bin, num_beam)
                spatial_filter_out_i = spatial_filter_out_i.view(-1, num_bin, self.num_beam).unsqueeze(1)   # (num_frame, 1, num_bin, num_beam)

                spatial_filter_out   = torch.cat((spatial_filter_out_r, spatial_filter_out_i), 1)             # (num_frame, 2, num_bin, num_beam)
        else:
            spatial_filter_out_r, spatial_filter_out_i = self.spatial_filter(input_r, input_i)     # (num_block, num_beam, num_bin, num_frame)

            spatial_filter_out_r = spatial_filter_out_r.permute([0, 3, 2, 1]).contiguous() # (num_block, num_frame, num_bin, num_beam)
            spatial_filter_out_i = spatial_filter_out_i.permute([0, 3, 2, 1]).contiguous() # (num_block, num_frame, num_bin, num_beam)

            spatial_filter_out_r = spatial_filter_out_r.view(-1, num_bin, self.num_beam).unsqueeze(1)   # (num_frame, 1, num_bin, num_beam)
            spatial_filter_out_i = spatial_filter_out_i.view(-1, num_bin, self.num_beam).unsqueeze(1)   # (num_frame, 1, num_bin, num_beam)

            spatial_filter_out   = torch.cat((spatial_filter_out_r, spatial_filter_out_i), 1)             # (num_frame, 2, num_bin, num_beam)

        if self.frequency_align_config['fixed']:
            with torch.no_grad():
                frequency_align_out = self.frequency_align(spatial_filter_out, guid_info = None, num_block = num_block)  # ( num_frame, 2, num_bin )
        else:
            frequency_align_out = self.frequency_align(spatial_filter_out, guid_info = None, num_block = num_block)      # ( num_frame, 2, num_bin )
        
        return frequency_align_out # ( num_frame, 2, num_bin ) 
    
    def get_trainable_params(self, print_info = False):
        if print_info:
            print("####### Trainable Parames of MultiEnhNet")
        weights, biases = [], []
        if not self.spatial_filter_config['fixed']:
            if print_info:
                print("spatial_filter = ")
            spatial_filter_param_names = []
            for name, param in self.spatial_filter.named_parameters():
                spatial_filter_param_names.append(name)
                if 'bias' in name:
                    biases += [param]
                else:
                    weights += [param]
            if print_info:
                print(spatial_filter_param_names)
        
        if not self.frequency_align_config['fixed']:
            if print_info:
                print("frequency_align = ")
            frequency_align_param_names = []
            for name, param in self.frequency_align.named_parameters():
                frequency_align_param_names.append(name)
                if 'bias' in name:
                    biases += [param]
                else:
                    weights += [param]
            if print_info:
                print(frequency_align_param_names)
            
        if len(weights) < 1:
            return None
        params = [{'params': weights, }, {'params': biases, }]
        return params
    
    def get_regularization_loss(self):
        
        reg_loss = 0.0

        if self.spatial_filter_regularization is not None:
            reg_loss = reg_loss + self.spatial_filter_regularization(self.spatial_filter)
        
        if self.frequency_align_regularization is not None:
            reg_loss = reg_loss + self.frequency_align_regularization(self.frequency_align)

        return reg_loss