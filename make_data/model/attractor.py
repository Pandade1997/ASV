import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import functools
import numpy as np
import math
from torch.autograd import Variable
import torch.nn.functional as F

from .model_config import DEBUG_QUANTIZE
from .rnnlayer import RNNLayer
from .fclayer import FCLayer

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


def fa_attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # query:  ( num_frame, 1, num_bin)          -->     attractor  : # ( num_block, 1, num_bin, num_frame ) --> ( num_frame, 1, num_bin)
    # key:    ( num_frame, num_bin, num_align)  -->     aligned_speech : # ( num_frame, num_bin, num_align)
    # value:  ( num_frame, 2, num_bin, num_align)
    # output: ( num_frame, 2, num_bin)

    # ( num_frame, 1, num_bin) x ( num_frame, num_bin, num_align) = (num_frame, 1, num_align)
    d_k = query.size(-1)
    scores = torch.matmul(query, key) / math.sqrt(d_k) # (num_frame, 1, num_align)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim = -1)      # (num_frame, 1, num_align)
    if dropout is not None:
        p_attn = dropout(p_attn)
    p_attn = torch.cat((p_attn, p_attn), 1).unsqueeze(3)    # (num_frame, 2, num_align, 1)
    
    # ( num_frame, 2, num_bin, num_align) x (num_frame, 2, num_align, 1) = (num_frame, 2, num_bin, 1) -- > ( num_frame, 2, num_bin)
    Z_o = torch.matmul(value, p_attn).squeeze(3)

    return Z_o, p_attn.squeeze() # ( num_frame, 2, num_bin), (num_frame, 2, num_align)

class SimAttractor(nn.Module):
    def __init__(self):
        super(SimAttractor, self).__init__()
        self.attr_fun = nn.Softmax(dim=-1)

    def forward(self, query, key, value, Low_Freq = 5, High_Freq = 70):
        "Compute 'Scaled Dot Product Attention'"
        # query:  ( num_frame, 1, num_bin)
        # key:    ( num_frame, num_bin, num_align)
        # value:  ( num_frame, 2, num_bin, num_align)
        # output: ( num_frame, 2, num_bin)

        key    = key.transpose(-1, -2).contiguous() # ( num_frame, num_bin, num_align) -> ( num_frame, num_align, num_bin)
        query  = query.expand_as(key).contiguous()  # ( num_frame, 1, num_bin)         -> ( num_frame, num_align, num_bin)

        scores = torch.cosine_similarity(query, key, dim=-1) # (num_frame, num_align)
        scores = scores.unsqueeze(1)                         # (num_frame, 1, num_align)

        p_attn = self.attr_fun(scores)                         # (num_frame, 1, num_align)
        p_attn = torch.cat((p_attn, p_attn), 1).unsqueeze(3)   # (num_frame, 2, num_align, 1)

        # ( num_frame, 2, num_bin, num_align) x (num_frame, 2, num_align, 1) = (num_frame, 2, num_bin, 1) -- > ( num_frame, 2, num_bin)
        Z_o = torch.matmul(value, p_attn).squeeze(3)

        return Z_o, p_attn.squeeze() # ( num_frame, 2, num_bin), (num_frame, 2, num_align)

'''    
class FFAttractor(nn.Module):
    def __init__(self, num_align = 80):
        super(FFAttractor, self).__init__()

        self.attractor_net = DeepFFNet(layer_size = [num_align, num_align, num_align], binary = False, hidden_act_type='relu', out_act_type=None, batch_norm=False, dropout=0.0, bias=True)

        self.attr_fun = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        "Compute 'Scaled Dot Product Attention'"
        # query:  ( num_frame, 1, num_bin)
        # key:    ( num_frame, num_bin, num_align)
        # value:  ( num_frame, 2, num_bin, num_align)
        # output: ( num_frame, 2, num_bin)

        key    = key.transpose(-1, -2).contiguous() # ( num_frame, num_bin, num_align) -> ( num_frame, num_align, num_bin)
        query  = query.expand_as(key)               #  ( num_frame, 1, num_bin)        -> ( num_frame, num_align, num_bin)

        scores = torch.cosine_similarity(query, key, dim=-1) # (num_frame, num_align)
        scores = self.attractor_net(scores)                  # (num_frame, num_align)
        scores = scores.unsqueeze(1)                         # (num_frame, 1, num_align)

        p_attn = self.attr_fun(scores)                       # (num_frame, 1, num_align)

        p_attn = torch.cat((p_attn, p_attn), 1).unsqueeze(3) # (num_frame, 2, num_align, 1)
        
        # ( num_frame, 2, num_bin, num_align) x (num_frame, 2, num_align, 1) = (num_frame, 2, num_bin, 1) -- > ( num_frame, 2, num_bin)
        Z_o = torch.matmul(value, p_attn).squeeze(3)

        return Z_o, p_attn.squeeze() # ( num_frame, 2, num_bin), (num_frame, 2, num_align)
'''
    
class RNNAttractor(nn.Module):
    def __init__(self, num_align = 80):
        super(RNNAttractor, self).__init__()

        self.attractor_net = DeepRNNNet(layer_size = [num_align, num_align], rnn_type = 'gru', bidirectional = False, batch_norm = False, bias = True, dropout = 0.0)

        self.attr_fun = Outputer(in_size = num_align, out_size = num_align, out_act_type = 'softmax', bias = True)
        
    def forward(self, query, key, value, num_block = 1):
        "Compute 'Scaled Dot Product Attention'"
        # query:  ( num_frame, 1, num_bin)
        # key:    ( num_frame, num_bin, num_align)
        # value:  ( num_frame, 2, num_bin, num_align)
        # output: ( num_frame, 2, num_bin)

        num_align = key.size(-1)

        key    = key.transpose(-1, -2).contiguous()          # ( num_frame, num_bin, num_align) -> ( num_frame, num_align, num_bin)
        query  = query.expand_as(key)                        # ( num_frame, 1, num_bin)        -> ( num_frame, num_align, num_bin)

        scores = torch.cosine_similarity(query, key, dim=-1) # (num_frame, num_align)

        scores = scores.view([num_block, -1, num_align]).contiguous() # (num_block, num_frame, num_align)

        scores = self.attractor_net(scores)                  # (num_block, num_frame, num_align)

        scores = scores.view([-1, num_align])                # (num_frame, num_align)

        scores = scores.unsqueeze(1)                         # (num_frame, 1, num_align)

        p_attn = self.attr_fun(scores)                       # (num_frame, 1, num_align)

        p_attn = torch.cat((p_attn, p_attn), 1).unsqueeze(3) # (num_frame, 2, num_align, 1)
        
        # ( num_frame, 2, num_bin, num_align) x (num_frame, 2, num_align, 1) = (num_frame, 2, num_bin, 1) -- > ( num_frame, 2, num_bin)
        Z_o = torch.matmul(value, p_attn).squeeze(3)

        return Z_o, p_attn.squeeze() # ( num_frame, 2, num_bin), (num_frame, 2, num_align)

