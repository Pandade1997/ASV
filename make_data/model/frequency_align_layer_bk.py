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
from .fbanklayer import to_cuda
from .model_config import DEBUG_QUANTIZE
from .complex_conv_layer import CConv2dLayer
from .rnnlayer import RNNLayer
from .gatelayer import RNNGate, Gate
import torch.nn.functional as F

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

#############################################################################################################
######################## Frequncy align layer for multi-direction beamforming output ########################
#############################################################################################################
class FCFALayer(nn.Module):
    r"""Using full-connection to implement frequncy align layer for multi-direction beamforming output
    Input:  multi-direction beamforming output, shape = (num_frame, 2, num_bin, num_beam)
    Output: frequncy alignment output, shape = (num_frame, 2, num_bin)

    Args:
        num_beam: number of beamforming filters
        num_align: number of frequency alignment ways
        num_bin: number of frequency bin
        pooling_type: type of pooling the multi-alignments, avg: average pooling, max, max pooling
    """
    def __init__(self, num_beam, num_align, num_bin = 256, bias = True, pooling_type = 'avg', batch_norm = False, dropout = 0.0, weight_init = None, bias_init = None, fixed_align = False):
        super(FCFALayer, self).__init__()
        
        self.num_beam     = num_beam
        self.num_align    = num_align
        self.num_bin      = num_bin
        self.bias         = bias
        self.pooling_type = pooling_type
        self.fixed_align  = fixed_align
        
        #self.batch_norm = nn.LayerNorm(num_bin) if batch_norm else None
        self.batch_norm  = nn.BatchNorm1d(num_bin) if batch_norm else None
        self.dropout     = nn.Dropout(dropout) if dropout > 0.0 else None

        self.fcfa = nn.Linear(num_beam, num_align, bias = bias)

        if weight_init is not None:
            self.fcfa.weight.data.copy_(weight_init)
        if bias_init is not None and bias:
            self.fcfa.bias.data.copy_(bias_init)

        if self.pooling_type is not None and self.pooling_type.lower() == 'gate': # pow:    ( num_frame, 1, num_bin, num_align) 
            self.gate = RNNGate(channels=[12, 24, 48], num_align = num_align)
        else:
            self.gate = None

        if DEBUG_QUANTIZE:
            self.fc_min        = 100000000.0
            self.fc_max        = -100000000.0
            self.fc_mean       = 0.0
            self.fc_std        = 0.0
            self.bn_min        = 100000000
            self.bn_max        = -100000000
            self.bn_mean       = 0.0
            self.bn_std        = 0.0

    def get_trainable_params(self, print_info = False):
        if print_info:
            print("####### Trainable Parames of frequncy_align_layer")
        if print_info:
                print("frequency_align = ")

        frequency_align_param_names = []
        weights, biases = [], []
        if self.gate is not None:
            for name, param in self.gate.named_parameters():
                frequency_align_param_names.append(name)
                if 'bias' in name:
                    biases += [param]
                else:
                    weights += [param]
        if not self.fixed_align:
            for name, param in self.fcfa.named_parameters():
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

    def forward(self, input, attractor = None, num_block = 1, numerical_protection = 1.0e-13, compressed_scale = 0.3):
        # input:         ( num_frame, 2, num_bin, num_beam ) * (num_beam, num_align)
        # attractor: ( num_block, 2, num_bin, num_frame )
        # output:        ( num_frame, 2, num_bin )
        num_frame, num_dim, num_bin, num_beam = input.size() # (num_frame, 2, num_bin, num_beam)
        assert num_dim == 2 and num_bin == self.num_bin and num_beam == self.num_beam, "illegal shape for input, the required input shape is (%d, 2, %d, %d), but got (%d, %s, %d, %d)" % (num_frame, self.num_bin, self.num_align, num_frame, num_dim, num_bin, num_beam)
        
        input = input.view(-1, num_beam)                     # (num_frame*2*num_bin, num_beam)

        if self.fixed_align:
             with torch.no_grad():
                align_out = self.fcfa(input) # (num_frame*2*num_bin, num_beam) * (num_beam, num_align) = (num_frame*2*num_bin, num_align)
        else:
            align_out = self.fcfa(input) # (num_frame*2*num_bin, num_beam) * (num_beam, num_align) = (num_frame*2*num_bin, num_align)

        align_out = align_out.view(num_frame, num_dim, num_bin, -1) # ( num_frame, 2, num_bin, num_align)

        if self.pooling_type is not None:
            if self.pooling_type.lower() == 'avg':
                align_out = torch.mean(align_out, 3).squeeze()                 # (num_frame, 2, num_bin)
            elif self.pooling_type.lower() == 'max':
                align_pow = torch.unsqueeze(align_out[:,0,:,:] ** 2 + align_out[:,1,:,:] ** 2, 1) # ( num_frame, 1, num_bin, num_align)
                align_idx = torch.argmax(align_pow, dim = 3, keepdim = True)                      # (num_frame, 1, num_bin, 1)
                align_idx = align_idx.expand((num_frame, num_dim, num_bin, 1))                    # (num_frame, 2, num_bin, 1)
                align_out = torch.gather(align_out, dim = 3, index = align_idx).squeeze()         # ( num_frame, 2, num_bin)
            elif self.pooling_type.lower() == 'gate':
                align_out, gate_mask = self.gate(align_out, numerical_protection = numerical_protection, compressed_scale = compressed_scale)
            elif self.pooling_type.lower() == 'attention':
                value = align_out                                                     # ( num_frame, 2, num_bin, num_align)
                if attractor is not None:
                    # attractor: ( num_block, 2, num_bin, num_frame )
                    '''
                    query = attractor[:,0,:,:] ** 2 + attractor[:,1,:,:] ** 2   # ( num_block, num_bin, num_frame )
                    query = torch.clamp(query, min = numerical_protection)      # ( num_block, num_bin, num_frame )
                    query = query ** (0.5 * compressed_scale)                   # ( num_block, num_bin, num_frame )
                    query = query.transpose(-1, -2).contiguous()                # ( num_block, num_frame, num_bin )
                    query = query.view(-1, 1, num_bin)                          # ( num_frame, 1, num_bin )
                    '''

                    query = attractor[:,0,:,:-1] ** 2 + attractor[:,1,:,:-1] ** 2 # ( num_block, num_bin, num_frame )
                    query = torch.clamp(query, min = numerical_protection)            # ( num_block, num_bin, num_frame )
                    query = query ** (0.5 * compressed_scale)                         # ( num_block, num_bin, num_frame )
                    query = query.transpose(-1, -2).contiguous()                      # ( num_block, num_frame, num_bin )

                    align_out_r = align_out[:, 0, :, :].squeeze() # (num_frame, num_bin, num_align)
                    align_out_i = align_out[:, 1, :, :].squeeze() # (num_frame, num_bin, num_align)
 
                    align_out_r = align_out_r.view(num_block, -1, self.num_align, num_bin) # (num_frame, num_bin, num_align) --> (num_block, num_frame, num_bin, num_align)
                    align_out_i = align_out_i.view(num_block, -1, self.num_align, num_bin) # (num_frame, num_bin, num_align) --> (num_block, num_frame, num_bin, num_align)

                    talign_out_r = align_out_r[:, 0, :, :]                          # (num_block, num_bin, num_align)
                    talign_out_i = align_out_i[:, 0, :, :]                          # (num_block, num_bin, num_align)

                    talign_out_r = torch.mean(talign_out_r, 2) # (num_block, num_bin)
                    talign_out_i = torch.mean(talign_out_i, 2) # (num_block, num_bin)
                    talign_out = torch.cat((talign_out_r.unsqueeze(1), talign_out_i.unsqueeze(1)), dim = 1) # ( num_block, 2, num_bin )

                    squery = talign_out[:,0,:] ** 2 + talign_out[:,1,:] ** 2   # ( num_block, num_bin )
                    squery = torch.clamp(squery, min = numerical_protection)   # ( num_block, num_bin )
                    squery = squery ** (0.5 * compressed_scale)                # ( num_block, num_bin )
                    squery = squery.unsqueeze(1)                               # ( num_block, 1, num_bin )

                    query = torch.cat((squery, query), dim = 1)                # ( num_block, num_frame, num_bin )
                    query = query.view(-1, 1, num_bin)                         # ( num_frame, 1, num_bin )

                    key = align_out[:,0,:,:] ** 2 + align_out[:,1,:,:] ** 2           # ( num_frame, num_bin, num_align )
                    key = torch.clamp(key, min = numerical_protection)                # ( num_frame, num_bin, num_align )
                    key = key ** (0.5 * compressed_scale)                             # ( num_frame, num_bin, num_align )

                    align_out, att_v = fa_attention(query, key, value)                # ( num_frame, 2, num_bin ), ( num_frame, 2, num_align )
                else:
                    align_out_r = align_out[:, 0, :, :].squeeze() # (num_frame, num_bin, num_align)
                    align_out_i = align_out[:, 1, :, :].squeeze() # (num_frame, num_bin, num_align)
 
                    align_out_r = align_out_r.view(num_block, -1, self.num_align, num_bin) # (num_frame, num_bin, num_align) --> (num_block, num_frame, num_bin, num_align)
                    align_out_i = align_out_i.view(num_block, -1, self.num_align, num_bin) # (num_frame, num_bin, num_align) --> (num_block, num_frame, num_bin, num_align)

                    sequential_length = align_out_r.size(1)

                    talign_out_r = align_out_r[:, 0, :, :]                          # (num_block, num_bin, num_align)
                    talign_out_i = align_out_i[:, 0, :, :]                          # (num_block, num_bin, num_align)

                    talign_out_r = torch.mean(talign_out_r, 2) # (num_block, num_bin)
                    talign_out_i = torch.mean(talign_out_i, 2) # (num_block, num_bin)
                    talign_out = torch.cat((talign_out_r.unsqueeze(1), talign_out_i.unsqueeze(1)), dim = 1) # ( num_block, 2, num_bin )

                    query = talign_out[:,0,:] ** 2 + talign_out[:,1,:] ** 2  # ( num_block, num_bin )
                    query = torch.clamp(query, min = numerical_protection)   # ( num_block, num_bin )
                    query = query ** (0.5 * compressed_scale)                # ( num_block, num_bin )
                    self.query = query.unsqueeze(1)                          # ( num_block, 1, num_bin )

                    align_out = talign_out                                   # ( num_block, 2, num_bin )
                    for t in torch.arange(1, sequential_length):

                        talign_out_r = align_out_r[:, t, :, :]                          # (num_block, num_bin, num_align)
                        talign_out_i = align_out_i[:, t, :, :]                          # (num_block, num_bin, num_align)

                        key = talign_out_r ** 2 + talign_out_i ** 2                     # ( num_block, num_bin, num_align )
                        key = torch.clamp(key, min = numerical_protection)              # ( num_block, num_bin, num_align )
                        key = key ** (0.5 * compressed_scale)                           # ( num_block, num_bin, num_align )

                        value = torch.cat((talign_out_r.unsqueeze(1), talign_out_i.unsqueeze(1)), dim = 1)        # (num_block, 2, num_bin, num_align)

                        # query: ( num_block, 1, num_bin ), key: ( num_block, num_bin, num_align ), value: ( num_block, 2, num_bin, num_align )
                        talign_out, tatt_v = fa_attention(self.query, key, value)   # ( num_block, 2, num_bin ), ( num_block, 2, num_align )
                        align_out = torch.cat((align_out, talign_out), dim = 0)     # ( num_frame, 2, num_bin )

                        query = talign_out[:,0,:] ** 2 + talign_out[:,1,:] ** 2  # ( num_block, num_bin )
                        query = torch.clamp(query, min = numerical_protection)   # ( num_block, num_bin )
                        query = query ** (0.5 * compressed_scale)                # ( num_block, num_bin )
                        self.query = query.unsqueeze(1)                          # ( num_block, 1, num_bin )
                    
                    if num_block > 1:
                        align_out = align_out.view(-1, num_block, 2, num_bin) # ( num_frame, num_block, 2, num_bin )
                        align_out = align_out.transpose(1, 0).contiguous()    # ( num_block, num_frame, 2, num_bin )
                        align_out = align_out.view(-1, 2, num_bin)            # ( num_block * num_frame, 2, num_bin )
            else:
                print("Unsurpported %s pooling_type!" % (self.pooling_type))
                exit(1)
        else:
            align_out = torch.mean(align_out, 3).squeeze()                 # (num_frame, 2, num_bin)
        
        if DEBUG_QUANTIZE:
            with torch.no_grad():
                self.fc_min  = min(align_out.min(), self.fc_min)
                self.fc_max  = max(align_out.max(), self.fc_max)
                self.fc_mean = (align_out.mean() +  self.fc_mean) / 2.0
                self.fc_std  = (align_out.std() +  self.fc_std) / 2.0

        if self.batch_norm is not None:
            align_out = align_out.view(-1, num_bin)
            align_out = self.batch_norm(align_out)
            align_out = align_out.view(num_frame, num_dim, num_bin) # ( num_frame, 2, num_bin)

            if DEBUG_QUANTIZE:
                with torch.no_grad():
                    self.bn_min  = min(align_out.min(), self.bn_min)
                    self.bn_max  = max(align_out.max(), self.bn_max)
                    self.bn_mean = (align_out.mean() + self.bn_mean) / 2.0
                    self.bn_std  = (align_out.std() + self.bn_std) / 2.0

        if self.dropout is not None:
            align_out = self.dropout(align_out) # ( num_frame, 2, num_bin)

        return align_out

class Conv2DFALayer(nn.Module):
    r"""Using Conv2D to implement frequncy align layer for multi-direction beamforming output
    Input:  multi-direction beamforming output, shape = (num_frame, 2, num_bin, num_beam)
    Output: frequncy alignment output, shape = (num_frame, 2, num_bin)

    Args:
        num_freq_kernel: size of kernel along the feqency axis
        num_beam_kernel: size of kernel along the beamforming axis
        num_align: number of frequency alignment ways
        num_bin: number of frequency bin
        pooling_type: type of pooling the multi-alignments, avg: average pooling, max, max pooling
    """
    def __init__(self, num_freq_kernel, num_beam_kernel, num_align, num_bin = 256, bias = True, pooling_type = 'avg', batch_norm = False, dropout = 0.0, weight_init = None, bias_init = None, fixed_align = False):
        super(Conv2DFALayer, self).__init__()
        
        self.num_beam_kernel     = num_beam_kernel
        self.num_freq_kernel     = num_freq_kernel
        self.num_align           = num_align
        self.num_bin             = num_bin
        self.bias                = bias
        self.pooling_type        = pooling_type
        self.fixed_align         = fixed_align
        
        #self.batch_norm = nn.LayerNorm(num_bin) if batch_norm else None
        self.batch_norm  = nn.BatchNorm1d(num_bin) if batch_norm else None
        self.dropout     = nn.Dropout(dropout) if dropout > 0.0 else None
        
        kernel_size = (num_freq_kernel, num_beam_kernel)
        stride      = (1, num_beam_kernel)
        padding     = (int(np.floor(num_freq_kernel / 2)), 0)
        self.conv2dfa = nn.Conv2d(in_channels = 1, out_channels = num_align, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)

        if weight_init is not None:
            self.conv2dfa.weight.data.copy_(weight_init)
        if bias_init is not None and bias:
            self.conv2dfa.bias.data.copy_(bias_init)
        
        if self.pooling_type is not None and self.pooling_type.lower() == 'gate': # pow:    ( num_frame, 1, num_bin, num_align) 
            self.gate = Gate(channels=[12, 24, 48], num_align = num_align)
        else:
            self.gate = None

        if DEBUG_QUANTIZE:
            self.fc_min        = 100000000.0
            self.fc_max        = -100000000.0
            self.fc_mean       = 0.0
            self.fc_std        = 0.0
            self.bn_min        = 100000000
            self.bn_max        = -100000000
            self.bn_mean       = 0.0
            self.bn_std        = 0.0

    def get_trainable_params(self, print_info = False):
        if print_info:
                print("frequency_align = ")

        frequency_align_param_names = []
        weights, biases = [], []
        if self.gate is not None:
            for name, param in self.gate.named_parameters():
                frequency_align_param_names.append(name)
                if 'bias' in name:
                    biases += [param]
                else:
                    weights += [param]
        if not self.fixed_align:
            for name, param in self.conv2dfa.named_parameters():
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

    def forward(self, input, attractor = None, num_block = 1, numerical_protection = 1.0e-23, compressed_scale = 0.3):
        # input  = ( num_frame, 2, num_bin, num_beam ) conv num_align * (num_freq_kernel, num_beam_kernel)
        # output = ( num_frame, 2, num_bin )
    
        num_frame, num_dim, num_bin, num_beam = input.size() # (num_frame, 2, num_bin, num_beam)
        assert num_dim == 2 and num_bin == self.num_bin and num_beam == self.num_beam_kernel, "illegal shape for input, the required input shape is (%d, 2, %d, %d), but got (%d, %s, %d, %d)" % (num_frame, self.num_bin, self.num_align, num_frame, num_dim, num_bin, num_beam)
        
        input_r = input[:, 0, :, :].unsqueeze(1) # (num_frame, 1, num_bin, num_beam)
        input_i = input[:, 1, :, :].unsqueeze(1) # (num_frame, 1, num_bin, num_beam)

        if self.fixed_align:
             with torch.no_grad():
                align_out_r = self.conv2dfa(input_r).squeeze()     # (num_frame, 1, num_align, num_bin) --> (num_frame, num_align, num_bin)
                align_out_i = self.conv2dfa(input_i).squeeze()     # (num_frame, 1, num_align, num_bin) --> (num_frame, num_align, num_bin) 
        else:
            align_out_r = self.conv2dfa(input_r).squeeze()     # (num_frame, 1, num_align, num_bin) --> (num_frame, num_align, num_bin)
            align_out_i = self.conv2dfa(input_i).squeeze()     # (num_frame, 1, num_align, num_bin) --> (num_frame, num_align, num_bin) 

        if self.pooling_type is not None:
            if self.pooling_type.lower() == 'avg': 
                align_out_r = torch.mean(align_out_r, 1).squeeze() # (num_frame, num_bin)
                align_out_i = torch.mean(align_out_i, 1).squeeze() # (num_frame, num_bin)
                align_out = torch.cat((align_out_r.unsqueeze(1), align_out_i.unsqueeze(1)), dim = 1) # ( num_frame, 2, num_bin )
            elif self.pooling_type.lower() == 'max':
                print("max")
                align_pow = align_out_r ** 2 + align_out_i ** 2  # ( num_frame, num_align, num_bin )
                align_idx = torch.argmax(align_pow, dim = 1, keepdim = True)                  # (num_frame, 1, num_bin)
                align_out_r = torch.gather(align_out_r, dim = 1, index = align_idx).squeeze() # (num_frame, num_bin)
                align_out_i = torch.gather(align_out_i, dim = 1, index = align_idx).squeeze() # (num_frame, num_bin)
                align_out = torch.cat((align_out_r.unsqueeze(1), align_out_i.unsqueeze(1)), dim = 1) # ( num_frame, 2, num_bin )
            elif self.pooling_type.lower() == 'attention':                                                
                align_out = torch.cat((align_out_r.unsqueeze(1), align_out_i.unsqueeze(1)), dim = 1) # ( num_frame, 1, num_align, num_bin )
                align_out = align_out.transpose(-1, -2)                                              # ( num_frame, 2, num_bin, num_align )
                value     = align_out     
                if attractor is not None:
                    query = attractor[:,0,:,:] ** 2 + attractor[:,1,:,:] ** 2 # ( num_block, num_bin, num_frame )
                    query = torch.clamp(query, min = numerical_protection)            # ( num_block, num_bin, num_frame )
                    query = query ** (0.5 * compressed_scale)                         # ( num_block, num_bin, num_frame )
                    query = query.transpose(-1, -2).contiguous()                      # ( num_block, num_frame, num_bin )
                    query = query.view(-1, 1, num_bin)                                # ( num_frame, 1, num_bin )
                    
                    key = align_out[:,0,:,:] ** 2 + align_out[:,1,:,:] ** 2           # ( num_frame, num_bin, num_align )
                    key = torch.clamp(key, min = numerical_protection)                # ( num_frame, num_bin, num_align )
                    key = key ** (0.5 * compressed_scale)                             # ( num_frame, num_bin, num_align )

                    '''
                    query = attractor[:,0,:,:-1] ** 2 + attractor[:,1,:,:-1] ** 2 # ( num_block, num_bin, num_frame )
                    query = torch.clamp(query, min = numerical_protection)            # ( num_block, num_bin, num_frame )
                    query = query ** (0.5 * compressed_scale)                         # ( num_block, num_bin, num_frame )
                    query = query.transpose(-1, -2).contiguous()                      # ( num_block, num_frame, num_bin )
                    

                    align_out_r = align_out_r.view(num_block, -1, self.num_align, num_bin) # (num_frame, num_align, num_bin) --> (num_block, num_frame, num_align, num_bin)
                    align_out_i = align_out_i.view(num_block, -1, self.num_align, num_bin) # (num_frame, num_align, num_bin) --> (num_block, num_frame, num_align, num_bin)

                    align_out_r = align_out_r.transpose(-1, -2).contiguous()        # (num_block, num_frame, num_bin, num_align)
                    align_out_i = align_out_i.transpose(-1, -2).contiguous()        # (num_block, num_frame, num_bin, num_align)

                    talign_out_r = align_out_r[:, 0, :, :]                          # (num_block, num_bin, num_align)
                    talign_out_i = align_out_i[:, 0, :, :]                          # (num_block, num_bin, num_align)

                    talign_out_r = torch.mean(talign_out_r, 2) # (num_block, num_bin)
                    talign_out_i = torch.mean(talign_out_i, 2) # (num_block, num_bin)
                    talign_out = torch.cat((talign_out_r.unsqueeze(1), talign_out_i.unsqueeze(1)), dim = 1) # ( num_block, 2, num_bin )

                    squery = talign_out[:,0,:] ** 2 + talign_out[:,1,:] ** 2   # ( num_block, num_bin )
                    squery = torch.clamp(squery, min = numerical_protection)   # ( num_block, num_bin )
                    squery = squery ** (0.5 * compressed_scale)                # ( num_block, num_bin )
                    squery = squery.unsqueeze(1)                               # ( num_block, 1, num_bin )

                    query = torch.cat((squery, query), dim = 1)                # ( num_block, num_frame, num_bin )
                    query = query.view(-1, 1, num_bin)                         # ( num_frame, 1, num_bin )

                    key = align_out[:,0,:,:] ** 2 + align_out[:,1,:,:] ** 2           # ( num_frame, num_bin, num_align )
                    key = torch.clamp(key, min = numerical_protection)                # ( num_frame, num_bin, num_align )
                    key = key ** (0.5 * compressed_scale)                             # ( num_frame, num_bin, num_align )
                    '''
                    # query: ( num_frame, 1, num_bin ), key: ( num_frame, num_bin, num_align ), value: ( num_frame, 2, num_bin, num_align )
                    align_out, att_v = fa_attention(query, key, value)                # ( num_frame, 2, num_bin ), ( num_frame, 2, num_align )
                else:
                    print("attention")
                    align_out_r = align_out_r.view(num_block, -1, self.num_align, num_bin) # (num_frame, num_align, num_bin) --> (num_block, num_frame, num_align, num_bin)
                    align_out_i = align_out_i.view(num_block, -1, self.num_align, num_bin) # (num_frame, num_align, num_bin) --> (num_block, num_frame, num_align, num_bin)

                    sequential_length = align_out_r.size(1)
                    
                    align_out_r = align_out_r.transpose(-1, -2).contiguous()        # (num_block, num_frame, num_bin, num_align)
                    align_out_i = align_out_i.transpose(-1, -2).contiguous()        # (num_block, num_frame, num_bin, num_align)

                    talign_out_r = align_out_r[:, 0, :, :]                          # (num_block, num_bin, num_align)
                    talign_out_i = align_out_i[:, 0, :, :]                          # (num_block, num_bin, num_align)

                    talign_out_r = torch.mean(talign_out_r, 2) # (num_block, num_bin)
                    talign_out_i = torch.mean(talign_out_i, 2) # (num_block, num_bin)
                    talign_out = torch.cat((talign_out_r.unsqueeze(1), talign_out_i.unsqueeze(1)), dim = 1) # ( num_block, 2, num_bin )

                    query = talign_out[:,0,:] ** 2 + talign_out[:,1,:] ** 2  # ( num_block, num_bin )
                    query = torch.clamp(query, min = numerical_protection)   # ( num_block, num_bin )
                    query = query ** (0.5 * compressed_scale)                # ( num_block, num_bin )
                    self.query = query.unsqueeze(1)                          # ( num_block, 1, num_bin )

                    align_out = talign_out                                              # ( num_block, 2, num_bin )
                    for t in torch.arange(1, sequential_length):

                        talign_out_r = align_out_r[:, t, :, :]                          # (num_block, num_bin, num_align)
                        talign_out_i = align_out_i[:, t, :, :]                          # (num_block, num_bin, num_align)

                        key = talign_out_r ** 2 + talign_out_i ** 2                     # ( num_block, num_bin, num_align )
                        key = torch.clamp(key, min = numerical_protection)              # ( num_block, num_bin, num_align )
                        key = key ** (0.5 * compressed_scale)                           # ( num_block, num_bin, num_align )

                        value = torch.cat((talign_out_r.unsqueeze(1), talign_out_i.unsqueeze(1)), dim = 1)        # (num_block, 2, num_bin, num_align)

                        # query: ( num_block, 1, num_bin ), key: ( num_block, num_bin, num_align ), value: ( num_block, 2, num_bin, num_align )
                        talign_out, tatt_v = fa_attention(self.query, key, value)   # ( num_block, 2, num_bin ), ( num_block, 2, num_align )
                        align_out = torch.cat((align_out, talign_out), dim = 0)     # ( num_frame * num_block, 2, num_bin )

                        query = talign_out[:,0,:] ** 2 + talign_out[:,1,:] ** 2  # ( num_block, num_bin )
                        query = torch.clamp(query, min = numerical_protection)   # ( num_block, num_bin )
                        query = query ** (0.5 * compressed_scale)                # ( num_block, num_bin )
                        self.query = query.unsqueeze(1)                          # ( num_block, 1, num_bin )
                    
                    if num_block > 1:
                        align_out = align_out.view(-1, num_block, 2, num_bin) # ( num_frame, num_block, 2, num_bin )
                        align_out = align_out.transpose(1, 0).contiguous()    # ( num_block, num_frame, 2, num_bin )
                        align_out = align_out.view(-1, 2, num_bin)            # ( num_block * num_frame, 2, num_bin )

            elif self.pooling_type.lower() == 'gate':
                #print("gate")
                align_fft = torch.cat((align_out_r.unsqueeze(1), align_out_i.unsqueeze(1)), dim = 1) # ( num_frame, 2, num_align, num_bin )
                align_fft = align_fft.transpose(-1, -2).contiguous()                                 # ( num_frame, 2, num_bin, num_align )
                #align_out, gate_mask = self.gate(align_fft, num_block = num_block, numerical_protection = numerical_protection, compressed_scale = compressed_scale)
                align_out, gate_mask = self.gate(align_fft, numerical_protection = numerical_protection, compressed_scale = compressed_scale)
            else:
                align_out_r = torch.mean(align_out_r, 1).squeeze() # (num_frame, num_bin)
                align_out_i = torch.mean(align_out_i, 1).squeeze() # (num_frame, num_bin)
                align_out = torch.cat((align_out_r.unsqueeze(1), align_out_i.unsqueeze(1)), dim = 1) # ( num_frame, 2, num_bin )
        else:
            align_out_r = torch.mean(align_out_r, 1).squeeze() # (num_frame, num_bin)
            align_out_i = torch.mean(align_out_i, 1).squeeze() # (num_frame, num_bin)
            align_out = torch.cat((align_out_r.unsqueeze(1), align_out_i.unsqueeze(1)), dim = 1) # ( num_frame, 2, num_bin)
        
        if DEBUG_QUANTIZE:
            with torch.no_grad():
                self.fc_min  = min(align_out.min(), self.fc_min)
                self.fc_max  = max(align_out.max(), self.fc_max)
                self.fc_mean = (align_out.mean() +  self.fc_mean) / 2.0
                self.fc_std  = (align_out.std() +  self.fc_std) / 2.0

        if self.batch_norm is not None:
            align_out = align_out.view(-1, num_bin)
            align_out = self.batch_norm(align_out)
            align_out = align_out.view(num_frame, num_dim, num_bin) # ( num_frame, 2, num_bin)

            if DEBUG_QUANTIZE:
                with torch.no_grad():
                    self.bn_min  = min(align_out.min(), self.bn_min)
                    self.bn_max  = max(align_out.max(), self.bn_max)
                    self.bn_mean = (align_out.mean() + self.bn_mean) / 2.0
                    self.bn_std  = (align_out.std() + self.bn_std) / 2.0

        if self.dropout is not None:
            align_out = self.dropout(align_out) # ( num_frame, 2, num_bin)
        return align_out

class DeepConv2DFALayer(nn.Module):
    r"""Using DeepConv2D to implement frequncy align layer for multi-direction beamforming output
    Input:  multi-direction beamforming output, shape = (num_frame, 2, num_bin, num_beam)
    Output: frequncy alignment output, shape = (num_frame, 2, num_bin)

    Args:
        num_freq_kernel: size of kernel along the feqency axis
        num_beam_kernel: size of kernel along the beamforming axis
        num_align: number of frequency alignment ways
        num_bin: number of frequency bin
        pooling_type: type of pooling the multi-alignments, avg: average pooling, max, max pooling
    """
    def __init__(self, num_freq_kernel, num_beam_kernel, num_align, num_bin = 256, bias = True, pooling_type = 'avg', batch_norm = False, dropout = 0.0, fixed_align = False):
        super(DeepConv2DFALayer, self).__init__()
        
        self.num_beam_kernel     = num_beam_kernel
        self.num_freq_kernel     = num_freq_kernel
        self.num_align           = num_align
        self.num_bin             = num_bin
        self.bias                = bias
        self.pooling_type        = pooling_type
        self.fixed_align         = fixed_align
        
        #self.batch_norm = nn.LayerNorm(num_bin) if batch_norm else None
        self.batch_norm  = nn.BatchNorm1d(num_bin) if batch_norm else None
        self.dropout     = nn.Dropout(dropout) if dropout > 0.0 else None
        
        kernel_size   = (num_freq_kernel, num_beam_kernel)
        stride        = (1, num_beam_kernel)
        padding       = (int(np.floor(num_freq_kernel / 2)), 0)
        self.conv2dfa = nn.Conv2d(in_channels = 1, out_channels = num_align, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)

        kernel_size   = [3, 3]
        stride        = [1, 1]
        padding       = [1, 1]
        self.cconv1    = CConv2dLayer(in_channels = num_align, out_channels = num_align, kernel_size = kernel_size, stride = stride, padding = padding, dilation = 1)
        self.cconv2    = CConv2dLayer(in_channels = num_align, out_channels = num_align, kernel_size = 1, stride = 1, padding = 0, dilation = 1)
        if self.pooling_type is not None and self.pooling_type.lower() == 'conv1x1':
            self.cconv_pooling = CConv2dLayer(in_channels = num_align, out_channels = 1, kernel_size = 1, stride = 1, padding = 0, dilation = 1)
        else:
            self.cconv_pooling = None

        if self.pooling_type is not None and self.pooling_type.lower() == 'gate': # pow:    ( num_frame, 1, num_bin, num_align) 
            self.gate = Gate(channels=[12, 24, 48], num_align = num_align)
        else:
            self.gate = None

        if DEBUG_QUANTIZE:
            self.fc_min        = 100000000.0
            self.fc_max        = -100000000.0
            self.fc_mean       = 0.0
            self.fc_std        = 0.0
            self.bn_min        = 100000000
            self.bn_max        = -100000000
            self.bn_mean       = 0.0
            self.bn_std        = 0.0

    def get_trainable_params(self, print_info = False):
        if print_info:
            print("####### Trainable Parames of frequncy_align_layer")
        if print_info:
                print("frequency_align = ")

        frequency_align_param_names = []
        weights, biases = [], []
        if self.gate is not None:
            for name, param in self.gate.named_parameters():
                frequency_align_param_names.append(name)
                if 'bias' in name:
                    biases += [param]
                else:
                    weights += [param]
        if not self.fixed_align:
            for name, param in self.conv2dfa.named_parameters():
                frequency_align_param_names.append(name)
                if 'bias' in name:
                    biases += [param]
                else:
                    weights += [param]
            for name, param in self.cconv1.named_parameters():
                frequency_align_param_names.append(name)
                if 'bias' in name:
                    biases += [param]
                else:
                    weights += [param]
            for name, param in self.cconv2.named_parameters():
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

    def forward(self, input, attractor = None, num_block = 1, numerical_protection = 1.0e-13, compressed_scale = 0.3):
        # input  = ( num_frame, 2, num_bin, num_beam ) conv num_align * (num_freq_kernel, num_beam_kernel)
        # output = ( num_frame, 2, num_bin )
        num_frame, num_dim, num_bin, num_beam = input.size() # (num_frame, 2, num_bin, num_beam)
        assert num_dim == 2 and num_bin == self.num_bin and num_beam == self.num_beam_kernel, "illegal shape for input, the required input shape is (%d, 2, %d, %d), but got (%d, %s, %d, %d)" % (num_frame, self.num_bin, self.num_align, num_frame, num_dim, num_bin, num_beam)
        
        input_r = input[:, 0, :, :].unsqueeze(1) # (num_frame, 1, num_bin, num_beam)
        input_i = input[:, 1, :, :].unsqueeze(1) # (num_frame, 1, num_bin, num_beam)

        if self.fixed_align:
             with torch.no_grad():
                align_out_r = self.conv2dfa(input_r).squeeze()     # (num_frame, num_align, num_bin, 1) --> (num_frame, num_align, num_bin)
                align_out_i = self.conv2dfa(input_i).squeeze()     # (num_frame, num_align, num_bin, 1) --> (num_frame, num_align, num_bin)
        else:
            align_out_r = self.conv2dfa(input_r).squeeze()     # (num_frame, num_align, num_bin, 1) --> (num_frame, num_align, num_bin)
            align_out_i = self.conv2dfa(input_i).squeeze()     # (num_frame, num_align, num_bin, 1) --> (num_frame, num_align, num_bin)

        # (num_frame, num_align, num_bin) --> (num_block, num_align, num_bin, num_frame)
        align_out_r = align_out_r.view(num_block, -1, self.num_align, num_bin) # (num_block, num_frame, num_align, num_bin)
        align_out_i = align_out_i.view(num_block, -1, self.num_align, num_bin) # (num_block, num_frame, num_align, num_bin)

        align_out_r = align_out_r.permute([0, 2, 3, 1]).contiguous()      # (num_block, num_align, num_bin, num_frame)
        align_out_i = align_out_i.permute([0, 2, 3, 1]).contiguous()      # (num_block, num_align, num_bin, num_frame)

        if self.fixed_align:
             with torch.no_grad():
                align_out_r, align_out_i = self.cconv1(align_out_r, align_out_i)   # (num_block, num_align, num_bin, num_frame)
                align_out_r, align_out_i = self.cconv2(align_out_r, align_out_i)   # (num_block, num_align, num_bin, num_frame)
        else:
            align_out_r, align_out_i = self.cconv1(align_out_r, align_out_i)   # (num_block, num_align, num_bin, num_frame)
            align_out_r, align_out_i = self.cconv2(align_out_r, align_out_i)   # (num_block, num_align, num_bin, num_frame)

        if self.cconv_pooling is not None:
            align_out_r, align_out_i = self.cconv_pooling(align_out_r, align_out_i) # (num_block, 1, num_bin, num_frame)
            
            align_out_r = align_out_r.permute([0, 3, 1, 2]).contiguous() # (num_block, num_frame, 1, num_bin)
            align_out_i = align_out_i.permute([0, 3, 1, 2]).contiguous() # (num_block, num_frame, 1, num_bin)

            align_out_r = align_out_r.view(-1, 1, num_bin)  # (num_frame, 1, num_bin)
            align_out_i = align_out_i.view(-1, 1, num_bin)  # (num_frame, 1, num_bin)
            align_out   = torch.cat((align_out_r, align_out_i), dim = 1) # ( num_frame, 2, num_bin )
        else:
            align_out_r = align_out_r.permute([0, 3, 1, 2]).contiguous() # (num_block, num_align, num_bin, num_frame) -> (num_block, num_frame, num_align, num_bin)
            align_out_i = align_out_i.permute([0, 3, 1, 2]).contiguous() # (num_block, num_align, num_bin, num_frame) -> (num_block, num_frame, num_align, num_bin)
            align_out_r = align_out_r.view(-1, self.num_align, num_bin)  # (num_frame, num_align, num_bin)
            align_out_i = align_out_i.view(-1, self.num_align, num_bin)  # (num_frame, num_align, num_bin)
            if self.pooling_type is not None:
                if self.pooling_type.lower() == 'avg': 
                    align_out_r = torch.mean(align_out_r, 1).squeeze() # (num_frame, num_bin)
                    align_out_i = torch.mean(align_out_i, 1).squeeze() # (num_frame, num_bin)
                    align_out = torch.cat((align_out_r.unsqueeze(1), align_out_i.unsqueeze(1)), dim = 1) # ( num_frame, 2, num_bin )
                elif self.pooling_type.lower() == 'max':
                    align_pow = align_out_r ** 2 + align_out_i ** 2  # ( num_frame, num_align, num_bin )
                    align_idx = torch.argmax(align_pow, dim = 1, keepdim = True)                  # (num_frame, 1, num_bin)
                    align_out_r = torch.gather(align_out_r, dim = 1, index = align_idx).squeeze() # (num_frame, num_bin)
                    align_out_i = torch.gather(align_out_i, dim = 1, index = align_idx).squeeze() # (num_frame, num_bin)
                    align_out = torch.cat((align_out_r.unsqueeze(1), align_out_i.unsqueeze(1)), dim = 1) # ( num_frame, 2, num_bin )
                elif self.pooling_type.lower() == 'attention':                                                
                    align_out = torch.cat((align_out_r.unsqueeze(1), align_out_i.unsqueeze(1)), dim = 1) # (num_frame, 1, num_align, num_bin)
                    align_out = align_out.transpose(-1, -2)                                              #( num_frame, 2, num_bin, num_align)
                    value     = align_out     
                    if attractor is not None:
                        '''
                        query = attractor[:,0,:,:] ** 2 + attractor[:,1,:,:] ** 2 # ( num_block, num_bin, num_frame )
                        query = torch.clamp(query, min = numerical_protection)            # ( num_block, num_bin, num_frame )
                        query = query ** (0.5 * compressed_scale)                         # ( num_block, num_bin, num_frame )
                        query = query.transpose(-2, -1).contiguous()                      # ( num_block, num_frame, num_bin )
                        query = query.view(-1, 1, num_bin)                                # ( num_frame, 1, num_bin )
                        '''
                        query = attractor[:,0,:,:-1] ** 2 + attractor[:,1,:,:-1] ** 2 # ( num_block, num_bin, num_frame )
                        query = torch.clamp(query, min = numerical_protection)            # ( num_block, num_bin, num_frame )
                        query = query ** (0.5 * compressed_scale)                         # ( num_block, num_bin, num_frame )
                        query = query.transpose(-1, -2).contiguous()                      # ( num_block, num_frame, num_bin )

                        align_out_r = align_out_r.view(num_block, -1, self.num_align, num_bin) # (num_frame, num_align, num_bin) --> (num_block, num_frame, num_align, num_bin)
                        align_out_i = align_out_i.view(num_block, -1, self.num_align, num_bin) # (num_frame, num_align, num_bin) --> (num_block, num_frame, num_align, num_bin)

                        align_out_r = align_out_r.transpose(-1, -2).contiguous()        # (num_block, num_frame, num_bin, num_align)
                        align_out_i = align_out_i.transpose(-1, -2).contiguous()        # (num_block, num_frame, num_bin, num_align)

                        talign_out_r = align_out_r[:, 0, :, :]                          # (num_block, num_bin, num_align)
                        talign_out_i = align_out_i[:, 0, :, :]                          # (num_block, num_bin, num_align)

                        talign_out_r = torch.mean(talign_out_r, 2) # (num_block, num_bin)
                        talign_out_i = torch.mean(talign_out_i, 2) # (num_block, num_bin)
                        talign_out = torch.cat((talign_out_r.unsqueeze(1), talign_out_i.unsqueeze(1)), dim = 1) # ( num_block, 2, num_bin )

                        squery = talign_out[:,0,:] ** 2 + talign_out[:,1,:] ** 2   # ( num_block, num_bin )
                        squery = torch.clamp(squery, min = numerical_protection)   # ( num_block, num_bin )
                        squery = squery ** (0.5 * compressed_scale)                # ( num_block, num_bin )
                        squery = squery.unsqueeze(1)                               # ( num_block, 1, num_bin )

                        query = torch.cat((squery, query), dim = 1)                # ( num_block, num_frame, num_bin )
                        query = query.view(-1, 1, num_bin)                         # ( num_frame, 1, num_bin )

                        key = align_out[:,0,:,:] ** 2 + align_out[:,1,:,:] ** 2           # ( num_frame, num_bin, num_align )
                        key = torch.clamp(key, min = numerical_protection)                # ( num_frame, num_bin, num_align )
                        key = key ** (0.5 * compressed_scale)                             # ( num_frame, num_bin, num_align )

                        align_out, att_v = fa_attention(query, key, value)                # ( num_frame, 2, num_bin ), ( num_frame, 2, num_align )
                    else:
                        # (num_frame, num_align, num_bin) --> (num_block, num_align, num_bin, num_frame)
                        align_out_r = align_out_r.view(num_block, -1, self.num_align, num_bin) # (num_frame, num_align, num_bin) --> (num_block, num_frame, num_align, num_bin)
                        align_out_i = align_out_i.view(num_block, -1, self.num_align, num_bin) # (num_frame, num_align, num_bin) --> (num_block, num_frame, num_align, num_bin)

                        sequential_length = align_out_r.size(1)

                        align_out_r = align_out_r.transpose(-1, -2).contiguous()        # (num_block, num_frame, num_bin, num_align)
                        align_out_i = align_out_i.transpose(-1, -2).contiguous()        # (num_block, num_frame, num_bin, num_align)

                        talign_out_r = align_out_r[:, 0, :, :]                          # (num_block, num_bin, num_align)
                        talign_out_i = align_out_i[:, 0, :, :]                          # (num_block, num_bin, num_align)

                        talign_out_r = torch.mean(talign_out_r, 2) # (num_block, num_bin)
                        talign_out_i = torch.mean(talign_out_i, 2) # (num_block, num_bin)
                        talign_out = torch.cat((talign_out_r.unsqueeze(1), talign_out_i.unsqueeze(1)), dim = 1) # ( num_block, 2, num_bin )

                        query = talign_out[:,0,:] ** 2 + talign_out[:,1,:] ** 2  # ( num_block, num_bin )
                        query = torch.clamp(query, min = numerical_protection)   # ( num_block, num_bin )
                        query = query ** (0.5 * compressed_scale)                # ( num_block, num_bin )
                        self.query = query.unsqueeze(1)                          # ( num_block, 1, num_bin )

                        align_out = talign_out                                              # ( num_block, 2, num_bin )
                        for t in torch.arange(1, sequential_length):

                            talign_out_r = align_out_r[:, t, :, :]                          # (num_block, num_bin, num_align)
                            talign_out_i = align_out_i[:, t, :, :]                          # (num_block, num_bin, num_align)

                            key = talign_out_r ** 2 + talign_out_i ** 2                     # ( num_block, num_bin, num_align )
                            key = torch.clamp(key, min = numerical_protection)              # ( num_block, num_bin, num_align )
                            key = key ** (0.5 * compressed_scale)                           # ( num_block, num_bin, num_align )

                            value = torch.cat((talign_out_r.unsqueeze(1), talign_out_i.unsqueeze(1)), dim = 1)        # (num_block, 2, num_bin, num_align)

                            # query: ( num_block, 1, num_bin ), key: ( num_block, num_bin, num_align ), value: ( num_block, 2, num_bin, num_align )
                            talign_out, tatt_v = fa_attention(self.query, key, value)   # ( num_block, 2, num_bin ), ( num_block, 2, num_align )
                            align_out = torch.cat((align_out, talign_out), dim = 0)     # ( num_frame, 2, num_bin )

                            query = talign_out[:,0,:] ** 2 + talign_out[:,1,:] ** 2  # ( num_block, num_bin )
                            query = torch.clamp(query, min = numerical_protection)   # ( num_block, num_bin )
                            query = query ** (0.5 * compressed_scale)                # ( num_block, num_bin )
                            self.query = query.unsqueeze(1)                          # ( num_block, 1, num_bin )
                        
                        if num_block > 1:
                            align_out = align_out.view(-1, num_block, 2, num_bin) # ( num_frame, num_block, 2, num_bin )
                            align_out = align_out.transpose(1, 0).contiguous()    # ( num_block, num_frame, 2, num_bin )
                            align_out = align_out.view(-1, 2, num_bin)            # ( num_block * num_frame, 2, num_bin )

                elif self.pooling_type.lower() == 'gate':
                    align_fft = torch.cat((align_out_r.unsqueeze(1), align_out_i.unsqueeze(1)), dim = 1) # ( num_frame, 2, num_align, num_bin )
                    align_fft = align_fft.transpose(-1, -2).contiguous()                                 # ( num_frame, 2, num_bin, num_align )
                    align_out, gate_mask = self.gate(align_fft, numerical_protection = numerical_protection, compressed_scale = compressed_scale)
                else:
                    align_out_r = torch.mean(align_out_r, 1).squeeze() # (num_frame, num_bin)
                    align_out_i = torch.mean(align_out_i, 1).squeeze() # (num_frame, num_bin)
                    align_out = torch.cat((align_out_r.unsqueeze(1), align_out_i.unsqueeze(1)), dim = 1) # ( num_frame, 2, num_bin )
            else:
                align_out_r = torch.mean(align_out_r, 1).squeeze() # (num_frame, num_bin)
                align_out_i = torch.mean(align_out_i, 1).squeeze() # (num_frame, num_bin)
                align_out = torch.cat((align_out_r.unsqueeze(1), align_out_i.unsqueeze(1)), dim = 1) # ( num_frame, 2, num_bin)
        
        if DEBUG_QUANTIZE:
            with torch.no_grad():
                self.fc_min  = min(align_out.min(), self.fc_min)
                self.fc_max  = max(align_out.max(), self.fc_max)
                self.fc_mean = (align_out.mean() +  self.fc_mean) / 2.0
                self.fc_std  = (align_out.std() +  self.fc_std) / 2.0

        if self.batch_norm is not None:
            align_out = align_out.view(-1, num_bin)
            align_out = self.batch_norm(align_out)
            align_out = align_out.view(num_frame, num_dim, num_bin) # ( num_frame, 2, num_bin)

            if DEBUG_QUANTIZE:
                with torch.no_grad():
                    self.bn_min  = min(align_out.min(), self.bn_min)
                    self.bn_max  = max(align_out.max(), self.bn_max)
                    self.bn_mean = (align_out.mean() + self.bn_mean) / 2.0
                    self.bn_std  = (align_out.std() + self.bn_std) / 2.0

        if self.dropout is not None:
            align_out = self.dropout(align_out) # ( num_frame, 2, num_bin)
        return align_out
