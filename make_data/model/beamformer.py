import os
import sys

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))


import torch
import torch.nn as nn
import functools
import numpy as np
import math
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn.parameter import Parameter
from .functions import to_cuda
from .spatial_filtering_layer import Filter
from .model_config import DEBUG_QUANTIZE

#############################################################################################################
##################################### Complex multi-channel beamformor Layer ################################
#############################################################################################################
class Beamformor(nn.Module):
    r"""Implement directionary beamformor to enhance given directionary areas, all operators on complex fft coefficients of audios
    Args:
        num_beam: number of beamforming filters
        num_bin: number of frequency bin
        num_channel: number of channels
    """
    def __init__(self, num_beam, num_bin, num_channel, weight_init = None, fix = True):
        super(Beamformor, self).__init__()
        
        self.num_beam    = num_beam
        self.num_bin     = num_bin
        self.num_channel = num_channel
        self.fix         = fix
        
        filters = []
        for n in range(num_beam):
            if weight_init is not None:
                init_weight = torch.from_numpy(weight_init[n][np.newaxis, :, :, :]) # (1, 2, num_bin, num_channel)
            else:
                init_weight = None
            filter = Filter(num_bin = self.num_bin, num_channel = self.num_channel, init_weight = init_weight, init_bias = None, bias = False, fix = self.fix)
            filters.append(filter)
        self.filter = nn.ModuleList(filters)

    def forward(self, input, beam_id = None):
        # input: ( num_block * num_frame, 2, num_bin, num_channel )
        # beam_id: a value or (num_block) or (num_block, num_frame) or None
        if beam_id is None:
            output_list = []
            for beamforming in self.filter:
                output = beamforming(input)         # (num_frame, 2, num_bin)
                output = output.unsqueeze(3)        # (num_frame, 2, num_bin, 1)
                output_list.append(output)          
            output = torch.cat(output_list, 3)      # (num_frame, 2, num_bin, num_beam)
            return output
        
        if torch.is_tensor(beam_id):
            if not beam_id.is_cuda:
                beam_id = to_cuda(self, beam_id)

            num_block_frame = input.size(0)
            num_bin         = input.size(2)

            beam_id    = torch.flatten(beam_id) # (num_block )
            num_repeat = int(num_block_frame / beam_id.size(0))
            if num_repeat > 1:
                beam_id = beam_id.repeat(num_repeat) # (num_block * num_frame)
            targ_out  = to_cuda( self, torch.zeros( num_block_frame, 2, num_bin, requires_grad = (not self.fix) ) ) # (num_block * num_frame, 2, num_bin)
            for beam in torch.arange(self.num_beam):
                beam_idx = torch.nonzero(beam_id == beam, as_tuple = True)[0]       # (num_block * num_frame)
                if beam_idx.size(0) > 0:
                    beam_x   = torch.index_select(input, dim = 0, index = beam_idx) # (num_block * num_frame, 2, num_bin, num_channel)
                    targ_out[beam_idx, :, :]  = self.filter[beam](beam_x)           # (num_block * num_frame, 2, num_bin)
            output  = targ_out.unsqueeze(3)          # (num_block * num_frame, 2, num_bin, 1)
        else:
            output = self.filter[beam_id](input)     # (num_block * num_frame, 2, num_bin)
            output = output.unsqueeze(3)             # (num_block * num_frame, 2, num_bin, 1)
        
        return output # (num_block * num_frame, 2, num_bin, 1)

class NullBeamformor(nn.Module):
    r"""Implement directionary beamformor with null directionary areas, all operators on complex fft coefficients of audios
    Args:
        num_beam: number of beamforming filters
        num_bin: number of frequency bin
        num_channel: number of channels
    """
    def __init__(self, num_beam, num_null, num_bin, num_channel, weight_init = None, fix = True):
        super(NullBeamformor, self).__init__()
        
        self.num_beam    = num_beam
        self.num_null    = num_null
        self.num_bin     = num_bin
        self.num_channel = num_channel
        self.fix         = fix
        
        filters = []
        for n in range(num_beam * num_null):
            if weight_init is not None:
                init_weight = torch.from_numpy(weight_init[n][np.newaxis, :, :, :]) # (1, 2, num_bin, num_channel)
            else:
                init_weight = None
            filter = Filter(num_bin = self.num_bin, num_channel = self.num_channel, init_weight = init_weight, init_bias = None, bias = False, fix = self.fix)
            filters.append(filter)
        self.filter = nn.ModuleList(filters)

    def forward(self, input, beam_id = None):
        # input: ( num_block * num_frame, 2, num_bin, num_channel )
        # beam_id: a value or (num_block) or (num_block, num_frame) or None
        if beam_id is None:
            output_list = []
            for beamforming in self.filter:
                output = beamforming(input)         # (num_frame, 2, num_bin)
                output = output.unsqueeze(3)        # (num_frame, 2, num_bin, 1)
                output_list.append(output)          
            output = torch.cat(output_list, 3)      # (num_frame, 2, num_bin, num_beam * num_null)
            return output
        
        if torch.is_tensor(beam_id):
            num_block_frame = input.size(0)
            num_bin         = input.size(2)

            beam_id    = torch.flatten(beam_id)                 # (num_block)
            num_repeat = int(num_block_frame / beam_id.size(0)) 
            if num_repeat > 1:
                beam_id = beam_id.repeat(num_repeat)            # (num_block * num_frame)

            targ_out  = to_cuda(self, torch.zeros(num_block_frame, 2, num_bin, self.num_null, requires_grad = (not self.fix))) # (num_block * num_frame, 2, num_bin, num_null)
            for beam in torch.arange(self.num_beam):
                beam_idx = torch.nonzero(beam_id == beam, as_tuple = True)[0]       # (num_block * num_frame)
                if beam_idx.size(0) > 0:
                    beam_x   = torch.index_select(input, dim = 0, index = beam_idx) # (num_block * num_frame, 2, num_bin, num_channel)

                    start_beam = beam * self.num_null
                    end_beam   = start_beam + self.num_null
                    output_list = []
                    for n in torch.arange(start_beam, end_beam):
                        beamforming = self.filter[n]
                        output = beamforming(beam_x)         # (num_frame, 2, num_bin)
                        output = output.unsqueeze(3)         # (num_frame, 2, num_bin, 1)
                        output_list.append(output)      
                    output = torch.cat(output_list, 3)       # (num_frame, 2, num_bin, num_null)
                    targ_out[beam_idx, :, :, :]  = output    # (num_block * num_frame, 2, num_bin, num_null)
        else:
            start_beam = beam_id * self.num_null
            end_beam   = start_beam + self.num_null

            targ_out_list = []
            for n in torch.arange(start_beam, end_beam):
                beamforming = self.filter[n]
                targ_out = beamforming(input)         # (num_frame, 2, num_bin)
                targ_out = targ_out.unsqueeze(3)      # (num_frame, 2, num_bin, 1)
                targ_out_list.append(targ_out)      
            targ_out = torch.cat(targ_out_list, 3)    # (num_frame, 2, num_bin, num_null)
        return targ_out                               # (num_block * num_frame, 2, num_bin, num_null)
