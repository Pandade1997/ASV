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
from .spatial_filtering_layer import Filter
from .beamformer import Beamformor

class DPR(nn.Module):
    r"""Implement Directional Coherence Function to extract the Directional Features
    Args:
        num_beam: number of beamforming filters
        num_bin: number of frequency bin
        num_channel: number of channels
    """
    def __init__(self, num_beam, num_bin, num_channel, targ_bf_weight = None, null_bf_weight = None, fix = True):
        super(DPR, self).__init__()
        
        self.num_beam    = num_beam
        self.num_bin     = num_bin
        self.num_channel = num_channel
        
        self.targ_bf = Beamformor(num_beam = num_beam, num_bin = num_bin, num_channel = num_channel, weight_init = targ_bf_weight, fix=fix)

    def forward(self, input, beam_id):
        # input: ( num_block * num_frame, 2, num_bin, num_channel )
        
        num_block   = beam_id.size(0)
        input       = input.view([num_block, -1, 2, self.num_bin, self.num_channel])    # (num_block, num_frame, 2, num_bin, num_channel )
        
        targ_bf_out_list = []
        for n in torch.arange(num_block):
            targ_bf_out = self.targ_bf(input[n], beam_id[n]) # (num_frame, 2, num_bin, 1)
            targ_bf_out = targ_bf_out.squeeze()              # (num_frame, 2, num_bin)
            targ_bf_out = targ_bf_out.unsqueeze(0)           # (1, num_frame, 2, num_bin)
            targ_bf_out_list.append(targ_bf_out)

        targ_bf_out   = torch.cat(targ_bf_out_list, 0)        # (num_block, num_frame, 2, num_bin)

        targ_bf_power = targ_bf_out[:, :, 0, :] ** 2 + targ_bf_out[:, :, 1, :] ** 2 # (num_block, num_frame, num_bin)

        sum_bf_power  = targ_bf_power                                               # (num_block, num_frame, num_bin)
        for tdoa in torch.arange(self.num_beam):
            for n in torch.arange(num_block):
                if tdoa == beam_id[n]:
                    continue
                bf_out = self.targ_bf(input[n], tdoa) # (num_frame, 2, num_bin, 1)
                bf_out = bf_out.squeeze()             # (num_frame, 2, num_bin)
                
                bf_power = bf_out[:, 0, :] ** 2 + bf_out[:, 1, :] ** 2 # (num_frame, num_bin)

                sum_bf_power[n, :, :] = sum_bf_power[n, :, :] + bf_power # (num_block, num_frame, num_bin)

        dpr = targ_bf_power / sum_bf_power # (num_block, num_frame, num_bin)

        return dpr, targ_bf_out.squeeze(4) # (num_block, num_frame, num_bin), (num_block, num_frame, 2, num_bin)

