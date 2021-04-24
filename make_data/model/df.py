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
import math

class DF(nn.Module):
    r"""Implement Directional Coherence Function to extract the Directional Features
    Args:
        num_beam: number of beamforming filters
        num_bin: number of frequency bin
        num_channel: number of channels
    """
    def __init__(self, num_bin, num_channel, mic_pos, sound_speed = 340.0, sample_rate = 16000.0):
        super(DF, self).__init__()
        
        self.num_bin     = num_bin
        self.num_channel = num_channel
        self.sound_speed = sound_speed
        self.sample_rate = sample_rate
        self.nFFT        = ( num_bin - 1 ) * 2

        self.mic_dist    = torch.zeros(shape = [self.num_channel, self.num_channel])
        self.const_theta = torch.zeros(shape = [self.num_channel, self.num_channel])

        for i in torch.arange(self.num_channel):
            for j in torch.arange(self.num_channel):
                x1, y1                 = mic_pos[i]
                x2, y2                 = mic_pos[j]
                self.mic_dist[i, j]    = ((x2 - x1) ** 2 + ( y2 - y1) ** 2) ** 0.5
                self.const_theta[i, j] = 2.0 * math.pi * ( (self.sample_rate * self.mic_dist[i, j]) / self.sound_speed ) / (self.nFFT)

    def forward(self, input, tdoas, pair_id):
        # input:   ( num_block * num_frame, 2, num_bin, num_channel )
        # tdoa:    ( num_block )
        # pair_id: ( num_pair, 2 )

        num_block   = tdoas.size(0)
        num_pair    = len(pair_id)
        input       = input.view([num_block, -1, 2, self.num_bin, self.num_channel])    # (num_block, num_frame, 2, num_bin, num_channel )
        num_frame   = input.size(1)

        DF   = to_cuda(torch.zeros(num_block, num_frame, self.num_bin)) # (num_block, num_frame, num_bin)

        freq = ( 8000.0 / ( self.num_bin - 1 ) ) * torch.arange(0, self.num_bin)
        for b in torch.arange(num_block):

            theta = freq * torch.cos(tdoas[b] / 180.0)  # (self.num_bin)
            for n in torch.arange(num_pair):
                p, q = pair_id[n]
                Yp   = input[:, :, :, :, p] # (num_block, num_frame, 2, num_bin)
                Yq   = input[:, :, :, :, q] # (num_block, num_frame, 2, num_bin)
                
                var_theta = theta * self.const_theta[p, q]       # (self.num_bin)
                var_theta = var_theta.view([1, 1, self.num_bin]) # (1, 1, self.num_bin)

                pq_theta  = torch.angle(Yp) - torch.angle(Yq) # (num_block, num_frame, num_bin)

                DF[b, :, :] += torch.cos(pq_theta - var_theta)
            
            DF[b, :, :] = DF[b, :, :] / num_pair 

        return DF # (num_block, num_frame, num_bin)

