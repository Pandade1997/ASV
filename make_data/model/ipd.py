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

class IPD(nn.Module):
    r"""Implement Directional Coherence Function to extract the Directional Features
    Args:
        num_beam: number of beamforming filters
        num_bin: number of frequency bin
        num_channel: number of channels
    """
    def __init__(self, num_bin, do_IPD = True, do_cosIPD = True, do_sinIPD = True, do_ILD = True):
        super(IPD, self).__init__()

        self.num_bin   = num_bin
        self.do_IPD    = do_IPD
        self.do_cosIPD = do_cosIPD
        self.do_sinIPD = do_sinIPD
        self.do_ILD    = do_ILD

        self.num_feat = int(do_IPD + do_cosIPD + do_sinIPD + do_ILD)

    def get_size(self, pair_id):
        num_pair    = len(pair_id)
        return self.num_feat * num_pair * self.num_bin

    def forward(self, input, pair_id):
        # input:   ( num_block * num_frame, 2, num_bin, num_channel )
        # tdoa:    ( num_block )
        # pair_id: ( num_pair, 2 )

        IPD_list    = []
        cosIPD_list = []
        sinIPD_list = []
        ILD_list    = []

        num_pair    = len(pair_id)
        for n in torch.arange(num_pair):
            p, q = pair_id[n]

            Yp   = input[:, :, :, :, p] # (num_block * num_frame, 2, num_bin)
            Yq   = input[:, :, :, :, q] # (num_block * num_frame, 2, num_bin)
            
            pq_theta = torch.angle(Yp) - torch.angle(Yq) # (num_block * num_frame, num_bin)

            if self.do_IPD:
                ipd = torch.mod(pq_theta + math.pi, 2 * math.pi) - math.pi # (num_block * num_frame, num_bin)
                IPD_list.append(ipd)
            
            if self.do_cosIPD:
                cosipd = torch.cos(pq_theta)        # (num_block * num_frame, num_bin)
                cosIPD_list.append(cosipd)
            
            if self.do_sinIPD:
                sinipd = torch.sin(pq_theta)        # (num_block * num_frame, num_bin)
                sinIPD_list.append(sinipd)

            if self.do_ILD:
                Yp_spect = (Yp[:, 0, :] ** 2 + Yp[:, 1, :] ** 2) ** 0.5 # (num_block * num_frame, num_bin)
                Yq_spect = (Yq[:, 0, :] ** 2 + Yq[:, 1, :] ** 2) ** 0.5 # (num_block * num_frame, num_bin)
                ild = torch.log( Yp_spect / ( Yq_spect + 1.0e-13 ) )
                ILD_list.append(ild)
        
        out = None
        if len(IPD_list) > 0:
            ipd = torch.cat(IPD_list, 0)       # (num_pair, num_block * num_frame, num_bin)
            if out is None:
                out = ipd
            else:
                out = torch.cat((out, ipd), -1)
        else:
            ipd = None
        
        if len(cosIPD_list) > 0:
            cosipd = torch.cat(cosIPD_list, 0) # (num_pair, num_block * num_frame, num_bin)
            if out is None:
                out = cosipd
            else:
                out = torch.cat((out, cosipd), -1)
        else:
            cosipd = None

        if len(sinIPD_list) > 0:
            sinipd = torch.cat(sinIPD_list, 0) # (num_pair, num_block * num_frame, num_bin)
            if out is None:
                out = sinipd
            else:
                out = torch.cat((out, sinipd), -1)
        else:
            sinipd = None

        if len(ILD_list) > 0:
            ild = torch.cat(ILD_list, 0)       # (num_pair, num_block * num_frame, num_bin)
            if out is None:
                out = ild
            else:
                out = torch.cat((out, ild), -1)
        else:
            ild = None
        
        return out # (num_pair, num_block * num_frame, num_bin * num)
