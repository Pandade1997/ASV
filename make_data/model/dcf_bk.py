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
from .beamformer import Beamformor, NullBeamformor

class DCF(nn.Module):
    r"""Implement Directional Coherence Function to extract the Directional Features
    Args:
        num_beam: number of beamforming filters
        num_bin: number of frequency bin
        num_channel: number of channels
    """
    def __init__(self, num_beam, num_null, num_bin, num_channel, targ_bf_weight = None, null_bf_weight = None):
        super(DCF, self).__init__()
        
        self.num_beam    = num_beam
        self.num_null    = num_null
        self.num_bin     = num_bin
        self.num_channel = num_channel
        
        self.targ_bf = Beamformor(num_beam = num_beam, num_bin = num_bin, num_channel = num_channel, weight_init = targ_bf_weight, fix=True)
        self.null_bf = NullBeamformor(num_beam = num_beam, num_null = num_null, num_bin = num_bin, num_channel = num_channel, weight_init = null_bf_weight, fix=True)
        
        self.num_null    = num_null - 1

    def forward(self, input, num_block = 1, beam_id = 0, alpha = 0.35, Low_Freq = 5, High_Freq = 70):
        # input: ( num_block * num_frame, 2, num_bin, num_channel )
        #assert beam_id >= 0 and beam_id <= self.num_beam, "illegal beam_id, the required beam_id shape is [0, %d], but got %d" % (self.num_beam, beam_id)

        ##targ_bf_out = self.targ_bf(input, beam_id) # (num_block * num_frame, 2, num_bin, 1)
        ##null_bf_out = self.null_bf(input, beam_id) # (num_block * num_frame, 2, num_bin, num_null)
        
        bf_out = self.null_bf(input, beam_id)           # (num_block * num_frame, 2, num_bin, num_null)
        targ_bf_out = bf_out[:, :, :, 0:1].unsqueeze(3) # (num_block * num_frame, 2, num_bin, 1)
        null_bf_out = bf_out[:, :, :, 1:]               # (num_block * num_frame, 2, num_bin, num_null)

        targ_bf_out = targ_bf_out.view([num_block, -1, 2, self.num_bin, 1])             # (num_block, num_frame, 2, num_bin, 1)
        null_bf_out = null_bf_out.view([num_block, -1, 2, self.num_bin, self.num_null]) # (num_block, num_frame, 2, num_bin, num_null)
        input       = input.view([num_block, -1, 2, self.num_bin, self.num_channel])    # (num_block, num_frame, 2, num_bin, num_channel )

        num_frame   = targ_bf_out.size(1)
        dcf = to_cuda(self, torch.zeros((num_block, num_frame, self.num_bin, self.num_null), dtype = torch.float32)) # (num_block, num_frame, num_bin, num_null)

        t = 0
        t_input       = input[:, t, :, :, :]            # ( num_block, 2, num_bin, num_channel )
        t_targ_bf_out = targ_bf_out[:, t, :, :, :]      # ( num_block, 2, num_bin, 1 )
        t_null_bf_out = null_bf_out[:, t, :, :, :]      # ( num_block, 2, num_bin, num_null )
        
        t_phi_r = (1.0 - alpha) * t_targ_bf_out[:, 0, :, :] * t_null_bf_out[:, 0, :, :] + t_targ_bf_out[:, 1, :, :] * t_null_bf_out[:, 1, :, :] #(num_block, num_bin, num_null)
        t_phi_i = (1.0 - alpha) * t_targ_bf_out[:, 1, :, :] * t_null_bf_out[:, 0, :, :] - t_targ_bf_out[:, 0, :, :] * t_null_bf_out[:, 1, :, :] #(num_block, num_bin, num_null)

        t_psd_out = (1.0 - alpha) * torch.mean((t_input[:,0,:,:] ** 2 + t_input[:,1,:,:] ** 2), dim = -1, keepdim = True) # ( num_block, num_bin, 1)

        t_phi   = (t_phi_r ** 2 + t_phi_i ** 2) ** 0.5  # (num_block, num_bin, num_null)
        t_dcf   = t_phi / t_psd_out                     # (num_block, 1, num_bin, num_null)
        t_dcf   = torch.clamp(t_dcf, min = 0.01, max = 1.0)
        dcf[:, t, :, :] = t_dcf.unsqueeze(1)            # (num_block, 1, num_bin, num_null)
        
        for t in torch.arange(1, num_frame):

            t_input       = input[:, t, :, :, :]        # ( num_block, 2, num_bin, num_channel )
            t_targ_bf_out = targ_bf_out[:, t, :, :, :]  # ( num_block, 2, num_bin, 1 )
            t_null_bf_out = null_bf_out[:, t, :, :, :]  # ( num_block, 2, num_bin, num_null )

            t_phi_r = alpha * t_phi_r + (1.0 - alpha) * (t_targ_bf_out[:, 0, :, :] * t_null_bf_out[:, 0, :, :] + t_targ_bf_out[:, 1, :, :] * t_null_bf_out[:, 1, :, :]) # (num_block, num_bin, num_null)
            t_phi_i = alpha * t_phi_i + (1.0 - alpha) * (t_targ_bf_out[:, 1, :, :] * t_null_bf_out[:, 0, :, :] - t_targ_bf_out[:, 0, :, :] * t_null_bf_out[:, 1, :, :]) # (num_block, num_bin, num_null)
            
            t_psd_out = alpha * t_psd_out + (1.0 - alpha) * torch.mean((t_input[:,0,:,:] ** 2 + t_input[:,1,:,:] ** 2), dim = -1, keepdim = True) # ( num_block, num_bin, 1)

            t_phi   = (t_phi_r ** 2 + t_phi_i ** 2) ** 0.5  # (num_block, num_bin, num_null)
            t_dcf   = t_phi / t_psd_out                     # (num_block, num_bin, num_null)
            t_dcf   = torch.clamp(t_dcf, min = 0.01, max = 1.0)
            
            if Low_Freq is not None and High_Freq is not None:
                DCF_Pre_Power = torch.sum(t_psd_out[:, Low_Freq:High_Freq, :], dim = 1, keepdim = True) # ( num_block, 1, 1 )
                DCF_Aft_Power = torch.sum(t_phi[:, Low_Freq:High_Freq, :], dim = 1, keepdim = True)     # ( num_block, 1, num_null )

                Frm_Ratio     = DCF_Aft_Power / (DCF_Pre_Power + 1e-10)         # ( num_block, 1, num_null )
                Frm_Ratio     = torch.clamp(Frm_Ratio, min = 0.01, max = 1.0)   # ( num_block, 1, num_null )
                
                t_dcf         = (t_dcf * Frm_Ratio) ** 0.5                      # (num_block, num_bin, num_null)

            dcf[:, t, :, :] = t_dcf.unsqueeze(1)                # (num_block, 1, num_bin, num_null)
        
        return dcf, targ_bf_out.squeeze(4) # (num_block, num_frame, num_bin, num_null), (num_block, num_frame, 2, num_bin)

