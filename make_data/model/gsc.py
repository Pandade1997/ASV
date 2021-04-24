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

#############################################################################################################
##################################### Complex multi-channel beamformor Layer ################################
#############################################################################################################
class GSC(nn.Module):
    r"""Implement Directional Coherence Function to extract the Directional Features
    Args:
        num_beam: number of beamforming filters
        num_bin: number of frequency bin
        num_channel: number of channels
    """
    def __init__(self, num_beam, num_null, num_bin, num_channel, targ_bf_weight = None, block_bf_weight = None, fix = True):
        super(GSC, self).__init__()
        
        self.num_beam    = num_beam
        self.num_null    = num_null
        self.num_bin     = num_bin
        self.num_channel = num_channel
        
        self.targ_bf  = Beamformor(num_beam = num_beam, num_bin = num_bin, num_channel = num_channel, weight_init = targ_bf_weight, fix = fix)
        self.block_bf = NullBeamformor(num_beam = num_beam, num_null = num_null, num_bin = num_bin, num_channel = num_channel, weight_init = block_bf_weight, fix = fix)
        
    def complex_mm(self, x1_r, x1_i, x2_r, x2_i):
        # x1: (num_block, num_bin, M, M)
        # x2: (num_block, num_bin, M, 1)
        y_r = torch.matmul(x1_r, x2_r) - torch.matmul(x1_i, x2_i) # (num_block, num_bin, M, 1)
        y_i = torch.matmul(x1_r, x2_i) + torch.matmul(x1_i, x2_r) # (num_block, num_bin, M, 1)

        return y_r, y_i # (num_block, num_bin, M, 1)

    def forward(self, input, mask, beam_id, alpha_v = 0.95, DCF_Ratio = 0.50):
        # input:     ( num_block * num_frame, 2, num_bin, num_channel )
        # mask:      ( num_block, num_frame, num_bin )

        # targ_out:  ( num_block, num_frame, 2, num_bin, 1 )
        # block_out: ( num_block, num_frame, 2, num_bin, M )

        print("DCF_Ratio = %.2f" % (DCF_Ratio))

        #assert beam_id >= 0 and beam_id <= self.num_beam, "illegal beam_id, the required beam_id shape is [0, %d], but got %d" % (self.num_beam, beam_id)
        num_block, num_frame, num_bin = mask.size()
        M = self.num_null
        num_block   = beam_id.size(0)

        input       = input.view([num_block, -1, 2, self.num_bin, self.num_channel])    # (num_block, num_frame, 2, num_bin, num_channel )
        
        targ_out_list  = []
        block_out_list = []
        for n in torch.arange(num_block):
            targ_out  = self.targ_bf(input[n], beam_id[n])   # (num_frame, 2, num_bin, 1)
            block_out = self.block_bf(input[n], beam_id[n])  # (num_frame, 2, num_bin, M)

            targ_out  = targ_out.unsqueeze(0)                # (1, num_frame, 2, num_bin, 1)
            block_out = block_out.unsqueeze(0)               # (1, num_frame, 2, num_bin, M)
            
            targ_out_list.append(targ_out)
            block_out_list.append(block_out)

        targ_out  = torch.cat(targ_out_list, 0)              # (num_block, num_frame, 2, num_bin, 1)
        block_out = torch.cat(block_out_list, 0)             # (num_block, num_frame, 2, num_bin, M)

        phi_bb_r = to_cuda(self, 1.0e5 * torch.eye(M))       # (M, M)
        phi_bb_i = to_cuda(self, torch.zeros(M, M))          # (M, M)

        phi_bb_r = phi_bb_r.view(1, 1, M, M)                 # (1, 1, M, M)
        phi_bb_i = phi_bb_i.view(1, 1, M, M)                 # (1, 1, M, M)

        phi_bb_r = phi_bb_r.expand(num_block, num_bin, M, M).contiguous()  # (num_block, num_bin, M, M)
        phi_bb_i = phi_bb_i.expand(num_block, num_bin, M, M).contiguous()  # (num_block, num_bin, M, M)

        W_nc_r   = to_cuda(self, torch.zeros(num_block, num_bin, M, 1))    # ( num_block, num_bin, M, 1)
        W_nc_i   = to_cuda(self, torch.zeros(num_block, num_bin, M, 1))    # ( num_block, num_bin, M, 1)

        out      = to_cuda(self, torch.zeros(num_block, num_frame, 2, num_bin))  # ( num_block, num_frame, 2, num_bin)

        for t in torch.arange(1, num_frame):
            
            t_mask      = mask[:, t, :]                          # ( num_block, num_bin )
            alpha_b     = alpha_v + (1.0 - alpha_v) * t_mask     # ( num_block, num_bin )
            alpha_b     = torch.clamp(alpha_b, max = 1.00)       # ( num_block, num_bin )

            alpha_b = torch.where( t_mask < DCF_Ratio, alpha_b, torch.ones_like(alpha_b) )

            t_targ_r   = targ_out[:, t, 0, :, :].unsqueeze(3)     # ( num_block, num_bin, 1, 1 )
            t_targ_i   = targ_out[:, t, 1, :, :].unsqueeze(3)     # ( num_block, num_bin, 1, 1 )

            t_block_r  = block_out[:, t, 0, :, :].unsqueeze(3)    # ( num_block, num_bin, M, 1 )
            t_block_i  = block_out[:, t, 1, :, :].unsqueeze(3)    # ( num_block, num_bin, M, 1 )
            tt_block_r = t_block_r.transpose(-1, -2).contiguous() # ( num_block, num_bin, 1, M )
            tt_block_i = t_block_i.transpose(-1, -2).contiguous() # ( num_block, num_bin, 1, M )

            kb_up_r, kb_up_i   = self.complex_mm(phi_bb_r, phi_bb_i, t_block_r, t_block_i)  # (num_block, num_bin, M, 1)
            kb_low_r, kb_low_i = self.complex_mm(tt_block_r, -tt_block_i, kb_up_r, kb_up_i) # (num_block, num_bin, 1, 1)

            kb_up_r = (1.0 - alpha_b.view(num_block, num_bin, 1, 1)) * kb_up_r              # (num_block, num_bin, M, 1)
            kb_up_i = (1.0 - alpha_b.view(num_block, num_bin, 1, 1)) * kb_up_i              # (num_block, num_bin, M, 1)

            kb_low_r = alpha_b.view(num_block, num_bin, 1, 1) + (1.0 - alpha_b.view(num_block, num_bin, 1, 1)) * kb_low_r # (num_block, num_bin, 1, 1)
            #kb_low_r = alpha_b.view(num_block, num_bin, 1, 1) + kb_low_r # (num_block, num_bin, 1, 1)

            kb_r = kb_up_r / (kb_low_r + 1.0e-13) # (num_block, num_bin, M, 1)
            kb_i = kb_up_i / (kb_low_r + 1.0e-13) # (num_block, num_bin, M, 1)

            phi_bb_r_new, phi_bb_i_new = self.complex_mm(kb_r, kb_i, tt_block_r, -tt_block_i) # (num_block, num_bin, M, M)

            phi_bb_r_new, phi_bb_i_new = self.complex_mm(phi_bb_r_new, phi_bb_i_new, phi_bb_r, phi_bb_i) # (num_block, num_bin, M, M)

            phi_bb_r = (phi_bb_r - phi_bb_r_new) / (alpha_b.view(num_block, num_bin, 1, 1) + 1.0e-13)  # (num_block, num_bin, M, M) 
            phi_bb_i = (phi_bb_i - phi_bb_i_new) / (alpha_b.view(num_block, num_bin, 1, 1) + 1.0e-13)  # (num_block, num_bin, M, M)

            Wnc_r, Wnc_i = self.complex_mm(tt_block_r, -tt_block_i, W_nc_r, W_nc_i) # ( num_block, num_bin, 1, 1)
            Wnc_r        = t_targ_r - Wnc_r                                        # ( num_block, num_bin, 1, 1)
            Wnc_i        = -t_targ_i - Wnc_i                                       # ( num_block, num_bin, 1, 1)

            Wnc_r, Wnc_i = self.complex_mm(kb_r, kb_i, Wnc_r, Wnc_i) # ( num_block, num_bin, M, 1)

            W_nc_r = W_nc_r + Wnc_r # ( num_block, num_bin, M, 1)
            W_nc_i = W_nc_i + Wnc_i # ( num_block, num_bin, M, 1)

            tt_W_nc_r = W_nc_r.transpose(-1, -2).contiguous() # ( num_block, num_bin, 1, M )
            tt_W_nc_i = W_nc_i.transpose(-1, -2).contiguous() # ( num_block, num_bin, 1, M )

            out_r, out_i = self.complex_mm(tt_W_nc_r, -tt_W_nc_i, t_block_r, t_block_i) # ( num_block, num_bin, 1, 1 )

            out_r = t_targ_r - out_r            # ( num_block, num_bin, 1, 1 )
            out_i = t_targ_i - out_i            # ( num_block, num_bin, 1, 1 )

            '''
            out_spect     = out_r ** 2 + out_i ** 2         # ( num_block, num_bin, 1, 1 )
            t_targ_spect  = t_targ_r ** 2 + t_targ_i ** 2   # ( num_block, num_bin, 1, 1 )
            out_r         = torch.where(out_spect < t_targ_spect, out_r, t_targ_r)
            out_i         = torch.where(out_spect < t_targ_spect, out_i, t_targ_i)

            out_r = torch.where(t_mask.view(num_block, num_bin, 1, 1) < DCF_Ratio, out_r, t_targ_r)
            out_i = torch.where(t_mask.view(num_block, num_bin, 1, 1) < DCF_Ratio, out_i, t_targ_i)
            
            out_r[:, 0:3, :, :] = t_targ_r[:, 0:3, :, :]
            out_i[:, 0:3, :, :] = t_targ_i[:, 0:3, :, :]
            '''

            out_r = t_mask * out_r.squeeze()    # ( num_block, num_bin )
            out_i = t_mask * out_i.squeeze()    # ( num_block, num_bin )

            out[:, t, 0, :] = out_r.squeeze() # ( num_block, num_frame, 2, num_bin)
            out[:, t, 1, :] = out_i.squeeze() # ( num_block, num_frame, 2, num_bin)

        return out # ( num_block, num_frame, 2, num_bin)


