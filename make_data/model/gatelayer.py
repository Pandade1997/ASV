import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import functools
import numpy as np
import math
from torch.autograd import Variable

class RNNGate(nn.Module):
    def __init__(self, channels = [12, 24, 48], num_align = 80):
        super(RNNGate, self).__init__()

        # pow: ( num_frame, 1, num_bin, num_align) --> gate_mask: ( num_frame, 1, h, w)
        self.gate = nn.Sequential(
                        nn.Conv2d(1, channels[0], kernel_size=3, dilation = 1, stride=2),
                        nn.BatchNorm2d(channels[0]),
                        nn.ReLU(),
                        nn.Conv2d(channels[0], channels[1], kernel_size=3, dilation = 2, stride=2),
                        nn.BatchNorm2d(channels[1]),
                        nn.ReLU(),
                        nn.Conv2d(channels[1], channels[2], kernel_size=3, dilation = 3, stride=2),
                        nn.BatchNorm2d(channels[2])
                    )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.rnn      = nn.GRU(input_size = channels[2], hidden_size = channels[2], num_layers = 2, bias = True, batch_first = True, dropout = 0.0, bidirectional = False)

        self.fc = nn.Linear(channels[2], num_align)

    def forward(self, align_fft, num_block = 1, numerical_protection = 1.0e-13, compressed_scale = 0.3):
        # align_fft: ( num_frame, 2, num_bin, num_align )
        
        # align_fft = torch.cat((align_out_r.unsqueeze(1), align_out_i.unsqueeze(1)), dim = 1) # ( num_frame, 1, num_align, num_bin )
        # align_fft = align_fft.transpose(-1, -2)                                              # ( num_frame, 2, num_bin, num_align )

        align_pow = torch.unsqueeze(align_fft[:,0,:,:] ** 2 + align_fft[:,1,:,:] ** 2, 1)    # ( num_frame, 1, num_bin, num_align )
        align_pow = torch.clamp(align_pow, min = numerical_protection)                       # ( num_frame, 1, num_bin, num_align )
        align_pow = align_pow ** (0.5 * compressed_scale)                                    # ( num_frame, 1, num_bin, num_align )

        gate_out = self.gate(align_pow)                                                      # ( num_frame, 48, w, h )                                    
        gate_out = self.avg_pool(gate_out)                                                   # ( num_frame, 48, 1, 1 )
        gate_out = gate_out.view(gate_out.size(0), -1)                                       # ( num_frame, 48)

        hsize = gate_out.size(-1)

        gate_out = gate_out.view([num_block, -1, hsize]).contiguous()                        # ( num_block, T, 48 )

        gate_out, _ = self.rnn(gate_out)                                                     # ( num_block, T, 48 )

        gate_out = gate_out.contiguous()

        hsize = gate_out.size(-1)

        gate_out = gate_out.view([-1, hsize]).contiguous()                                   # ( num_frame, 48 )

        gate_out = self.fc(gate_out)                                                         # ( num_frame, num_align)

        gate_out = F.softmax(gate_out, dim = -1)                                             # ( num_frame, num_align)
        gate_out = gate_out.unsqueeze(1)                                                     # ( num_frame, 1, num_align)

        gate_mask = torch.cat((gate_out, gate_out), dim = 1)                                 # ( num_frame, 2, num_align )
        gate_mask = gate_mask.unsqueeze(3)                                                   # ( num_frame, 2, num_align, 1)
        
        # ( num_frame, 2, num_bin, num_align ) x (num_frame, 2, num_align, 1) = (num_frame, 2, num_bin, 1) -- > ( num_frame, 2, num_bin)
        align_out = torch.matmul(align_fft, gate_mask).squeeze(3) # ( num_frame, 2, num_bin)

        return align_out, gate_mask.squeeze(3)   # ( num_frame, 2, num_bin), (num_frame, 2, num_align)


class Gate(nn.Module):
    def __init__(self, channels = [12, 24, 48], num_align = 80):
        super(Gate, self).__init__()

        # pow: ( num_frame, 1, num_bin, num_align) --> gate_mask: ( num_frame, 1, h, w)
        self.gate = nn.Sequential(
                        nn.Conv2d(1, channels[0], kernel_size=3, dilation = 1, stride=2),
                        nn.BatchNorm2d(channels[0]),
                        nn.ReLU(),
                        nn.Conv2d(channels[0], channels[1], kernel_size=3, dilation = 2, stride=2),
                        nn.BatchNorm2d(channels[1]),
                        nn.ReLU(),
                        nn.Conv2d(channels[1], channels[2], kernel_size=3, dilation = 3, stride=2),
                        nn.BatchNorm2d(channels[2])
                    )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[2], num_align)

    def forward(self, align_fft, numerical_protection = 1.0e-13, compressed_scale = 0.3):
        # align_fft: ( num_frame, 2, num_bin, num_align )
        
        # align_fft = torch.cat((align_out_r.unsqueeze(1), align_out_i.unsqueeze(1)), dim = 1) # ( num_frame, 1, num_align, num_bin )
        # align_fft = align_fft.transpose(-1, -2)                                              # ( num_frame, 2, num_bin, num_align )

        align_pow = torch.unsqueeze(align_fft[:,0,:,:] ** 2 + align_fft[:,1,:,:] ** 2, 1)    # ( num_frame, 1, num_bin, num_align )
        align_pow = torch.clamp(align_pow, min = numerical_protection)                       # ( num_frame, 1, num_bin, num_align )
        align_pow = align_pow ** (0.5 * compressed_scale)                                    # ( num_frame, 1, num_bin, num_align )

        gate_out = self.gate(align_pow)                                                      # ( num_frame, 48, w, h )                                    
        gate_out = self.avg_pool(gate_out)                                                   # ( num_frame, 48, 1, 1 )
        gate_out = gate_out.view(gate_out.size(0), -1)                                       # ( num_frame, 48)
        gate_out = self.fc(gate_out)                                                         # ( num_frame, num_align)

        gate_out = F.softmax(gate_out, dim = -1)                                             # ( num_frame, num_align)
        gate_out = gate_out.unsqueeze(1)                                                     # ( num_frame, 1, num_align)

        gate_mask = torch.cat((gate_out, gate_out), dim = 1)                                 # ( num_frame, 2, num_align )
        gate_mask = gate_mask.unsqueeze(3)                                                   # ( num_frame, 2, num_align, 1)
        
        # ( num_frame, 2, num_bin, num_align ) x (num_frame, 2, num_align, 1) = (num_frame, 2, num_bin, 1) -- > ( num_frame, 2, num_bin)
        align_out = torch.matmul(align_fft, gate_mask).squeeze(3) # ( num_frame, 2, num_bin)

        return align_out, gate_mask.squeeze(3)   # ( num_frame, 2, num_bin), (num_frame, 2, num_align)