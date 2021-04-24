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
from .attention import MultiHeadAttention
import torch.nn.functional as F

#############################################################################################################
############################################# Transformer Layer #############################################
#############################################################################################################
class PositionwiseFeedForward(nn.Module):
    """Implements position-wise feedforward sublayer.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class TransformerLayer(nn.Module):
    "DeepTransformer is made up of self-attn and feed forward (defined below)"
    def __init__(
        self,
        in_size,
        d_model,
        query_key_size,
        value_size,
        num_head = 1,
        layer_norm = False,
        residual_op = False,
        dropout = 0.0):
        super(TransformerLayer, self).__init__()

        self.residual_op = residual_op
        if residual_op:
            assert d_model == in_size, "d_model = %d and in_size = %d MUST be same for the self-attention with residual_op" % (in_size, d_model)

        self.self_attn = MultiHeadAttention(
            n_head = num_head,
            in_d_q = in_size,
            in_d_k = in_size,
            in_d_v = in_size,
            out_d_k = query_key_size,
            out_d_v = value_size,
            out_d = d_model,
            dropout = 0.1)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(d_model)
        else:
            self.layer_norm = None

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, non_pad_mask = None, slf_attn_mask = None):
        "Follow Figure 1 (left) for connections."

        residual = x

        x, attn = self.self_attn( x, x, x, mask = slf_attn_mask)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.residual_op:
            x = x + residual

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if non_pad_mask is not None:
            x *= non_pad_mask

        return x, attn

class TransformerFFLayer(nn.Module):
    "DeepTransformer is made up of self-attn and feed forward (defined below)"
    def __init__(
        self,
        in_size,
        d_model,
        d_ff_inner,
        query_key_size,
        value_size,
        num_head = 1,
        layer_norm = False,
        residual_op = False,
        dropout = 0.0):
        super(TransformerFFLayer, self).__init__()

        self.residual_op = residual_op
        if residual_op:
            assert d_model == in_size, "d_model = %d and in_size = %d MUST be same for the self-attention with residual_op" % (in_size, d_model)

        self.self_attn = MultiHeadAttention(
            n_head = num_head,
            in_d_q = in_size,
            in_d_k = in_size,
            in_d_v = in_size,
            out_d_k = query_key_size,
            out_d_v = value_size,
            out_d = d_model,
            dropout = 0.1)

        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff_inner, dropout = dropout)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(d_model)
        else:
            self.layer_norm = None

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, non_pad_mask = None, slf_attn_mask = None):
        "Follow Figure 1 (left) for connections."

        residual = x

        x, attn = self.self_attn( x, x, x, mask = slf_attn_mask)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.residual_op:
            x = x + residual

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if non_pad_mask is not None:
            x *= non_pad_mask

        x = self.pos_ffn(x)

        if non_pad_mask is not None:
            x *= non_pad_mask

        return x, attn