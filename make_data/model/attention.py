import numpy as np
import torch
import torch.nn as nn
from model.base_model import clones
import math
import torch.nn.functional as F

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # query: (num_block, H, num_frame, d_q)
    # key: (num_block, H, num_frame, d_k)
    # value: (num_block, H, num_frame, d_v)

    # (num_block, H, num_frame, d_q) x (num_block, H, d_k, num_frame) = (num_block, H, num_frame, num_frame)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim = -1) # (num_block, H, num_frame, num_frame)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    # (num_block, H, num_frame, num_frame) x (num_block, H, num_frame, d_v) = (num_block, H, num_frame, d_v)
    # Z_o: (num_block, H, num_frame, d_v), p_attn: (num_block, H, num_frame, num_frame)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, in_d_q, in_d_k, in_d_v, out_d_k, out_d_v, out_d, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = out_d_k
        self.d_v = out_d_v

        self.w_qs = nn.Linear(in_d_q, n_head * out_d_k)
        self.w_ks = nn.Linear(in_d_k, n_head * out_d_k)
        self.w_vs = nn.Linear(in_d_v, n_head * out_d_v)
        self.w_o = nn.Linear(n_head * out_d_k, out_d)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (in_d_q + out_d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (in_d_k + out_d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (in_d_v + out_d_v)))
        nn.init.xavier_normal_(self.w_o.weight)

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, q, k, v, mask = None):
        '''
        q: (num_block, num_frame, d_x), W_q: (d_x, H*d_q)
        k: (num_block, num_frame, d_x), W_k: (d_x, H*d_k)
        v: (num_block, num_frame, d_x), W_v: (d_x, H*d_v)
        '''

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        out_d_k, out_d_v, n_head = self.d_k, self.d_v, self.n_head

        num_block, num_frame_q, in_d_q = q.size()
        num_block, num_frame_k, in_d_k = k.size()
        num_block, num_frame_v, in_d_v = v.size()

        # q: (num_block, num_frame, d_x) X (d_x, H*d_q) = (num_block, num_frame, H*d_q) --> (num_block, num_frame, H, d_q) --> (num_block, H, num_frame, d_q)
        q = self.w_qs(q).view(num_block, -1, n_head, out_d_k).transpose(1, 2) 
        
        # k: (num_block, num_frame, d_x) X (d_x, H*d_k) = (num_block, num_frame, H*d_k) --> (num_block, num_frame, H, d_k) --> (num_block, H, num_frame, d_k)
        k = self.w_ks(k).view(num_block, -1, n_head, out_d_k).transpose(1, 2)
        
        # v: (num_block, num_frame, d_x) X (d_x, H*d_v) = (num_block, num_frame, H*d_v) --> (num_block, num_frame, H, d_v) --> (num_block, H, num_frame, d_v)
        v = self.w_vs(v).view(num_block, -1, n_head, out_d_v).transpose(1, 2)

        x, attn = attention(q, k, v, mask=mask, dropout=self.dropout)

        # x: (num_block, H, num_frame, d_v) --> (num_block, num_frame, H, d_v) --> (num_block, num_frame, H * d_v)
        x = x.transpose(1, 2).contiguous().view(num_block, -1, n_head * out_d_v)

        x = self.w_o(x) # (num_block, num_frame, d_o)

        #if self.dropout is not None:
        #    x = self.dropout(x)

        return x, attn # (num_block, num_frame, d_o)
'''
class MultiHeadAttention(nn.Module):
    #Multi-Head Attention module

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature = np.power(d_k, 0.5),
                                                   attn_dropout = dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):
        #q: (num_block, num_frame, d_x), W_q: (d_x, H*d_q)
        #k: (num_block, num_frame, d_x), W_k: (d_x, H*d_k)
        #v: (num_block, num_frame, d_x), W_v: (d_x, H*d_v)

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) # (num_block, num_frame, H*d_q) --> (num_block, num_frame, H, d_q)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k) # (num_block, num_frame, H*d_k) --> (num_block, num_frame, H, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v) # (num_block, num_frame, H*d_v) --> (num_block, num_frame, H, d_v)
        
        # (num_block, num_frame, H, d_q) --> (H, num_block, num_frame, d_q) --> ( H * num_block, num_frame, d_q )
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) 

        # (num_block, num_frame, H, d_k) --> (H, num_block, num_frame, d_k) --> ( H * num_block, num_frame, d_k )
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)

        # (num_block, num_frame, H, d_v) --> (H, num_block, num_frame, d_v) --> ( H * num_block, num_frame, d_v )
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..

        # output: ( H * num_block, num_frame, d_v ), atten: ( H * num_block, num_frame, num_frame )
        output, attn = self.attention(q, k, v, mask=mask)

        # ( H * num_block, num_frame, d_v ) --> ( H, num_block, num_frame, d_v )
        output = output.view(n_head, sz_b, len_q, d_v)

        # ( H, num_block, num_frame, d_v ) --> ( num_block, num_frame, H, d_v ) --> ( num_block, num_frame, H*d_v )
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output)) # ( num_block, num_frame, H*d_v ) X (H*d_v, d_o) --> ( num_block, num_frame, d_o )
        output = self.layer_norm(output + residual)

        return output, attn # output: ( num_block, num_frame, d_o ), attn: ( H * num_block, num_frame, num_frame )

class ScaledDotProductAttention(nn.Module):
    #Scaled Dot-Product Attention 

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # q: ( H * num_block, num_frame, d_q )
        # k: ( H * num_block, num_frame, d_k )
        # v: ( H * num_block, num_frame, d_v )

        # ( H * num_block, num_frame, d_q ) X ( H * num_block, d_k, num_frame ) = ( H * num_block, num_frame, num_frame )
        attn = torch.bmm(q, k.transpose(1, 2)) 
        attn = attn / self.temperature # ( H * num_block, num_frame, num_frame )

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.softmax(attn) # ( H * num_block, num_frame, num_frame )
        attn = self.dropout(attn) # ( H * num_block, num_frame, num_frame )
        # ( H * num_block, num_frame, num_frame ) X ( H * num_block, num_frame, d_v ) = ( H * num_block, num_frame, d_v )
        output = torch.bmm(attn, v) # ( H * num_block, num_frame, d_v )

        return output, attn # output: ( H * num_block, num_frame, d_v ), atten: ( H * num_block, num_frame, num_frame )
'''