import numpy as np
import time
import torch
import os
import sys
import math

def choose_quantization_params(w_abs_max, qmax):
    w_abs_max = float(qmax) / w_abs_max 
    Q = torch.floor(torch.log2(w_abs_max))    
    return Q

def quantize(qparams, fdata, qBit):
    bound = math.pow(2.0, qBit-1)
    min_val = - bound
    max_val = bound - 1
    if len(fdata.size()) == 1:
        fdata = fdata.unsqueeze(1)    
    times = torch.pow(torch.FloatTensor([2.0]).to(device), qparams)
    transformed_val = fdata * times
    clipped_value = torch.round(torch.clamp(transformed_val, min_val, max_val))
    if qBit == 8:
        return clipped_value.to(torch.int8)
    elif qBit == 16:
        return clipped_value.to(torch.int16)
    else:
        return clipped_value
        
def dequantize(qparams, fixed_data):    
    times = torch.pow(torch.FloatTensor([2.0]).to(device), -qparams)
    fixed_data = fixed_data.to(torch.float32)
    transformed_val = fixed_data * times
    return transformed_val