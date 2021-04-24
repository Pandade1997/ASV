# -*- coding: utf-8 -*-
'''
Created on 2020-03-06, 10:50

@autorh: shuai nie
'''

import torch
from torch.autograd import Function


def to_cuda(m, x):
    assert isinstance(m, torch.nn.Module)
    device_id = torch.cuda.device_of(next(m.parameters()).data).idx
    if device_id == -1:
        return x
    return x.cuda(device_id)

class BinarizeF(Function):

    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# aliases
binarize = BinarizeF.apply


