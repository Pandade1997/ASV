#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def show_params(nnet, name = ''):
    print("=" * 40, "%s Model Parameters" % (name), "=" * 40)
    num_params = 0
    for module_name, m in nnet.named_modules():
        if module_name == '':
            for name, params in m.named_parameters():
                print(name, params.size())
                i = 1
                for j in params.size():
                    i = i * j
                num_params += i
    print('[*] Parameter Size: {}'.format(num_params))
    print("=" * 98)

def show_model(nnet, name = ''):
    num_params = 0
    print("=" * 40, "%s Model Structures" % (name), "=" * 40)
    for module_name, net in nnet.named_modules():
        if module_name == '':
            print(net)
        for param in net.parameters():
            num_params += param.numel()
    print("=" * 28, '%s Model Total number of parameters : %.3f K' % (name, num_params / 1024), "=" * 28)
