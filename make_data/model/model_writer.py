import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
import math, copy, time
import torch.nn as nn
from torch.nn import init
import numpy as np
from model.scheduler import get_scheduler
import codecs

supported_rnns = {'lstm': nn.LSTM, 'rnn': nn.RNN, 'gru': nn.GRU}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())
supported_acts = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'softmax': nn.Softmax(), 'tanh': nn.Tanh(),'leakyrelu': nn.LeakyReLU(), 'prelu': nn.PReLU(), 'softplus': nn.Softplus()}
supported_acts_inv = dict((v, k) for k, v in supported_acts.items())
supported_loss = {'mseloss': nn.MSELoss(), 'kldivloss': nn.KLDivLoss(), 'smoothl1loss': nn.SmoothL1Loss()}
supported_loss_inv = dict((v, k) for k, v in supported_loss.items())

def ChooseQuantizationQParams(vmax, qmax):
    vmax = np.abs(vmax)
    Q = 0
    if vmax < qmax:
        while vmax * 2 <= qmax:
            Q = Q + 1
            vmax = vmax * 2.0
    else:
        while  vmax >= qmax:
            Q = Q - 1
            vmax = vmax * 0.5
    return Q

def QQuantize(qparams, fdata, qBit):
    if len(fdata.shape) < 2:
        fdata = fdata[np.newaxis, :]
    row, col = fdata.shape
    if qBit == 8:
        fixed_data = np.zeros(shape = (row, col), dtype=np.int8)
    elif qBit == 16:
        fixed_data = np.zeros(shape = (row, col), dtype=np.int16)
    elif qBit == 32:
        fixed_data = np.zeros(shape = (row, col), dtype=np.int32)
    elif qBit == 64:
        fixed_data = np.zeros(shape = (row, col), dtype=np.int64)
    else:
        fixed_data = np.zeros(shape = (row, col), dtype=np.int8)
    
    for i in range(row):
        for j in range(col):
            real_val = fdata[i, j]
            transformed_val = real_val * 2**qparams
            clamped_val = max( -2**(qBit-1),  min( 2**(qBit-1) - 1, transformed_val ))

            if qBit == 8:
                fixed_data[i, j]  = np.int8(np.round(clamped_val))
            elif qBit == 16:
                fixed_data[i, j]  = np.int16(np.round(clamped_val))
            elif qBit == 32:
                fixed_data[i, j]  = np.int32(np.round(clamped_val))
            elif qBit == 64:
                fixed_data[i, j]  = np.int64(np.round(clamped_val))
            else:
                fixed_data[i, j]  = np.int8(np.round(clamped_val))
    return fixed_data

def aQQuantize(qparams, fdata, bits):

    transformed_val = fdata * 2 ** qparams
    clamped_val = max( -2**(bits-1),  min( 2**(bits-1) - 1, transformed_val ))
    if bits == 8:
        fixed_data  = np.int8(np.round(clamped_val))
    elif bits == 16:
        fixed_data  = np.int16(np.round(clamped_val))
    elif bits == 32:
        fixed_data  = np.int32(np.round(clamped_val))
    elif bits == 64:
        fixed_data  = np.int64(np.round(clamped_val))
    else:
        fixed_data  = np.int8(np.round(clamped_val))
    return fixed_data

def write_gru_enhance_to_ccode(model, out_file_name, basic_file_name, skip_write_net = None, QMAX = 8192):

    basic_args   = model.basic_args
    if hasattr(model, 'mask_args'):
        mask_args = model.mask_args
    else:
        mask_args = None
    
    ## integrete the cmvn into the weight and bias of 1st layer ##
    cmvn_file = basic_args.cmvn_file
    cmvn = None
    if cmvn_file is not None and os.path.exists(cmvn_file):
        print("load cmvn from %s" % (cmvn_file))
        cmvn = np.load(cmvn_file)
    if cmvn is not None:
        feat_size     = cmvn.shape[1]
    else:
        feat_size     = 40
    input_size    = mask_args.layer_size[0]
    context_frame = int(input_size / feat_size)
    
    if cmvn is not None:
        addshift = cmvn[0, :]
        rescale  = cmvn[1, :]
        
        feat_delta = 1.0 / rescale
        feat_u     = -addshift

        delta = np.tile(feat_delta, [1, context_frame]) # [1, 40]
        u     = np.tile(feat_u, [1, context_frame])     # [1, 40]
    else:
        delta = None
        u     = None
    
    ## quantize the weight and bias ##
    qmax = 127
    quantized_models = {}
    for model_name in model.model_names:
        if skip_write_net is not None and model_name == skip_write_net:
            continue
        
        quantized_net = []
        
        net = getattr(model, model_name)
        net_dict = net.cpu().state_dict()
        
        # fuse the delta and mean into the first layer
        if delta is not None and u is not None and model_name == 'netE':
            
            net_keys = list(net_dict.keys())
            weight   = net_dict[net_keys[0]].cpu()               # shape = [384, 40]
            bias     = net_dict[net_keys[2]].cpu()               # shape = [384]

            weight = weight.numpy()      # shape = [384, 40]
            bias   = bias.numpy()
            bias   = bias[:, np.newaxis] # shape = [384, 1]
                        
            weight = weight / delta            # [384, 40] / [1, 40]
            bias   = bias - weight.dot(u.T)    # [384, 1] - [384, 40] * [1, 40]'
            bias   = bias.squeeze()
            
            net_dict[net_keys[0]] = torch.from_numpy(weight)
            net_dict[net_keys[2]] = torch.from_numpy(bias)
        
        net_layers = {}
        for k in net_dict.keys():
            ilayer = int(k.split('.')[1])
            if ilayer in net_layers:
                layer = net_layers[ilayer]
                layer.append(k)
                net_layers[ilayer] = layer
            else:
                net_layers[ilayer] = [k]
        
        for ilayer, layers in net_layers.items():
            weight_ih = None
            bias_ih   = None
            weight_hh = None
            bias_hh   = None
            for k in layers:
                if 'weight' in k:
                    if 'weight_hh' in k:
                        weight_hh = net_dict[k].cpu().numpy()
                    else:
                        weight_ih = net_dict[k].cpu().numpy()
                if 'bias' in k:
                    if 'bias_hh' in k:
                        bias_hh   = net_dict[k].cpu().numpy()
                    else:
                        bias_ih   = net_dict[k].cpu().numpy()

            if weight_ih is not None:
                if bias_ih is not None:
                    qparams_ih = np.zeros(shape = weight_ih.shape[0], dtype = np.int)
                    qweight_ih = np.zeros(shape = weight_ih.shape, dtype = np.int8)
                    qbias_ih   = np.zeros(shape = bias_ih.shape, dtype = np.int32)
                    for r in range(weight_ih.shape[0]):
                        w_abs_max = max(np.max(np.abs(weight_ih[r, :])), 0.001)
                        params    = ChooseQuantizationQParams(w_abs_max, qmax)
                        qweight_ih[r, :] = QQuantize(params, weight_ih[r, :], 8)
                        qbias_ih[r] = aQQuantize(params, bias_ih[r], 32)
                        qparams_ih[r] = params
                else:
                    qparams_ih = np.zeros(shape = weight_ih.shape[0], dtype = np.int)
                    qweight_ih = np.zeros(shape = weight_ih.shape, dtype = np.int8)
                    qbias_ih   = None
                    for r in range(weight_ih.shape[0]):
                        w_abs_max = max(np.max(np.abs(weight_ih[r, :])), 0.001)
                        params    = ChooseQuantizationQParams(w_abs_max, qmax)
                        qweight_ih[r, :] = QQuantize(params, weight_ih[r, :], 8)
                        qparams_ih[r] = params
            else: 
                qparams_ih = None
                qweight_ih = None
                qbias_ih   = None

            if weight_hh is not None:
                if bias_hh is not None:
                    qparams_hh = np.zeros(shape = weight_hh.shape[0], dtype = np.int)
                    qweight_hh = np.zeros(shape = weight_hh.shape, dtype = np.int8)
                    qbias_hh   = np.zeros(shape = bias_hh.shape, dtype = np.int32)
                    for r in range(weight_hh.shape[0]):
                        w_abs_max = max(np.max(np.abs(weight_hh[r, :])), 0.001)
                        params    = ChooseQuantizationQParams(w_abs_max, qmax)
                        qweight_hh[r, :] = QQuantize(params, weight_hh[r, :], 8)
                        qbias_hh[r] = aQQuantize(params, bias_hh[r], 32)
                        qparams_hh[r] = params
                else:
                    qparams_hh = np.zeros(shape = weight_hh.shape[0], dtype = np.int)
                    qweight_hh = np.zeros(shape = weight_hh.shape, dtype = np.int8)
                    qbias_hh   = None
                    for r in range(weight_hh.shape[0]):
                        w_abs_max = max(np.max(np.abs(weight_hh[r, :])), 0.001)
                        params    = ChooseQuantizationQParams(w_abs_max, qmax)
                        qweight_hh[r, :] = QQuantize(params, weight_hh[r, :], 8)
                        qparams_hh[r] = params
            else: 
                qparams_hh = None
                qweight_hh = None
                qbias_hh   = None
            quantized_net.append((qweight_ih, qbias_ih, qparams_ih, qweight_hh, qbias_hh, qparams_hh))
        
        quantized_models[model_name] = quantized_net
    
    f = codecs.open(out_file_name, 'w', 'utf-8')
    
    ## write header file ##
    f.write('#include \"skv_layers.h\"\n')
    f.write('#include \"../basic/os_support.h\"\n')
    f.write('#include \"../math/skv_math_core_fix.h\"\n')
    f.write('#include \"../math/skv_fastmath.h\"\n')
    f.write('#include <stdio.h>\n')
    f.write('#include <assert.h>\n')
    f.write('#include <stdbool.h>\n\n')
    
    ## write the weight and bias ##
    for model_name, quantized_net in quantized_models.items():
        if skip_write_net is not None and model_name == skip_write_net:
            continue
        
        suffix_names = model_name
        for ilayer in range(len(quantized_net)):
            qweight_ih, qbias_ih, qparams_ih, qweight_hh, qbias_hh, qparams_hh = quantized_net[ilayer]

            num_w = 0
            num   = 0
            if qweight_ih is not None:
                num = num + qweight_ih.shape[0] * qweight_ih.shape[1]
            if qweight_hh is not None:
                num = num + qweight_hh.shape[0] * qweight_hh.shape[1]

            f.write('static const skv_weight %s_weight_%d[%d] = {' % (suffix_names, ilayer + 1, num))
            if qweight_ih is not None:
                for i in range(qweight_ih.shape[0]):
                    for j in range(qweight_ih.shape[1]):
                        if num_w % 8 == 0:
                            if num_w == num - 1:
                                f.write('\n\t%d ' % qweight_ih[i, j])
                            else:
                                f.write('\n\t%d, ' % qweight_ih[i, j])
                            num_w = num_w + 1
                        else:
                            if num_w == num - 1:
                                f.write('%d ' % qweight_ih[i, j])
                            else:
                                f.write('%d, ' % qweight_ih[i, j])
                            num_w = num_w + 1
            if qweight_hh is not None:
                for i in range(qweight_hh.shape[0]):
                    for j in range(qweight_hh.shape[1]):
                        if num_w % 8 == 0:
                            if num_w == num - 1:
                                f.write('\n\t%d ' % qweight_hh[i, j])
                            else:
                                f.write('\n\t%d, ' % qweight_hh[i, j])
                            num_w = num_w + 1
                        else:
                            if num_w == num - 1:
                                f.write('%d ' % qweight_hh[i, j])
                            else:
                                f.write('%d, ' % qweight_hh[i, j])
                            num_w = num_w + 1
            f.write('\n\t};\n')

            num_b = 0
            num   = 0
            if qbias_ih is not None:
                num = num + qbias_ih.shape[0]
            if qbias_hh is not None:
                num = num + qbias_hh.shape[0]
            f.write('static const skv_bias %s_bias_%d[%d] = {' % (suffix_names, ilayer + 1, num))
            if qbias_ih is not None:
                for i in range(qbias_ih.shape[0]):
                    if num_b % 8 == 0:
                        if num_b == num - 1:
                            f.write('\n\t%d ' % (qbias_ih[i]))
                        else:
                            f.write('\n\t%d, ' % (qbias_ih[i]))
                        num_b = num_b + 1
                    else:
                        if num_b == num - 1:
                            f.write('%d ' % (qbias_ih[i]))
                        else:
                            f.write('%d, ' % (qbias_ih[i]))
                        num_b = num_b + 1
            if qbias_hh is not None:
                for i in range(qbias_hh.shape[0]):
                    if num_b % 8 == 0:
                        if num_b == num - 1:
                            f.write('\n\t%d ' % (qbias_hh[i]))
                        else:
                            f.write('\n\t%d, ' % (qbias_hh[i]))
                        num_b = num_b + 1
                    else:
                        if num_b == num - 1:
                            f.write('%d ' % (qbias_hh[i]))
                        else:
                            f.write('%d, ' % (qbias_hh[i]))
                        num_b = num_b + 1
            f.write('\n\t};\n\n')
    
    ## write the basic code from the basic code file ##
    with open(basic_file_name) as freader:
        for line in freader.readlines():
            f.write('%s' % (line))
    
    in_qParams  = 0
    out_qParams = 0
    ## write the skv_init_layer  ##
    for model_name in model.model_names:
        if skip_write_net is not None and model_name == skip_write_net:
            continue
        
        suffix_names = model_name
        net = getattr(model, model_name)
        num_output_frame = 1
        
        if model_name in quantized_models:
            quantized_net = quantized_models[model_name]
        else:
            continue
        
        input_layer_insize   = net.input_size
        input_layer_outsize  = net.input_size
        if 'netE' == model_name:
            hact_type            = 'linear'
            oact_type            = 'linear'
            num_layer            = len(net.NNet) + 1 + 1
            in_qParams           = 0
            out_qParams          = 0
        elif 'spe_mask_prj' == model_name:
            hact_type            = 'linear'
            oact_type            = 'sigmoid'
            if net.vectorized_type is None:
                num_layer = len(net.NNet) + 1 + 1 + 1
            else:
                num_layer = len(net.NNet) + 1 + 1 + 1 + 1
            
            in_min = 0
            in_max = model.netE.NNet[-1].act_mean + 3.0 * model.netE.NNet[-1].act_std
            max_abs = max(abs(in_min), abs(in_max))
            in_qParams = ChooseQuantizationQParams(max_abs.cpu(), QMAX)
            out_qParams = 0
        else:
            continue
        
        f.write('\nEXPORT SKVLayerState * skv_%s_layers_init(int num_output_frame)\n' % (suffix_names))
        f.write('{\n')
        
        f.write('\tSKVLayerState * st = NULL;\n')
        f.write('\tst = (SKVLayerState *)speex_alloc(sizeof(SKVLayerState));\n')
        f.write('\tst->num_layers = %d;\n' % (num_layer))
        f.write('\tst->layers = (void **)speex_alloc(st->num_layers * sizeof(void *));\n\n')
        
        f.write('\tBasicLayer        * basic_layer           = NULL;\n')
        f.write('\tVectorizedLayer   * vectorized_layer      = NULL;\n')
        f.write('\tActiveLayer       * active_layer          = NULL;\n')
        f.write('\tAffineLayer       * affine_layer          = NULL;\n')
        f.write('\tAffineActiveLayer * affine_active_layer   = NULL;\n')
        f.write('\tTDNNLayer         * tdnn_layer            = NULL;\n')
        f.write('\tTDNNActiveLayer   * tdnn_active_layer     = NULL;\n')
        f.write('\tGRULayer          * gru_layer             = NULL;\n')
        f.write('\tint layer = 0;\n')
        
        act_mean = net.input_mean
        act_std  = net.input_std
        out_min = 0
        out_max = act_mean + 3.0 * act_std
        max_abs = max(abs(out_min), abs(out_max))
        out_qParams = ChooseQuantizationQParams(max_abs.cpu(), QMAX)
        
        f.write('\n\n')
        f.write('\tst->layers[layer] = (void *)speex_alloc(sizeof(BasicLayer));\n')
        f.write('\tbasic_layer = (BasicLayer *)st->layers[layer];\n')
        f.write('\tbasic_layer->layer_type = INPUTLayer;\n')
        if 'netE' == model_name:
            f.write('\tbasic_layer->isQuantized = false;\n')
        else:
            f.write('\tbasic_layer->isQuantized = true;\n')
        f.write('\tbasic_layer->pre_ptr  = layer - 1;\n')
        f.write('\tbasic_layer->next_ptr = layer + 1;\n')
        f.write('\tbasic_layer->in_size     = %d;\n' % (input_layer_insize))
        f.write('\tbasic_layer->out_size    = %d;\n' % (input_layer_outsize))
        f.write('\tbasic_layer->out_qParams = %d;\n' % (out_qParams))
        f.write('\tbasic_layer->in_qParams  = %d;\n' % (in_qParams))
        f.write('\tlayer++;\n')
        in_qParams = out_qParams
        
        num_layer     = len(net.NNet)
        for i in range(num_layer):
            layer = net.NNet[i]
            num_affine = i + 1

            act_mean = layer.act_mean
            act_std  = layer.act_std
            out_min = 0
            out_max = act_mean + 3.0 * act_std
            max_abs = max(abs(out_min), abs(out_max))
            out_qParams = ChooseQuantizationQParams(max_abs.cpu(), QMAX)
            
            qweight_ih, qbias_ih, qparams_ih, qweight_hh, qbias_hh, qparams_hh = quantized_net[i]
            qparams = None
            if qparams_ih is not None:
                if qparams is None:
                    qparams = qparams_ih
                else:
                    qparams = np.concatenate((qparams, qparams_ih), axis = 0)
            if qparams_hh is not None:
                if qparams is None:
                    qparams = qparams_hh
                else:
                    qparams = np.concatenate((qparams, qparams_hh), axis = 0)

            outsize, insize = qweight_ih.shape
            h_size          = outsize / 3
            if hasattr(layer, 'rnn'):
                f.write('\n\n')
                f.write('\tst->layers[layer] = (void *)speex_alloc(sizeof(GRULayer));\n')
                f.write('\tgru_layer = (GRULayer *)st->layers[layer];\n')
                f.write('\tgru_layer->layer_type = GRUFFLayer;\n')
                f.write('\tgru_layer->pre_ptr = layer - 1;\n')
                f.write('\tgru_layer->next_ptr = layer + 1;\n')
                f.write('\tgru_layer->in_size    = %d;\n' % (insize))
                f.write('\tgru_layer->h_size     = %d;\n' % (h_size))
                f.write('\tgru_layer->out_size   = %d;\n' % (outsize))
                f.write('\tgru_layer->isQuantized= true;\n')
                f.write('\tgru_layer->out_qParams= %d;\n' % (out_qParams))
                f.write('\tgru_layer->in_qParams= %d;\n' % (in_qParams))
                f.write('\tlayer++;\n')
                f.write('\tgru_layer->layer_w = %s_weight_%d;\n' % (suffix_names, num_affine))
                f.write('\tgru_layer->layer_b = %s_bias_%d;\n' % (suffix_names, num_affine))
                f.write('\tgru_layer->layer_qParams = (skv_int16_t *)speex_alloc( %d * sizeof(skv_int16_t));\n' % (qparams.shape[0]))
                for r in range(qparams.shape[0]):
                    f.write('\tgru_layer->layer_qParams[%d] = %d;\n' % (r, qparams[r]))
                f.write('\tgru_layer->h_t = (skv_int16_t *)speex_alloc( num_output_frame * %d * sizeof(skv_int16_t));\n' % (h_size))
                f.write('\tSPEEX_MEMSET(gru_layer->h_t, 0, num_output_frame * %d);\n' % (h_size))
                
            elif hasattr(layer, 'tdnn'):
                kernel_size = layer.kernel_size
                stride      = layer.stride
                padding     = layer.padding
                dilation    = layer.dilation
                if hact_type == 'relu':
                    f.write('\n\n')
                    f.write('\tst->layers[layer] = (void *)speex_alloc(sizeof(TDNNActiveLayer));\n')
                    f.write('\ttdnn_active_layer = (TDNNActiveLayer *)st->layers[layer];\n')
                    f.write('\ttdnn_active_layer->layer_type = TDNNACTIVELayer;\n')
                    f.write('\ttdnn_active_layer->pre_ptr = layer - 1;\n')
                    f.write('\ttdnn_active_layer->next_ptr = layer + 1;\n')
                    f.write('\ttdnn_active_layer->in_size    = %d;\n' % (insize))
                    f.write('\ttdnn_active_layer->out_size   = %d;\n' % (outsize))
                    f.write('\ttdnn_active_layer->kernel_size   = %d;\n' % (kernel_size))
                    f.write('\ttdnn_active_layer->stride   = %d;\n' % (stride))
                    f.write('\ttdnn_active_layer->padding   = %d;\n' % (padding))
                    f.write('\ttdnn_active_layer->dilation   = %d;\n' % (dilation))
                    f.write('\ttdnn_active_layer->isQuantized= true;\n')
                    f.write('\ttdnn_active_layer->out_qParams= %d;\n' % (out_qParams))
                    f.write('\ttdnn_active_layer->in_qParams= %d;\n' % (in_qParams))
                    f.write('\ttdnn_active_layer->active_type= ReLU;\n')
                    f.write('\tlayer++;\n')
                    f.write('\ttdnn_active_layer->layer_w = %s_weight_%d;\n' % (suffix_names, num_affine))
                    f.write('\ttdnn_active_layer->layer_b = %s_bias_%d;\n' % (suffix_names, num_affine))
                    f.write('\ttdnn_active_layer->layer_qParams = (skv_int16_t *)speex_alloc( %d * sizeof(skv_int16_t));\n' % (qparams.shape[0]))
                    for r in range(qparams.shape[0]):
                        f.write('\ttdnn_active_layer->layer_qParams[%d] = %d;\n' % (r, qparams[r]))
                else:
                    f.write('\n\n')
                    f.write('\tst->layers[layer] = (void *)speex_alloc(sizeof(TDNNLayer));\n')
                    f.write('\ttdnn_layer = (TDNNLayer *)st->layers[layer];\n')
                    f.write('\ttdnn_layer->layer_type = TDNNFFLayer;\n')
                    f.write('\ttdnn_layer->pre_ptr = layer - 1;\n')
                    f.write('\ttdnn_layer->next_ptr = layer + 1;\n')
                    f.write('\ttdnn_layer->in_size    = %d;\n' % (insize))
                    f.write('\ttdnn_layer->out_size   = %d;\n' % (outsize))
                    f.write('\ttdnn_layer->kernel_size   = %d;\n' % (kernel_size))
                    f.write('\ttdnn_layer->stride   = %d;\n' % (stride))
                    f.write('\ttdnn_layer->padding   = %d;\n' % (padding))
                    f.write('\ttdnn_layer->dilation   = %d;\n' % (dilation))
                    f.write('\ttdnn_layer->isQuantized= true;\n')
                    f.write('\ttdnn_layer->out_qParams= %d;\n' % (out_qParams))
                    f.write('\ttdnn_layer->in_qParams= %d;\n' % (in_qParams))
                    f.write('\tlayer++;\n')
                    f.write('\ttdnn_layer->layer_w = %s_weight_%d;\n' % (suffix_names, num_affine))
                    f.write('\ttdnn_layer->layer_b = %s_bias_%d;\n' % (suffix_names, num_affine))
                    f.write('\ttdnn_layer->layer_qParams = (skv_int16_t *)speex_alloc( %d * sizeof(skv_int16_t));\n' % (qparams.shape[0]))
                    for r in range(outsize):
                        f.write('\ttdnn_layer->layer_qParams[%d] = %d;\n' % (r, qparams[r]))
            
            else:
                if hact_type == 'relu':
                    f.write('\n\n')
                    f.write('\tst->layers[layer] = (void *)speex_alloc(sizeof(AffineActiveLayer));\n')
                    f.write('\taffine_active_layer = (AffineActiveLayer *)st->layers[layer];\n')
                    f.write('\taffine_active_layer->layer_type = AFFINEACTIVELayer;\n')
                    f.write('\taffine_active_layer->pre_ptr = layer - 1;\n')
                    f.write('\taffine_active_layer->next_ptr = layer + 1;\n')
                    f.write('\taffine_active_layer->in_size    = %d;\n' % (insize))
                    f.write('\taffine_active_layer->out_size   = %d;\n' % (outsize))
                    f.write('\taffine_active_layer->isQuantized= true;\n')
                    f.write('\taffine_active_layer->out_qParams= %d;\n' % (out_qParams))
                    f.write('\taffine_active_layer->in_qParams= %d;\n' % (in_qParams))
                    f.write('\taffine_active_layer->active_type= ReLU;\n')
                    f.write('\tlayer++;\n')
                    f.write('\taffine_active_layer->layer_w = %s_weight_%d;\n' % (suffix_names, num_affine))
                    f.write('\taffine_active_layer->layer_b = %s_bias_%d;\n' % (suffix_names, num_affine))
                    f.write('\taffine_active_layer->layer_qParams = (skv_int16_t *)speex_alloc( %d * sizeof(skv_int16_t));\n' % (qparams.shape[0]))
                    for r in range(outsize):
                        f.write('\taffine_active_layer->layer_qParams[%d] = %d;\n' % (r, qparams[r]))
                else:
                    f.write('\n\n')
                    f.write('\tst->layers[layer] = (void *)speex_alloc(sizeof(AffineLayer));\n')
                    f.write('\taffine_layer = (AffineLayer *)st->layers[layer];\n')
                    f.write('\taffine_layer->layer_type = AFFINELayer;\n')
                    f.write('\taffine_layer->pre_ptr = layer - 1;\n')
                    f.write('\taffine_layer->next_ptr = layer + 1;\n')
                    f.write('\taffine_layer->in_size    = %d;\n' % (insize))
                    f.write('\taffine_layer->out_size   = %d;\n' % (outsize))
                    f.write('\taffine_layer->isQuantized= true;\n')
                    f.write('\taffine_layer->out_qParams= %d;\n' % (out_qParams))
                    f.write('\taffine_layer->in_qParams= %d;\n' % (in_qParams))
                    f.write('\tlayer++;\n')
                    f.write('\taffine_layer->layer_w = %s_weight_%d;\n' % (suffix_names, num_affine))
                    f.write('\taffine_layer->layer_b = %s_bias_%d;\n' % (suffix_names, num_affine))
                    f.write('\taffine_layer->layer_qParams = (skv_int16_t *)speex_alloc( %d * sizeof(skv_int16_t));\n' % (qparams.shape[0]))
                    for r in range(outsize):
                        f.write('\taffine_layer->layer_qParams[%d] = %d;\n' % (r, qparams[r]))
            in_qParams = out_qParams
            
            if oact_type.lower() == 'sigmoid':
                act_func = 'Sigmoid'
            elif oact_type.lower() == 'tanh':
                act_func = 'Tanh'
            elif oact_type.lower() == 'softmax':
                act_func = 'Softmax'
            else:
                act_func = None
            
            if act_func is not None:
                f.write('\n\n')
                f.write('\tst->layers[layer] = (void *)speex_alloc(sizeof(ActiveLayer));\n')
                f.write('\tactive_layer = (ActiveLayer *)st->layers[layer];\n')
                f.write('\tactive_layer->layer_type = ACTIVELayer;\n')
                f.write('\tactive_layer->pre_ptr = layer - 1;\n')
                f.write('\tactive_layer->next_ptr = layer + 1;\n')
                f.write('\tactive_layer->in_size    = %d;\n' % (outsize))
                f.write('\tactive_layer->out_size   = %d;\n' % (outsize))
                f.write('\tactive_layer->isQuantized= false;\n')
                f.write('\tactive_layer->out_qParams= 0;\n')
                f.write('\tactive_layer->in_qParams= 0;\n')
                f.write('\tactive_layer->active_type = %s;\n' % (act_func))
                f.write('\tlayer++;\n')
        
        outsize = net.output_size
        if act_func is None:
            out_qParams = in_qParams
            f.write('\n\n')
            f.write('\tst->layers[layer] = (void *)speex_alloc(sizeof(BasicLayer));\n')
            f.write('\tbasic_layer = (BasicLayer *)st->layers[layer];\n')
            f.write('\tbasic_layer->layer_type = OUTPUTLayer;\n')
            f.write('\tbasic_layer->pre_ptr = layer - 1;\n')
            f.write('\tbasic_layer->next_ptr = layer + 1;\n')
            f.write('\tbasic_layer->in_size    = %d;\n' % (outsize))
            f.write('\tbasic_layer->out_size   = %d;\n' % (outsize))
            f.write('\tbasic_layer->isQuantized= true;\n')
            f.write('\tbasic_layer->out_qParams= %d;\n' % (in_qParams))
            f.write('\tbasic_layer->in_qParams= %d;\n' % (out_qParams))
            f.write('\tlayer++;\n')
        else:
            f.write('\n\n')
            f.write('\tst->layers[layer] = (void *)speex_alloc(sizeof(BasicLayer));\n')
            f.write('\tbasic_layer = (BasicLayer *)st->layers[layer];\n')
            f.write('\tbasic_layer->layer_type = OUTPUTLayer;\n')
            f.write('\tbasic_layer->pre_ptr = layer - 1;\n')
            f.write('\tbasic_layer->next_ptr = layer + 1;\n')
            f.write('\tbasic_layer->in_size    = %d;\n' % (outsize))
            f.write('\tbasic_layer->out_size   = %d;\n' % (outsize))
            f.write('\tbasic_layer->isQuantized= false;\n')
            f.write('\tbasic_layer->out_qParams= 0;\n')
            f.write('\tbasic_layer->in_qParams= 0;\n')
            f.write('\tlayer++;\n')
        
        f.write('\n\n')
        f.write('\tst->num_output_frame = num_output_frame <= 0? %d : num_output_frame;\n' % (num_output_frame))
        f.write('\tif (preComputeLayerParam(st) == false)\n')
        f.write('\t{\n')
        f.write('\t\tst = skv_layers_destroy(st);\n')
        f.write('\t\tst = NULL;\n')
        f.write('\t}\n')
        f.write('\treturn st;\n')
        f.write('}\n\n\n')

    f.close()