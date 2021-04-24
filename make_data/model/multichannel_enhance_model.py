import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
from torch import nn
import torch.nn.functional as F

import math
import numpy as np
import scipy.io as sio

from .base_model import BaseModel, get_non_pad_mask, init_net, get_attn_pad_mask
from .networks import MultiEnhNet
from .phasen import PHASEN
from .conv_stft import ConvSTFT, ConviSTFT
from .fbanklayer import KaldiFbankModel, FbankModel
from .dcf import DCF
from .gsc import GSC
from .phasen import remove_dc

class multiple_single_enh(BaseModel):
    def __init__(self, basic_args, multiple_args, single_args):
            """Initialize the DNN AM class.
            Parameters:
                basic_args:
                    num_channel: number of channels, Default: 2
                    sample_rate: sample rate of audio, Default: 16000
                    win_len: length of windows for one frame, Default: 512
                    win_shift: shift of windows, Default: 256
                    nfft: size of FFT, Default: 512
                    win_type: type of windows, Default: hamming
                    num_bin: number of frequency bin, Default: nfft / 2 + 1
                    fbnak_size: number of filterbank bin for fbank extractor, Default: 80
                    cmvn_file: path to cmvn file

                multiple_args:
                    spect_type: type of estimated speech spectrum, value: fbank or spect
                    fft_consistency: If set to False, fft consistency constrain will be consider. Default: True
                    spect_weight: weight of spectrum loss, Default: 0.5
                    fft_weight: weight of fft loss: Default: 0.5

                    opt_type: type of optimizer, adadelta | adam | SGD
                    init_type: type of network initialization, normal | xavier | kaiming | orthogonal
                    init_gain: scaling factor for normal, xavier and orthogonal.
                    max_norm: Norm cutoff to prevent explosion of gradients

                    sf_num_beam: number of beamforming filters
                    sf_bias: If set to False, the layer will not learn an additive bias. Default: True
                    sf_fixed: If set to True, the spatial_filter layer will not be learn and updated. Default: True
                    sf_regularization_weight: regularization weight of spatial filter layer, Default: 0.1
                    sf_regularization_p: p-norm of spatial filter layer, Default: 1.0 or 2.0
                    sf_weight_init: None
                    sf_bias_init: None

                    fa_type: type of frequency alignment layer, fc or conv
                    fa_num_align: number of frequency alignment ways
                    fa_pooling_type: type of alignment pooling, agv or max
                    fa_num_freq_kernel: if the type of frequency alignment layer is conv, num_freq_kernel MUST be given, it means that the size along frequency axis
                    fa_bias: If set to False, the layer will not learn an additive bias. Default: True
                    fa_batch_norm: Default: False
                    fa_dropout: rate of dropout, Default 0.0
                    fa_fixed: If set to True, the spatial_filter layer will not be learn and updated. Default: True
                    fa_weight_init: None
                    fa_bias_init: None
                    fa_regularization_weight: regularization weight of spatial filter layer, Default: 0.1
                    fa_regularization_p: p-norm of spatial filter layer, Default: 1.0 or 2.0
                    
                single_args:
                    spect_type: type of estimated speech spectrum, value: fbank or spect
                    fft_consistency: If set to False, fft consistency constrain will be consider. Default: True
                    spect_weight: weight of spectrum loss, Default: 0.5
                    fft_weight: weight of fft loss: Default: 0.5

                    opt_type: type of optimizer, adadelta | adam | SGD
                    init_type: type of network initialization, normal | xavier | kaiming | orthogonal
                    init_gain: scaling factor for normal, xavier and orthogonal.
                    max_norm: Norm cutoff to prevent explosion of gradients

                    regularization_weight: regularization weight of spatial filter layer, Default: 0.1
                    regularization_p: p-norm of spatial filter layer, Default: 1.0 or 2.0

                    num_tsb: number of TSB block in PHASEN, Default: 3
                    channel_amp: number of conv channel for amplitude stream in PHASEN, Default: 24
                    channel_phase: number of conv channel for phase stream in PHASEN, Default: 12
                    rnn_nums: size of rnn hidden layer, Default: 300
                    fixed: If set to True, the phasen will not be learn and updated. Default: True
            """
            BaseModel.__init__(self, basic_args)
            
            ## Solving some compatibility issues for basic
            if not hasattr(basic_args, 'steps'):
                basic_args.steps = 0
            self.steps        = basic_args.steps
            
            basic_args.num_bin = int(basic_args.nfft / 2 + 1)

            self.basic_args    = basic_args
            self.multiple_args = multiple_args
            self.single_args   = single_args

            self.num_bin       = basic_args.num_bin
            self.num_channel   = self.basic_args.num_channel
            self.nfft          = self.basic_args.nfft
            self.compressed_scale     = self.basic_args.compressed_scale
            self.numerical_protection = self.basic_args.numerical_protection

            if basic_args.cmvn_file is not None and os.path.exists(basic_args.cmvn_file):
                self.input_cmvn  = torch.load(basic_args.cmvn_file)
                self.input_cmvn  = self.input_cmvn.to(self.device)
                print("Load cmvn from %s" % (basic_args.cmvn_file))
            else:
                self.input_cmvn  = None
            
            multiple_spect_weight = self.multiple_args.spect_weight
            multiple_fft_weight   = self.multiple_args.fft_weight
            single_spect_weight   = self.single_args.spect_weight
            single_fft_weight     = self.single_args.fft_weight

            #self.model_names = ['multipleEnh', 'singleEnh', 'convstft']
            self.model_names = []
            if multiple_spect_weight > 0.0 or multiple_fft_weight > 0.0:
                self.model_names.append('multipleEnh')
            if single_spect_weight > 0.0 or single_fft_weight > 0.0:
                self.model_names.append('singleEnh')
            self.model_names.append('convstft')
            
            ## define and initialize the convstft
            win_len   = self.basic_args.win_len
            win_inc   = self.basic_args.win_shift 
            fft_len   = self.basic_args.nfft
            win_type  = self.basic_args.win_type
            self.convstft  = ConvSTFT(win_len, win_inc, fft_len, win_type = win_type, feature_type='complex', fix = True)
            self.convstft.to(self.device)
            self.convistft = ConviSTFT(win_len, win_inc, fft_len, win_type = win_type, feature_type = 'complex', fix = True)
            self.convistft.to(self.device)

            ## define and initialize the MultiEnhNet
            num_bin               = self.basic_args.num_bin
            num_channel           = self.basic_args.num_channel
            num_beam              = self.multiple_args.sf_num_beam
            bias                  = self.multiple_args.sf_bias
            fixed                 = self.multiple_args.sf_fixed
            weight_init           = self.multiple_args.sf_weight_init
            bias_init             = self.multiple_args.sf_bias_init
            regularization_weight = self.multiple_args.sf_regularization_weight
            regularization_p      = self.multiple_args.sf_regularization_p
            spatial_filter_config = {'num_bin': num_bin, 'num_channel': num_channel, 'num_beam': num_beam, 'bias': bias, 'fixed': fixed, 'weight_init': weight_init, 
                        'bias_init': bias_init, 'regularization_weight': regularization_weight, 'regularization_p': regularization_p}
            
            fa_type                = self.multiple_args.fa_type
            num_align              = self.multiple_args.fa_num_align
            num_bin                = self.basic_args.num_bin
            num_beam               = self.multiple_args.sf_num_beam
            pooling_type           = self.multiple_args.fa_pooling_type
            num_freq_kernel        = self.multiple_args.fa_num_freq_kernel
            bias                   = self.multiple_args.fa_bias
            batch_norm             = self.multiple_args.fa_batch_norm
            dropout                = self.multiple_args.fa_dropout
            fixed                  = self.multiple_args.fa_fixed
            weight_init            = self.multiple_args.fa_weight_init
            bias_init              = self.multiple_args.fa_bias_init
            regularization_weight  = self.multiple_args.fa_regularization_weight
            regularization_p       = self.multiple_args.fa_regularization_p
            self.pooling_type      = pooling_type
            frequency_align_config = {'type': fa_type, 'num_align': num_align, 'num_bin': num_bin, 'num_beam': num_beam, 'pooling_type': pooling_type, 
                        'num_freq_kernel': num_freq_kernel, 'bias': bias, 'batch_norm': batch_norm, 'dropout': dropout, 'fixed': fixed, 'weight_init': weight_init, 'bias_init': bias_init, 'regularization_weight': regularization_weight, 'regularization_p': regularization_p}
            if multiple_spect_weight > 0.0 or multiple_fft_weight > 0.0:
                self.multipleEnh = MultiEnhNet(spatial_filter_config, frequency_align_config)
                self.multipleEnh = init_net(self.multipleEnh, self.multiple_args.init_type, self.multiple_args.init_gain, self.device)
            else:
                self.multipleEnh = None

            ## define and initialize the singleEnh
            num_bin                = self.basic_args.num_bin
            num_tsb                = self.single_args.num_tsb
            channel_amp            = self.single_args.channel_amp
            channel_phase          = self.single_args.channel_phase
            rnn_nums               = self.single_args.rnn_nums
            fixed                  = self.single_args.fixed
            regularization_weight  = self.single_args.regularization_weight
            regularization_p       = self.single_args.regularization_p
            if single_spect_weight > 0.0 or single_fft_weight > 0.0:
                self.singleEnh = PHASEN(num_bin = num_bin, num_tsb = num_tsb, channel_amp = channel_amp, channel_phase = channel_phase, rnn_nums = rnn_nums, regularization_weight = regularization_weight, regularization_p = regularization_p, fixed = fixed)
                self.singleEnh = init_net(self.singleEnh, self.single_args.init_type, self.single_args.init_gain, self.device)
            else:
                self.singleEnh = None

            # define and initialize DCF and GSC NetWorks
            if self.pooling_type is not None and self.pooling_type.lower() == 'attention':
                ## debug ##
                ## Load dcf targ beamformor
                filter_path = 'egs/huawei_enh/exp/'
                dcf_targ_bf = os.path.join(filter_path, 'dcf_targ_bf.mat')
                key         = sio.whosmat(dcf_targ_bf)[0][0]
                data        = sio.loadmat(dcf_targ_bf)
                if key in data:
                    dcf_targ_bf = data[key]  # MUST BE [1, 2, num_bin, num_channel] egs. [10, 2, 257, 5]
                    print(dcf_targ_bf.shape)
                else:
                    dcf_targ_bf = None
                    exit("Load dcf_targ_bf Failed!")

                ## Load dcf null beamformor
                dcf_null_bf = os.path.join(filter_path, 'dcf_null_bf.mat')
                key  = sio.whosmat(dcf_null_bf)[0][0]
                data = sio.loadmat(dcf_null_bf)
                if key in data:
                    dcf_null_bf = data[key]  # MUST BE [1, 2, num_bin, num_channel] egs. [20, 2, 257, 5]
                    print(dcf_null_bf.shape)
                else:
                    dcf_null_bf = None
                    exit("Load dcf_null_bf Failed!")

                ## Load gsc targ beamformor
                gsc_targ_bf = os.path.join(filter_path, 'gsc_targ_bf.mat')
                key = sio.whosmat(gsc_targ_bf)[0][0]
                data = sio.loadmat(gsc_targ_bf)
                if key in data:
                    gsc_targ_bf = data[key]  # MUST BE [1, 2, num_bin, num_channel] egs. [20, 2, 257, 5]
                    print(gsc_targ_bf.shape)
                else:
                    gsc_targ_bf = None
                    exit("Load gsc_targ_bf Failed!")

                ## Load gsc null beamformor
                gsc_null_bf = os.path.join(filter_path, 'gsc_null_bf.mat')
                key  = sio.whosmat(gsc_null_bf)[0][0]
                data = sio.loadmat(gsc_null_bf)
                if key in data:
                    gsc_null_bf = data[key]  # MUST BE [1, 2, num_bin, num_channel] egs. [20, 2, 257, 5]
                    print(gsc_null_bf.shape)
                else:
                    gsc_null_bf = None
                    exit("Load gsc_null_bf Failed!")

                self.multiple_args.dcf_targ_bf = dcf_targ_bf
                self.multiple_args.dcf_null_bf = dcf_null_bf
                self.multiple_args.gsc_targ_bf = gsc_targ_bf
                self.multiple_args.gsc_null_bf = gsc_null_bf
                ## debug ##

                dcf_num_beam, num_dim, num_bin, num_channel = self.multiple_args.dcf_targ_bf.shape
                dcf_num_null                                = int(self.multiple_args.dcf_null_bf.shape[0] / dcf_num_beam)
                print("dcf_num_beam = %d, num_bin = %d, num_channel = %d, dcf_num_null = %d" % ( dcf_num_beam, num_bin, num_channel, dcf_num_null) )
                self.dcf = DCF(num_beam = dcf_num_beam, num_null = dcf_num_null, num_bin = num_bin, num_channel = num_channel, targ_bf_weight = self.multiple_args.dcf_targ_bf, null_bf_weight = self.multiple_args.dcf_null_bf)
                self.dcf = self.dcf.to(self.device)

                gsc_num_beam, num_dim, num_bin, num_channel = self.multiple_args.gsc_targ_bf.shape
                gsc_num_null                                = int(self.multiple_args.gsc_null_bf.shape[0] / gsc_num_beam)
                print("gsc_num_beam = %d, num_bin = %d, num_channel = %d, gsc_num_null = %d" % ( gsc_num_beam, num_bin, num_channel, gsc_num_null) )
                self.gsc = GSC(num_beam = gsc_num_beam, num_null = gsc_num_null, num_bin = num_bin, num_channel = num_channel, targ_bf_weight = self.multiple_args.gsc_targ_bf, block_bf_weight = self.multiple_args.gsc_null_bf)
                self.gsc = self.gsc.to(self.device)

                self.num_tdoa = gsc_num_beam
            else:
                self.dcf      = None
                self.gsc      = None
                self.num_tdoa = 0

            # define and initialize fbank network
            if self.multiple_args.spect_type.lower() == 'fbank' or self.single_args.spect_type.lower() == 'fbank':
                self.fbankNet = KaldiFbankModel(nFFT = self.basic_args.nfft, nbank = self.basic_args.fbank_size, samplerate = self.basic_args.sample_rate, fixed = True)
                self.fbankNet.to(self.device)
                self.model_names.append('fbankNet')
            else:
                self.fbankNet = None
            
            ## loss_names = ['multipleFFT', 'multipleSpect', 'singleFFT', 'singleSpect']
            self.loss_names = []
            if self.multiple_args.spect_weight > 0.0:
                self.loss_names.append('multipleSpect')
            if self.multiple_args.fft_weight > 0.0:
                self.loss_names.append('multipleFFT')
            if self.single_args.spect_weight > 0.0:
                self.loss_names.append('singleSpect')
            if self.single_args.fft_weight > 0.0:
                self.loss_names.append('singleFFT')
            
            # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
            self.visual_names = ['mixture_spect']
            if self.multipleEnh is not None:
                self.visual_names.append('multiple_enh_spect')
                #self.visual_names.append('gsc_spect')
            if self.singleEnh is not None:
                self.visual_names.append('single_enh_spect')
                #self.visual_names.append('dcf_mask')
            self.visual_names.append('spe_spect')

            if self.basic_args.isTrain:
                ## define loss functions
                self.criterionL1 = torch.nn.L1Loss() # self.criterionL1 = torch.nn.MSELoss()
                
                ## define the optimizer for multipleEnh
                if self.multipleEnh is not None:
                    multipleEnh_parameters = self.multipleEnh.get_trainable_params(print_info = True)
                else:
                    multipleEnh_parameters = None
                if multipleEnh_parameters is not None:
                    if self.multiple_args.opt_type.lower() == 'adadelta':
                        self.multipleEnh_optimizer = torch.optim.Adadelta(multipleEnh_parameters, rho=0.95, eps=1e-6)
                    elif self.multiple_args.opt_type.lower() == 'adam':
                        self.multipleEnh_optimizer = torch.optim.Adam(multipleEnh_parameters, lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                        #self.multipleEnh_optimizer = torch.optim.SGD(multipleEnh_parameters, lr=basic_args.lr, momentum=0.9, weight_decay=1e-4)
                    elif self.multiple_args.opt_type.lower() == 'sgd':
                        self.multipleEnh_optimizer = torch.optim.SGD(multipleEnh_parameters, lr=basic_args.lr, momentum=0.9)
                    else:
                        self.multipleEnh_optimizer = torch.optim.Adam(multipleEnh_parameters, lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                else:
                    self.multipleEnh_optimizer = None
                if self.multipleEnh_optimizer is not None:
                    self.optimizers.append(self.multipleEnh_optimizer)
                
                ## define the optimizer for singleEnh
                if self.singleEnh is not None:
                    singleEnh_parameters = self.singleEnh.get_trainable_params(print_info = True)
                else:
                    singleEnh_parameters = None
                if singleEnh_parameters is not None:
                    if self.single_args.opt_type.lower() == 'adadelta':
                        self.singleEnh_optimizer = torch.optim.Adadelta(singleEnh_parameters, rho=0.95, eps=1e-6)
                    elif self.single_args.opt_type.lower() == 'adam':
                        self.singleEnh_optimizer = torch.optim.Adam(singleEnh_parameters, lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                        #self.singleEnh_optimizer = torch.optim.SGD(singleEnh_parameters, lr=basic_args.lr, momentum=0.9, weight_decay=1e-4)
                    elif self.single_args.opt_type.lower() == 'sgd':
                        self.singleEnh_optimizer = torch.optim.SGD(singleEnh_parameters, lr=basic_args.lr, momentum=0.9)
                    else:
                        self.singleEnh_optimizer = torch.optim.Adam(singleEnh_parameters, lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                else:
                    self.singleEnh_optimizer = None
                if self.singleEnh_optimizer is not None:
                    self.optimizers.append(self.singleEnh_optimizer)

            self.setup(basic_args)

    @classmethod
    def load_model(cls, model_path, continue_from_name = 'best', given_basic_args = None, given_multiple_args = None, given_single_args = None, gpu_ids = [1], isTrain = False):
        if continue_from_name is None:
            exit("ERROR: continue_from_model is None")
        
        continue_name = continue_from_name.split('-')[0]
        load_filename = 'model_%s.configure' % (continue_name)
        configure_path = os.path.join(model_path, load_filename)
        if not os.path.exists(configure_path):
            exit("ERROR: %s is not existed" % (configure_path))
        
        model_configure = torch.load(configure_path, map_location=lambda storage, loc: storage)
        
        if given_basic_args is None:
            basic_args   = model_configure['basic_args']
        else:
            basic_args = given_basic_args
        
        if given_multiple_args is None:
            multiple_args = model_configure['multiple_args']
        else:
            multiple_args = given_multiple_args
        
        if given_single_args is None:
            single_args = model_configure['single_args']
        else:
            single_args = given_single_args
                
        basic_args.model_dir          = model_path
        basic_args.continue_from_name = continue_from_name
        basic_args.gpu_ids            = gpu_ids
        basic_args.isTrain            = isTrain
        basic_args.steps              = model_configure['tr_steps']
        
        model = cls(basic_args, multiple_args, single_args)
        model.load_networks(continue_from_name)
        model.steps =  basic_args.steps

        # loss_names = ['multipleFFT', 'multipleSpect', 'singleFFT', 'singleSpect']
        model_state = {
            'epoch': model_configure['epoch'],
            'tr_steps': model_configure['tr_steps'],
            'val_steps': model_configure['val_steps'],
            'tr_multipleFFT_loss': model_configure['tr_multipleFFT_loss'],
            'tr_multipleSpect_loss': model_configure['tr_multipleSpect_loss'],
            'val_multipleFFT_loss': model_configure['val_multipleFFT_loss'],
            'val_multipleSpect_loss': model_configure['val_multipleSpect_loss'],
            'tr_singleFFT_loss': model_configure['tr_singleFFT_loss'],
            'tr_singleSpect_loss': model_configure['tr_singleSpect_loss'],
            'val_singleFFT_loss': model_configure['val_singleFFT_loss'],
            'val_singleSpect_loss': model_configure['val_singleSpect_loss']
        }
        return model, model_state, basic_args, multiple_args, single_args

    def save_model(self, suffix_name, epoch, val_steps, tr_multipleFFT_loss = None, tr_multipleSpect_loss = None, val_multipleFFT_loss = None, val_multipleSpect_loss = None, tr_singleFFT_loss = None, tr_singleSpect_loss = None, val_singleFFT_loss = None, val_singleSpect_loss = None):
        configure_package = {
            'epoch': epoch,
            'tr_steps': self.steps,
            'val_steps': val_steps,
            'tr_multipleFFT_loss': tr_multipleFFT_loss,
            'tr_multipleSpect_loss': tr_multipleSpect_loss,
            'val_multipleFFT_loss': val_multipleFFT_loss,
            'val_multipleSpect_loss': val_multipleSpect_loss,
            'tr_singleFFT_loss': tr_singleFFT_loss,
            'tr_singleSpect_loss': tr_singleSpect_loss,
            'val_singleFFT_loss': val_singleFFT_loss,
            'val_singleSpect_loss': val_singleSpect_loss,
            'basic_args': self.basic_args,
            'multiple_args': self.multiple_args,
            'single_args': self.single_args
        }
        save_filename = 'model_%s.configure' % (suffix_name)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(configure_package, save_path)
        self.save_networks(suffix_name)

    def set_input(self, input_audio, target_audio, input_lengths = None, tdoas = None):
        """
        Args:
            input_audio: (num_block, num_channel, num_sample)
            target_audio: (num_block, 1, num_sample)
            input_lengths: (num_block)
        """
        self.steps = self.steps + 1
        self.num_block, self.num_channel, self.num_sample = input_audio.size()
        if tdoas is not None and self.num_tdoa > 0:
            self.tdoas = tdoas / ( 360.0 / self.num_tdoa ) + 0.5
            self.tdoas = self.tdoas.int() % self.num_tdoa
        else:
            self.tdoas = None

        ## set the input feats
        input_audio  = input_audio.to(self.device)
        target_audio = target_audio.to(self.device)                                # (num_block, 1, num_sample)
        input_audio  = input_audio.view(self.num_block * self.num_channel, 1, -1)  # (num_block * num_channel, 1, num_sample)

        mFFT = self.convstft(input_audio)  # (num_block * num_channel, num_bin * 2, num_frame)
        sFFT = self.convstft(target_audio) # (num_block, num_bin * 2, num_frame)
        self.mmFFT = mFFT                  # (num_block * num_channel, num_bin * 2, num_frame)

        # apply cmvn to mFFT
        if self.input_cmvn is not None:
            add_shift = self.input_cmvn[0, :].squeeze()      # (num_bin * 2)
            add_shift = add_shift.unsqueeze(0).unsqueeze(2)  # (1, num_bin * 2, 1)
            add_shift = add_shift.expand_as(mFFT)            # (num_block * num_channel, num_bin * 2, num_frame)

            add_scale = self.input_cmvn[1, :].squeeze()      # (num_bin * 2)
            add_scale = add_scale.unsqueeze(0).unsqueeze(2)  # (1, num_bin * 2, 1)
            add_scale = add_scale.expand_as(mFFT)            # (num_block * num_channel, num_bin * 2, num_frame)

            mFFT = ( mFFT + add_shift ) * add_scale

        self.num_frame = min(mFFT.size(2), sFFT.size(2))
        mFFT = mFFT[:, :, :self.num_frame]  # (num_block * num_channel, num_bin * 2, num_frame)
        sFFT = sFFT[:, :, :self.num_frame]  # (num_block, num_bin * 2, num_frame)
        
        mFFT   = mFFT.view(self.num_block, self.num_channel, self.num_bin * 2, -1) # (num_block, num_channel, num_bin * 2, num_frame)
        mFFT_r = mFFT[:, :, :self.num_bin, :] #( num_block, num_channel, num_bin, num_frame)
        mFFT_i = mFFT[:, :, self.num_bin:, :] #( num_block, num_channel, num_bin, num_frame)

        mFFT_r = mFFT_r.permute([0, 3, 2, 1]).contiguous() #( num_block, num_frame, num_bin, num_channel)
        mFFT_i = mFFT_i.permute([0, 3, 2, 1]).contiguous() #( num_block, num_frame, num_bin, num_channel)

        mFFT_r = mFFT_r.view(self.num_block * self.num_frame, self.num_bin, self.num_channel) # ( num_block * num_frame, num_bin, num_channel)
        mFFT_i = mFFT_i.view(self.num_block * self.num_frame, self.num_bin, self.num_channel) # ( num_block * num_frame, num_bin, num_channel)

        self.mFFT = torch.cat([torch.unsqueeze(mFFT_r, 1), torch.unsqueeze(mFFT_i, 1)], dim = 1) # ( num_block * num_frame, 2, num_bin, num_channel )

        # (num_block, num_bin * 2, num_frame) --> ( num_block, 2, num_bin, num_frame )
        self.sFFT = torch.cat([torch.unsqueeze(sFFT[:, :self.num_bin, :], 1), torch.unsqueeze(sFFT[:, self.num_bin:, :], 1)], dim = 1) # ( num_block, 2, num_bin, num_frame )

    def compute_input_cmvn(self, input_audio = None, sum_all = None, sum_square_all = None, frame_count = 0, is_finished = False):
        self.num_block, self.num_channel, self.num_sample = input_audio.size()
        
        if self.input_cmvn is None:
            self.input_cmvn = torch.zeros(2, self.num_bin * 2, dtype = torch.float32)

        if sum_all is None or sum_square_all is None or frame_count < 1:
            sum_all         = torch.zeros(1, self.num_bin * 2, dtype = torch.float64)
            sum_square_all  = torch.zeros(1, self.num_bin * 2, dtype = torch.float64)
        
        sum_all         = sum_all.to(self.device)
        sum_square_all  = sum_square_all.to(self.device)
        self.input_cmvn = self.input_cmvn.to(self.device)

        with torch.no_grad():
            input_audio  = input_audio.to(self.device)
            input_audio  = input_audio.view(self.num_block * self.num_channel, 1, -1)

            mFFT = self.convstft(input_audio)      # (num_block * num_channel, num_bin * 2, num_frame)
            
            mFFT = mFFT.permute([0, 2, 1]).contiguous()  # (num_block * num_channel, num_frame, num_bin * 2)
            mFFT = mFFT.view(-1, self.num_bin * 2)       # (num_block * num_channel * num_frame, num_bin * 2)

            frame_count += mFFT.size(0)

            sum_all = sum_all + torch.sum(mFFT, dim = 0)
            sum_square_all = sum_square_all + torch.sum(mFFT**2, dim = 0)
        
        if frame_count > 0 and is_finished:
            mean = sum_all / frame_count
            var  = sum_square_all / frame_count - mean ** 2
            print(mean)
            print(var)
            self.input_cmvn[0, :] = -mean
            self.input_cmvn[1, :] = 1.0 / torch.sqrt(var)
            
        return self.input_cmvn, sum_all, sum_square_all, frame_count

    def inference(self, input_audio, tdoas):
        with torch.no_grad():
            num_block, num_channel, num_sample = input_audio.size()
            
            ## convert tdoa
            if tdoas is not None and self.num_tdoa > 0:
                self.tdoas = tdoas / ( 360.0 / self.num_tdoa ) + 0.5
                self.tdoas = self.tdoas.int() % self.num_tdoa
                print(tdoas)
                print(self.tdoas)
            else:
                self.tdoas = None

            ## set the input feats
            input_audio  = input_audio.to(self.device)                       # (num_block, num_channel, num_sample)
            input_audio  = input_audio.view(num_block * num_channel, 1, -1)  # (num_frame, 1, num_sample)

            mFFT = self.convstft(input_audio)  # (num_block * num_channel, num_bin * 2, num_frame)

            # apply cmvn to mFFT
            if self.input_cmvn is not None:
                add_shift = self.input_cmvn[0, :].squeeze()      # (num_bin * 2)
                add_shift = add_shift.unsqueeze(0).unsqueeze(2)  # (1, num_bin * 2, 1)
                add_shift = add_shift.expand_as(mFFT)            # (num_block * num_channel, num_bin * 2, num_frame)
    
                add_scale = self.input_cmvn[1, :].squeeze()      # (num_bin * 2)
                add_scale = add_scale.unsqueeze(0).unsqueeze(2)  # (1, num_bin * 2, 1)
                add_scale = add_scale.expand_as(mFFT)            # (num_block * num_channel, num_bin * 2, num_frame)
    
                mFFT = ( mFFT + add_shift ) * add_scale
                
            num_frame = mFFT.size(2)

            mFFT   = mFFT.view(num_block, num_channel, self.num_bin * 2, -1) # (num_block, num_channel, num_bin * 2, num_frame)
            mFFT_r = mFFT[:, :, :self.num_bin, :] #( num_block, num_channel, num_bin, num_frame)
            mFFT_i = mFFT[:, :, self.num_bin:, :] #( num_block, num_channel, num_bin, num_frame)

            mFFT_r = mFFT_r.permute([0, 3, 2, 1]).contiguous() #( num_block, num_frame, num_bin, num_channel)
            mFFT_i = mFFT_i.permute([0, 3, 2, 1]).contiguous() #( num_block, num_frame, num_bin, num_channel)

            mFFT_r = mFFT_r.view(num_block * num_frame, self.num_bin, num_channel) # ( num_block * num_frame, num_bin, num_channel)
            mFFT_i = mFFT_i.view(num_block * num_frame, self.num_bin, num_channel) # ( num_block * num_frame, num_bin, num_channel)

            mFFT = torch.cat([torch.unsqueeze(mFFT_r, 1), torch.unsqueeze(mFFT_i, 1)], dim = 1) # ( num_block * num_frame, 2, num_bin, num_channel )

            if self.multipleEnh is not None:
                if self.pooling_type is not None and self.pooling_type.lower() == 'attention':
                    dcf_mask_null, _ = self.dcf(input = mFFT, beam_id = self.tdoas, alpha = 0.35) # (num_block, num_frame, num_bin, num_null), (num_block, num_frame, 2, num_bin)
                    dcf_mask, _      = torch.min(dcf_mask_null, dim = -1, keepdim = False)                     # ( num_block, num_frame, num_bin )
                    gsc_targ_bf_out  = self.gsc(input = mFFT, mask = dcf_mask, beam_id = self.tdoas, alpha_v = 0.95) # ( num_block, num_frame, 2, num_bin)
                    
                    gsc_targ_bf_out  = gsc_targ_bf_out.permute([0, 2, 3, 1]).contiguous()                         # ( num_block, 2, num_bin, num_frame )
                    multiple_enh_fft = self.multipleEnh(mFFT, guid_info = gsc_targ_bf_out, num_block = num_block) # ( num_block * num_frame, 2, num_bin)

                    ## debug ##
                    ##multiple_enh_fft = gsc_targ_bf_out.view(-1, 2, self.num_bin)
                    ## debug ##
                else:
                    multiple_enh_fft = self.multipleEnh(mFFT, num_block = num_block) # ( num_block * num_frame, 2, num_bin)

                multiple_enh_fft = multiple_enh_fft.view(num_block, num_frame, 2, self.num_bin) # ( num_block, num_frame, 2, num_bin)
                multiple_enh_fft = multiple_enh_fft.permute([0, 2, 3, 1]).contiguous()          # ( num_block, 2, num_bin, num_frame )
                '''
                ## debug ##
                mFFT1 = mFFT[:, :, :, 0]                             # ( num_block * num_frame, 2, num_bin )
                mFFT1 = mFFT1.view(num_block, -1, 2, self.num_bin)   # ( num_block, num_frame, 2, num_bin )
                mFFT1 = mFFT1.permute([0, 2, 3, 1]).contiguous()     # ( num_block, 2, num_bin, num_frame )            

                multiple_enh_fft_tmp = multiple_enh_fft

                multiple_enh_fft_tmp[:, :, 138:256, :] = 0.1 * multiple_enh_fft_tmp[:, :, 138:256, :] + 0.9 * mFFT1[:, :, 138:256, :] # ( num_block, 2, num_bin, num_frame )
                est_fft = torch.cat([multiple_enh_fft_tmp[:,0], multiple_enh_fft_tmp[:,1]], 1) # ( num_block, num_bin * 2, num_frame )
                
                ## debug ##
                '''
                est_fft = torch.cat([multiple_enh_fft[:,0], multiple_enh_fft[:,1]], 1) # ( num_block, num_bin * 2, num_frame )
                multiple_enh_wav = self.convistft(est_fft)
                multiple_enh_wav = torch.squeeze(multiple_enh_wav, 1)
            else:
                multiple_enh_wav = None

            if self.singleEnh is not None and multiple_enh_fft is not None:
                single_enh_fft   = self.singleEnh(multiple_enh_fft)                        # ( num_block, 2, num_bin, num_frame )
                '''
                ## debug ##
                mFFT1 = mFFT[:, :, :, 0]                             # ( num_block * num_frame, 2, num_bin )
                mFFT1 = mFFT1.view(num_block, -1, 2, self.num_bin)   # ( num_block, num_frame, 2, num_bin )
                mFFT1 = mFFT1.permute([0, 2, 3, 1]).contiguous()     # ( num_block, 2, num_bin, num_frame )            

                single_enh_fft[:, :, 138:256, :] = 0.1 * single_enh_fft[:, :, 138:256, :] + 0.9 * mFFT1[:, :, 138:256, :] # ( num_block, 2, num_bin, num_frame )
                ## debug ##
                '''
                est_fft = torch.cat([single_enh_fft[:,0], single_enh_fft[:,1]], 1) # ( num_block, num_bin * 2, num_frame )
                single_enh_wav = self.convistft(est_fft)
                single_enh_wav = torch.squeeze(single_enh_wav, 1)  # (num_block, 1, num_sample)
            else:
                single_enh_wav = None

        return multiple_enh_wav, single_enh_wav

    def compute_visuals(self):
        # self.visual_names = ['mixture_spect', 'multiple_enh_spect', 'single_enh_spect', 'spe_spect']        
        with torch.no_grad():
            
            self.mmFFT = self.mmFFT.view(self.num_block, self.num_channel, self.num_bin * 2, -1)[:, :, :, :self.num_frame] # (num_block, num_channel, num_bin * 2, num_frame)
            
            mixture_spect       = self.mmFFT[:, 0, :self.num_bin, :] ** 2 + self.mmFFT[:, 0, self.num_bin:, :] ** 2 # (num_block, num_bin, num_frame)
            mixture_spect      = torch.clamp(mixture_spect, min=self.numerical_protection)
            mixture_spect       = mixture_spect.permute([0, 2, 1])                                                  # (num_block, num_frame , num_bin)
            mixture_spect       = torch.unsqueeze(mixture_spect, 1)                                                 # ( num_block, 1, num_frame , num_bin )
    
            if self.multiple_args.spect_type.lower() == 'fbank' or self.single_args.spect_type.lower() == 'fbank':
                self.mixture_spect = self.fbankNet(torch.sqrt(mixture_spect))                    # ( num_block, 1, num_frame , num_bin )
            else:
                self.mixture_spect = mixture_spect ** (0.5 * self.compressed_scale)              # ( num_block, 1, num_frame , num_bin )
    
    def forward(self):
        # mFFT: ( num_block * num_frame, 2, num_bin, num_channel )
        if self.multipleEnh is not None:
            if self.pooling_type is not None and self.pooling_type.lower() == 'attention':
                with torch.no_grad():
                    dcf_mask_null, _ = self.dcf(input = self.mFFT, beam_id = self.tdoas, alpha = 0.35) # (num_block, num_frame, num_bin, num_null), (num_block, num_frame, 2, num_bin)
                    dcf_mask, _      = torch.min(dcf_mask_null, dim = -1, keepdim = False)                          # ( num_block, num_frame, num_bin )
                
                    gsc_targ_bf_out  = self.gsc(input = self.mFFT, mask = dcf_mask, beam_id = self.tdoas, alpha_v = 0.95) # ( num_block, num_frame, 2, num_bin)
                    gsc_targ_bf_out  = gsc_targ_bf_out.permute([0, 2, 3, 1]).contiguous()                           # ( num_block, 2, num_bin, num_frame )

                    '''
                    self.dcf_mask    = dcf_mask.unsqueeze(1)                                                        # ( num_block, 1, num_frame, num_bin )
                    gsc_spect       = gsc_targ_bf_out[:, 0, :, :] ** 2 + gsc_targ_bf_out[:, 0, :, :] ** 2           # (num_block, num_bin, num_frame)
                    gsc_spect       = torch.clamp(gsc_spect, min=self.numerical_protection)
                    gsc_spect       = gsc_spect.permute([0, 2, 1])                                                  # (num_block, num_frame , num_bin)
                    gsc_spect       = torch.unsqueeze(gsc_spect, 1)                                                 # ( num_block, 1, num_frame , num_bin )
                    self.gsc_spect = gsc_spect ** (0.5 * self.compressed_scale)                                    # ( num_block, 1, num_frame , num_bin )
                    '''
                    
                self.multiple_enh_fft = self.multipleEnh(self.mFFT, guid_info = gsc_targ_bf_out, num_block = self.num_block) # ( num_block * num_frame, 2, num_bin)
            else:
                self.multiple_enh_fft = self.multipleEnh(self.mFFT, num_block = self.num_block) # ( num_block * num_frame, 2, num_bin)

            self.multiple_enh_fft = self.multiple_enh_fft.view(self.num_block, self.num_frame, 2, self.num_bin)        # ( num_block, num_frame, 2, num_bin)
            self.multiple_enh_fft = self.multiple_enh_fft.permute([0, 2, 3, 1]).contiguous()                           # ( num_block, 2, num_bin, num_frame )
        else:
            self.multiple_enh_fft = None
        
        if self.singleEnh is not None and self.multiple_enh_fft is not None:
            self.single_enh_fft   = self.singleEnh(self.multiple_enh_fft)                        # ( num_block, 2, num_bin, num_frame )
        else:
            self.single_enh_fft = None

        if self.multiple_args.fft_consistency and self.multiple_enh_fft is not None:
            self.multiple_enh_fft = self.multiple_enh_fft.permute([0, 3, 2, 1])              # [num_block, num_frame, num_bin, 2]
            self.multiple_enh_fft = torch.ifft(self.multiple_enh_fft, 3)
            self.multiple_enh_fft = torch.fft(self.multiple_enh_fft, 3)
            self.multiple_enh_fft = self.multiple_enh_fft.permute([0, 3, 2, 1])              # ( num_block, 2, num_bin, num_frame )

        if self.single_args.fft_consistency and self.single_enh_fft is not None:
            self.single_enh_fft = self.single_enh_fft.permute([0, 3, 2, 1])              # [num_block, num_frame, num_bin, 2]
            self.single_enh_fft = torch.ifft(self.single_enh_fft, 3)
            self.single_enh_fft = torch.fft(self.single_enh_fft, 3)
            self.single_enh_fft = self.single_enh_fft.permute([0, 3, 2, 1])              # ( num_block, 2, num_bin, num_frame )

        if self.multiple_enh_fft is not None:
            multiple_enh_spect = torch.unsqueeze(self.multiple_enh_fft[:,0,:,:] ** 2 + self.multiple_enh_fft[:,1,:,:] ** 2, 1)  # ( num_block, 1, num_bin, num_frame )
            multiple_enh_spect = torch.clamp(multiple_enh_spect, min=self.numerical_protection)
        else:
            multiple_enh_spect = None

        if self.single_enh_fft is not None:
            single_enh_spect   = torch.unsqueeze(self.single_enh_fft[:,0,:,:] ** 2 + self.single_enh_fft[:,1,:,:] ** 2, 1)      # ( num_block, 1, num_bin, num_frame )
            single_enh_spect   = torch.clamp(single_enh_spect, min=self.numerical_protection)
        else:
            single_enh_spect = None

        if self.sFFT is not None:
            spe_spect          = torch.unsqueeze(self.sFFT[:,0,:,:] ** 2 + self.sFFT[:,1,:,:] ** 2, 1)                          # ( num_block, 1, num_bin, num_frame )
            spe_spect          = torch.clamp(spe_spect, min=self.numerical_protection)
        else:
            spe_spect = None
        
        self.lowf_scale  = 1.0
        self.highf_scale = 1.0
        if self.compressed_scale != 1.0 and spe_spect is not None:
            #scale = (spe_spect + self.numerical_protection) ** (0.5 * (self.compressed_scale - 1.0))
            highf_scale = (spe_spect ** (0.5*0.3) / (1.0e-8 + spe_spect ** 0.5))
            highf_scale = torch.clamp(highf_scale, min=0.001, max = 100.0)
            self.highf_scale = torch.cat((highf_scale, highf_scale), dim = 1)

            lowf_scale = spe_spect ** (0.5 * self.compressed_scale)  # ( num_block, 1, num_bin, num_frame )
            self.lowf_scale = torch.cat((lowf_scale, lowf_scale), dim = 1)
        
        if multiple_enh_spect is not None:
            multiple_enh_spect = multiple_enh_spect.permute([0, 1, 3, 2])                 # ( num_block, 1, num_frame, num_bin )
        if single_enh_spect is not None:
            single_enh_spect   = single_enh_spect.permute([0, 1, 3, 2])                   # ( num_block, 1, num_frame, num_bin )
        if spe_spect is not None:
            spe_spect          = spe_spect.permute([0, 1, 3, 2])                          # ( num_block, 1, num_frame, num_bin )

        if multiple_enh_spect is not None:
            if self.multiple_args.spect_type.lower() == 'fbank':
                self.multiple_enh_spect = self.fbankNet(torch.sqrt(multiple_enh_spect))
            else:
                self.multiple_enh_spect = multiple_enh_spect ** (0.5 * self.compressed_scale)
        else:
            self.multiple_enh_spect = None

        if single_enh_spect is not None:
            if self.single_args.spect_type.lower() == 'fbank':
                self.single_enh_spect = self.fbankNet(torch.sqrt(single_enh_spect))
            else:
                self.single_enh_spect = single_enh_spect ** (0.5 * self.compressed_scale)
        else:
            self.single_enh_spect = None
        
        if spe_spect is not None:
            if self.multiple_args.spect_type.lower() == 'fbank' or self.single_args.spect_type.lower() == 'fbank':
                self.spe_spect = self.fbankNet(torch.sqrt(spe_spect))
            else:
                self.spe_spect = spe_spect ** (0.5 * self.compressed_scale)
        else:
            self.spe_spect = None
        
    def test(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.backward_E(perform_backward = False)

    def backward_E(self, perform_backward = True):
        losses = None

        multiple_spect_weight = self.multiple_args.spect_weight
        multiple_fft_weight   = self.multiple_args.fft_weight

        single_spect_weight   = self.single_args.spect_weight
        single_fft_weight     = self.single_args.fft_weight
        
        # loss_names = ['multipleFFT', 'multipleSpect', 'singleFFT', 'singleSpect']
        if multiple_fft_weight > 0.0 and self.multiple_enh_fft is not None:
            self.loss_multipleFFT = 0.5 * self.criterionL1(self.highf_scale * self.multiple_enh_fft, self.highf_scale * self.sFFT) * self.num_bin * 2 + 0.5 * self.criterionL1(self.lowf_scale * self.multiple_enh_fft, self.lowf_scale * self.sFFT) * self.num_bin * 2
            #self.loss_multipleFFT = self.criterionL1(self.lowf_scale * self.multiple_enh_fft, self.lowf_scale * self.sFFT) * self.num_bin * 2
            if losses is None:
                losses = multiple_fft_weight * self.loss_multipleFFT
            else:
                losses = losses + multiple_fft_weight * self.loss_multipleFFT
        else:
            self.loss_multipleFFT   = 0.0

        if multiple_spect_weight > 0.0 and self.multiple_enh_spect is not None:
            self.loss_multipleSpect = self.criterionL1(self.multiple_enh_spect, self.spe_spect) * self.num_bin * 2
            if losses is None:
                losses = multiple_spect_weight * self.loss_multipleSpect
            else:
                losses = losses + multiple_spect_weight * self.loss_multipleSpect
        else:
            self.loss_multipleSpect = 0.0
        
        if single_fft_weight > 0.0 and self.single_enh_fft is not None:
            self.loss_singleFFT   = 0.5 * self.criterionL1(self.highf_scale * self.single_enh_fft, self.highf_scale * self.sFFT) * self.num_bin * 2 + 0.5 * self.criterionL1(self.lowf_scale * self.single_enh_fft, self.lowf_scale * self.sFFT) * self.num_bin * 2                
            if losses is None:
                losses = single_fft_weight * self.loss_singleFFT
            else:
                losses = losses + single_fft_weight * self.loss_singleFFT
        else:
            self.loss_singleFFT = 0.0

        if single_spect_weight > 0.0 and self.single_enh_spect is not None:
            self.loss_singleSpect = self.criterionL1(self.single_enh_spect, self.spe_spect) * self.num_bin * 2
            if losses is None:
                losses = single_spect_weight * self.loss_singleSpect
            else:
                losses = losses + single_spect_weight * self.loss_singleSpect
        else:
            self.loss_singleSpect = 0.0

        if self.multipleEnh is not None:
            multipleEnh_penality      = self.multipleEnh.get_regularization_loss()
        else:
            multipleEnh_penality = 0.0
        if multipleEnh_penality > 0.0:
            losses = losses + multipleEnh_penality

        if self.singleEnh is not None:
            singleEnh_penality        = self.singleEnh.get_regularization_loss()
        else:
            singleEnh_penality = 0.0
        if singleEnh_penality > 0.0:
            losses = losses + singleEnh_penality
        
        if perform_backward and losses is not None:
            losses.backward()

    def zero_grad(self):
        if self.singleEnh_optimizer is not None:
            self.singleEnh_optimizer.zero_grad()

        if self.multipleEnh_optimizer is not None:
            self.multipleEnh_optimizer.zero_grad()
    
    def step(self):
        ## singleEnh_optimizer step
        if self.singleEnh_optimizer is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.singleEnh.parameters(), self.single_args.max_norm)
            if math.isnan(grad_norm) or math.isinf(grad_norm):
                print('grad norm is nan. Do not update singleEnh.')  
            else:
                self.singleEnh_optimizer.step()

        ## multipleEnh_optimizer step
        if self.multipleEnh_optimizer is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.multipleEnh.parameters(), self.multiple_args.max_norm)
            if math.isnan(grad_norm) or math.isinf(grad_norm):
                print('grad norm is nan. Do not update multipleEnh.')  
            else:
                self.multipleEnh_optimizer.step()

    def optimize_parameters(self):
        # Perform forward 
        self.forward()

        # Set zero_grad to the optimizer
        self.zero_grad()

        # Perform backward
        self.backward_E()

        # Update the weight
        self.step()
