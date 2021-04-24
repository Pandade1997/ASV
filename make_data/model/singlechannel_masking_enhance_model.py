import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
from torch import nn
import torch.nn.functional as F
import codecs

import math
import numpy as np
import scipy.io as sio

from .base_model import BaseModel, get_non_pad_mask, init_net, get_attn_pad_mask
from .networks import DeepRNNNet, Outputer, InferenceBatchSigmoid
from .conv_stft import ConvSTFT, ConviSTFT
from .fbanklayer import KaldiFbankModel, FbankModel
from .phasen import remove_dc

class single_mask_enh(BaseModel):
    def __init__(self, basic_args, mask_args):
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

                mask_args:
                    spect_type: type of estimated speech spectrum, value: fbank or spect
                    fft_consistency: If set to False, fft consistency constrain will be consider. Default: True
                    spect_weight: weight of spectrum loss, Default: 0.5
                    mask_weight: weight of mask loss, Default: 0.5
                    fft_weight: weight of fft loss: Default: 0.5
                    mask_type: type of mask, rfft_mask | cfft_mask | fft_filter | None
                    mask_act_type: type of mask activation, None|relu|sigmoid|softmax|tanh|softplus|prelu|leakyrelu

                    opt_type: type of optimizer, adadelta | adam | SGD
                    init_type: type of network initialization, normal | xavier | kaiming | orthogonal
                    init_gain: scaling factor for normal, xavier and orthogonal.
                    max_norm: Norm cutoff to prevent explosion of gradients

                    regularization_weight: regularization weight of spatial filter layer, Default: 0.1
                    regularization_p: p-norm of spatial filter layer, Default: 1.0 or 2.0
                    
                    layer_size: size of rnn hidden layer, Default: [40, 128, 128, 128, 257]
                    rnn_type: type of rnn, gru | lstm | rnn
                    bidirectional: If set to True, RNN is bidirectional, otherwise False, Default: False
                    batch_norm: Default: False
                    bias: Default: True
                    dropout: Default: 0.0
            """
            BaseModel.__init__(self, basic_args)
            
            ## Solving some compatibility issues for basic
            if not hasattr(basic_args, 'steps'):
                basic_args.steps = 0
            self.steps        = basic_args.steps
            
            basic_args.num_bin = int(basic_args.nfft / 2 + 1)

            self.basic_args    = basic_args
            self.mask_args     = mask_args
            
            self.num_bin              = self.basic_args.num_bin
            self.nfft                 = self.basic_args.nfft
            self.compressed_scale     = self.basic_args.compressed_scale
            self.numerical_protection = self.basic_args.numerical_protection

            ## define and initialize the convstft
            self.model_names = ['netE', 'spe_mask_prj', 'convstft']
            win_len        = self.basic_args.win_len
            win_inc        = self.basic_args.win_shift
            fft_len        = self.basic_args.nfft
            win_type       = self.basic_args.win_type
            self.convstft  = ConvSTFT(win_len, win_inc, fft_len, win_type = win_type, feature_type='complex', fix = True)
            self.convstft.to(self.device)
            self.convistft = ConviSTFT(win_len, win_inc, fft_len, win_type = win_type, feature_type = 'complex', fix = True)
            self.convistft.to(self.device)

            ## define and initialize the netE
            layer_size      = self.mask_args.layer_size
            rnn_type        = self.mask_args.rnn_type
            bidirectional   = self.mask_args.bidirectional
            batch_norm      = self.mask_args.batch_norm
            bias            = self.mask_args.bias
            dropout         = self.mask_args.dropout
            self.netE       = DeepRNNNet(layer_size = layer_size, rnn_type = rnn_type, bidirectional = bidirectional, batch_norm = batch_norm, bias = bias, dropout = dropout)
            self.netE       = init_net(self.netE, self.mask_args.init_type, self.mask_args.init_gain, self.device)
            
            ## define and initialize the spe_mask_prj
            in_size      = layer_size[-1]
            if self.mask_args.mask_type.lower() ==  'rfft_mask':
                out_size = int(self.basic_args.nfft / 2) + 1
            else:
                out_size  = self.basic_args.nfft + 2
            mask_act_type = self.mask_args.mask_act_type
            bias          = self.mask_args.bias
            self.spe_mask_prj = Outputer(in_size = in_size, out_size = out_size, out_act_type = mask_act_type, bias = bias)
            self.spe_mask_prj = init_net(self.spe_mask_prj, self.mask_args.init_type, self.mask_args.init_gain, self.device)

            # define and initialize fbank network
            if self.mask_args.spect_type.lower() == 'fbank':
                self.fbankNet = KaldiFbankModel(nFFT = self.basic_args.nfft, nbank = self.basic_args.fbank_size, samplerate = self.basic_args.sample_rate, fixed = True)
                self.fbankNet.to(self.device)
                self.model_names.append('fbankNet')
            else:
                self.fbankNet = None
            
            spect_weight   = self.mask_args.spect_weight
            fft_weight     = self.mask_args.fft_weight
            mask_weight    = self.mask_args.mask_weight

            # loss_names = ['Spect', 'FFT', 'Mask']
            self.loss_names = []
            if spect_weight > 0.0:
                self.loss_names.append('Spect')
            if fft_weight > 0.0:
                self.loss_names.append('FFT')
            if mask_weight > 0.0:
                self.loss_names.append('Mask')
            
            # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
            self.visual_names = ['mixture_spect', 'spe_enh_spect', 'emask', 'spe_spect']
            #self.visual_names = ['mixture_spect', 'spe_enh_spect', 'emask', 'mask']
            
            if self.basic_args.isTrain:
                ## define loss functions
                self.criterionL1 = torch.nn.L1Loss()
                #self.criterionL1 = torch.nn.MSELoss()
                
                ## define the optimizer for netE
                if self.netE is not None:
                    netE_parameters = self.netE.parameters()
                else:
                    netE_parameters = None
                if netE_parameters is not None:
                    if self.mask_args.opt_type.lower() == 'adadelta':
                        self.netE_optimizer = torch.optim.Adadelta(netE_parameters, rho=0.95, eps=1e-6)
                    elif self.mask_args.opt_type.lower() == 'adam':
                        self.netE_optimizer = torch.optim.Adam(netE_parameters, lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                        # self.netE_optimizer = torch.optim.SGD(netE_parameters, lr=basic_args.lr, momentum=0.9, weight_decay=1e-4)
                    elif self.mask_args.opt_type.lower() == 'sgd':
                        self.netE_optimizer = torch.optim.SGD(netE_parameters, lr=basic_args.lr, momentum=0.9)
                    else:
                        self.netE_optimizer = torch.optim.Adam(netE_parameters, lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                else:
                    self.netE_optimizer = None
                if self.netE_optimizer is not None:
                    self.optimizers.append(self.netE_optimizer)
                
                ## define the optimizer for spe_mask_prj
                if self.spe_mask_prj is not None:
                    spe_mask_prj_parameters = self.spe_mask_prj.parameters()
                else:
                    spe_mask_prj_parameters = None
                if spe_mask_prj_parameters is not None:
                    if self.mask_args.opt_type.lower() == 'adadelta':
                        self.spe_mask_prj_optimizer = torch.optim.Adadelta(spe_mask_prj_parameters, rho=0.95, eps=1e-6)
                    elif self.mask_args.opt_type.lower() == 'adam':
                        self.spe_mask_prj_optimizer = torch.optim.Adam(spe_mask_prj_parameters, lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                        #self.spe_mask_prj_optimizer = torch.optim.SGD(spe_mask_prj_parameters, lr=basic_args.lr, momentum=0.9, weight_decay=1e-4)
                    elif self.mask_args.opt_type.lower() == 'sgd':
                        self.spe_mask_prj_optimizer = torch.optim.SGD(spe_mask_prj_parameters, lr=basic_args.lr, momentum=0.9)
                    else:
                        self.spe_mask_prj_optimizer = torch.optim.Adam(spe_mask_prj_parameters, lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                else:
                    self.spe_mask_prj_optimizer = None
                if self.spe_mask_prj_optimizer is not None:
                    self.optimizers.append(self.spe_mask_prj_optimizer)
            self.setup(basic_args)

    @classmethod
    def load_model(cls, model_path, continue_from_name = 'best', given_basic_args = None, given_mask_args = None, gpu_ids = [1], isTrain = False):
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
        
        if given_mask_args is None:
            mask_args = model_configure['mask_args']
        else:
            mask_args = given_mask_args
                
        basic_args.model_dir          = model_path
        basic_args.continue_from_name = continue_from_name
        basic_args.gpu_ids            = gpu_ids
        basic_args.isTrain            = isTrain
        basic_args.steps              = model_configure['tr_steps']
        
        model = cls(basic_args, mask_args)
        model.load_networks(continue_from_name)
        model.steps =  basic_args.steps

        # loss_names =  ['Spect', 'FFT', 'Mask']
        model_state = {
            'epoch': model_configure['epoch'],
            'tr_steps': model_configure['tr_steps'],
            'val_steps': model_configure['val_steps'],
            'tr_FFT_loss': model_configure['tr_FFT_loss'],
            'tr_Spect_loss': model_configure['tr_Spect_loss'],
            'tr_Mask_loss': model_configure['tr_Mask_loss'],
            'val_FFT_loss': model_configure['val_FFT_loss'],
            'val_Spect_loss': model_configure['val_Spect_loss'],
            'val_Mask_loss': model_configure['val_Mask_loss']
        }
        return model, model_state, basic_args, mask_args

    def save_model(self, suffix_name, epoch, val_steps, tr_FFT_loss = None, tr_Spect_loss = None, tr_Mask_loss = None, val_FFT_loss = None, val_Spect_loss = None, val_Mask_loss = None):
        configure_package = {
            'epoch': epoch,
            'tr_steps': self.steps,
            'val_steps': val_steps,
            'tr_FFT_loss': tr_FFT_loss,
            'tr_Spect_loss': tr_Spect_loss,
            'tr_Mask_loss': tr_Mask_loss,
            'val_FFT_loss': val_FFT_loss,
            'val_Spect_loss': val_Spect_loss,
            'val_Mask_loss': val_Mask_loss,
            'basic_args': self.basic_args,
            'mask_args': self.mask_args
        }
        save_filename = 'model_%s.configure' % (suffix_name)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(configure_package, save_path)
        self.save_networks(suffix_name)

    def print_statistical_information(self, suffix_name = None):
        if suffix_name is not None:
            save_filename = '%s.txt' % (suffix_name)
            out_file_name = os.path.join(self.save_dir, save_filename)
            f = codecs.open(out_file_name, 'w', 'utf-8')
        else:
            f = None
            return
        
        ## write the statistical_information of maskE
        print("maskE_layes bn_min bn_max bn_mean bn_std act_min act_max act_mean act_std")
        f.write("maskE_layes bn_min bn_max bn_mean bn_std act_min act_max act_mean act_std\r\n")
        
        if hasattr(self.netE, 'NNet'):
            num_layer = len(self.netE.NNet)
            for layer in range(num_layer):
                bn_min  = self.netE.NNet[layer].bn_min
                bn_max  = self.netE.NNet[layer].bn_max
                bn_mean = self.netE.NNet[layer].bn_mean
                bn_std  = self.netE.NNet[layer].bn_std
    
                act_min  = self.netE.NNet[layer].act_min
                act_max  = self.netE.NNet[layer].act_max
                act_mean = self.netE.NNet[layer].act_mean
                act_std  = self.netE.NNet[layer].act_std
    
                print("netE_layer_%d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\r\n" % (layer, bn_min, bn_max, bn_mean, bn_std, act_min, act_max, act_mean, act_std))
                f.write("netE_layer_%d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\r\n" % (layer, bn_min, bn_max, bn_mean, bn_std, act_min, act_max, act_mean, act_std))
        
        if self.spe_mask_prj is not None and hasattr(self.spe_mask_prj, 'NNet'):
            num_layer = len(self.spe_mask_prj.NNet)
            for layer in range(num_layer):
                bn_min  = self.spe_mask_prj.NNet[layer].bn_min
                bn_max  = self.spe_mask_prj.NNet[layer].bn_max
                bn_mean = self.spe_mask_prj.NNet[layer].bn_mean
                bn_std  = self.spe_mask_prj.NNet[layer].bn_std
    
                act_min  = self.spe_mask_prj.NNet[layer].act_min
                act_max  = self.spe_mask_prj.NNet[layer].act_max
                act_mean = self.spe_mask_prj.NNet[layer].act_mean
                act_std  = self.spe_mask_prj.NNet[layer].act_std
    
                print("mask_out_prj_layer_%d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\r\n" % (layer, bn_min, bn_max, bn_mean, bn_std, act_min, act_max, act_mean, act_std))
                f.write("mask_out_prj_layer_%d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\r\n" % (layer, bn_min, bn_max, bn_mean, bn_std, act_min, act_max, act_mean, act_std))
        
        f.close()

    def set_input(self, input, mix_audio, target_audio, input_lengths = None):
        """
        Args:
            input: (num_block, num_frame, num_dim)
            mix_audio: (num_block, 1, num_sample)
            target_audio: (num_block, 1, num_sample)
            input_lengths: (num_block)
        """
        self.steps = self.steps + 1
        self.num_block, self.num_channel, self.num_sample = mix_audio.size()

        ## set the input feats
        self.input   = input.to(self.device)            # (num_block, num_frame, num_dim)
        mix_audio    = mix_audio.to(self.device)        # (num_block, 1, num_sample)
        target_audio = target_audio.to(self.device)     # (num_block, 1, num_sample)
        
        mFFT = self.convstft(mix_audio)    # (num_block, num_bin * 2, num_frame)
        sFFT = self.convstft(target_audio) # (num_block, num_bin * 2, num_frame)
        
        self.num_frame = min(mFFT.size(2), sFFT.size(2), self.input.size(1))
        mFFT           = mFFT[:, :, :self.num_frame]        # (num_block, num_bin * 2, num_frame)
        sFFT           = sFFT[:, :, :self.num_frame]        # (num_block, num_bin * 2, num_frame)
        self.input     = self.input[:, :self.num_frame, :]  # (num_block, num_frame, num_dim)

        mFFT = mFFT.permute([0, 2, 1]).contiguous()         #( num_block, num_frame, num_bin * 2)
        sFFT = sFFT.permute([0, 2, 1]).contiguous()         #( num_block, num_frame, num_bin * 2)

        self.mFFT = torch.cat([torch.unsqueeze(mFFT[:, :, :self.num_bin], 1), torch.unsqueeze(mFFT[:, :, self.num_bin:], 1)], dim = 1) # ( num_block, 2, num_frame, num_bin )
        self.sFFT = torch.cat([torch.unsqueeze(sFFT[:, :, :self.num_bin], 1), torch.unsqueeze(sFFT[:, :, self.num_bin:], 1)], dim = 1) # ( num_block, 2, num_frame, num_bin )
        self.nFFT = self.mFFT - self.sFFT

    def inference(self, input, mix_audio, target_audio = None):
        with torch.no_grad():
            
            num_block, num_channel, num_sample = mix_audio.size()

            ## set the input feats
            input        = input.to(self.device)            # (num_block, num_frame, num_dim)
            mix_audio    = mix_audio.to(self.device)        # (num_block, 1, num_sample)
            mFFT         = self.convstft(mix_audio)         # (num_block, num_bin * 2, num_frame)

            num_frame    = min(mFFT.size(2), input.size(1))

            mFFT  = mFFT[:, :, :num_frame]                  # (num_block, num_bin * 2, num_frame)
            input = input[:, :num_frame, :]                 # (num_block, num_frame, num_dim)
            mFFT  = mFFT.permute([0, 2, 1]).contiguous()    #( num_block, num_frame, num_bin * 2)
            mFFT  = torch.cat([torch.unsqueeze(mFFT[:, :, :self.num_bin], 1), torch.unsqueeze(mFFT[:, :, self.num_bin:], 1)], dim = 1) # ( num_block, 2, num_frame, num_bin )

            if target_audio is not None:
                target_audio = target_audio.to(self.device)     # (num_block, 1, num_sample)
                sFFT = self.convstft(target_audio)              # (num_block, num_bin * 2, num_frame)
                sFFT = sFFT[:, :, :num_frame]                   # (num_block, num_bin * 2, num_frame)
                sFFT = sFFT.permute([0, 2, 1]).contiguous()     #( num_block, num_frame, num_bin * 2)
                sFFT = torch.cat([torch.unsqueeze(sFFT[:, :, :self.num_bin], 1), torch.unsqueeze(sFFT[:, :, self.num_bin:], 1)], dim = 1) # ( num_block,2,num_frame,num_bin )
                nFFT = mFFT - sFFT
            else:
                sFFT = None
                nFFT = None

            if self.netE is not None:
                netE_out = self.netE(input) # (num_block, num_frame, output_size)
            else:
                netE_out = None

            if self.spe_mask_prj is not None and netE_out is not None:
                emask = self.spe_mask_prj(netE_out) # (num_block, num_frame, num_bin) or (num_block, num_frame, num_bin * 2)
            else:
                emask = None
            
            if self.mask_args.mask_type.lower() == 'rfft_mask':
                emask        = torch.unsqueeze(emask, 1)  # (num_block, 1, num_frame, num_bin)
                spe_enh_fft  = emask * mFFT               # (num_block, 2, num_frame, num_bin)
            elif self.mask_args.mask_type.lower() == 'cfft_mask':
                emask = torch.cat((torch.unsqueeze(emask[:, :, :self.num_bin], 1), torch.unsqueeze(emask[:, :, self.num_bin:], 1)), 1) # (num_block, 2, num_frame, num_bin)
                spe_enh_fft = emask * mFFT                                                                                             # (num_block, 2, num_frame, num_bin)
            else:
                emask = torch.cat((torch.unsqueeze(emask[:, :, :self.num_bin], 1), torch.unsqueeze(emask[:, :, self.num_bin:], 1)), 1) # (num_block, 2, num_frame, num_bin)
                spe_enh_rfft = emask[:,0,:,:] * mFFT[:,0,:,:] - emask[:,1,:,:] * mFFT[:,1,:,:]
                spe_enh_ifft = emask[:,0,:,:] * mFFT[:,1,:,:] + emask[:,1,:,:] * mFFT[:,0,:,:]
                spe_enh_fft = torch.cat((torch.unsqueeze(spe_enh_rfft, 1), torch.unsqueeze(spe_enh_ifft, 1)), 1)                        # (num_block, 2, num_frame, num_bin)

            if sFFT is not None:
                spe_spect          = torch.unsqueeze(sFFT[:,0,:,:] ** 2 + sFFT[:,1,:,:] ** 2, 1) # ( num_block, 1, num_frame, num_bin )
                spe_spect          = torch.clamp(spe_spect, min=self.numerical_protection)
            else:
                spe_spect = None
            if nFFT is not None:
                noi_spect          = torch.unsqueeze(nFFT[:,0,:,:] ** 2 + nFFT[:,1,:,:] ** 2, 1) # ( num_block, 1, num_frame, num_bin )
                noi_spect          = torch.clamp(noi_spect, min=self.numerical_protection)
            else:
                noi_spect = None

            if self.mask_args.mask_weight > 0.0 and spe_spect is not None and noi_spect is not None:
                mask = (spe_spect / (spe_spect + noi_spect)) ** 0.5   # ( num_block, 1, num_frame, num_bin )
                mask = torch.clamp(mask, min=1.0e-3)                  # ( num_block, 1, num_frame, num_bin )
                sFFT      = mask * mFFT                               # ( num_block, 2, num_frame, num_bin )
            else:
                mask = None
                sFFT = None

            if spe_enh_fft is not None:
                spe_enh_fft = spe_enh_fft.permute([0, 1, 3, 2]).contiguous()     # ( num_block, 2, num_bin, num_frame )
                est_fft     = torch.cat([spe_enh_fft[:,0], spe_enh_fft[:,1]], 1) # ( num_block, num_bin * 2, num_frame )
                spe_enh_wav = self.convistft(est_fft)
                spe_enh_wav = torch.squeeze(spe_enh_wav, 1)
            else:
                spe_enh_wav = None
            
            if sFFT is not None:
                sFFT    = sFFT.permute([0, 1, 3, 2]).contiguous()   # ( num_block, 2, num_bin, num_frame )
                est_fft = torch.cat([sFFT[:,0], sFFT[:,1]], 1)      # ( num_block, num_bin * 2, num_frame )
                spe_wav = self.convistft(est_fft)
                spe_wav = torch.squeeze(spe_wav, 1)
            else:
                spe_wav = None
        return spe_enh_wav, spe_wav

    def compute_visuals(self):
        # visual_names = ['mixture_spect', 'spe_enh_spect', 'mask', 'spe_spect']      
        with torch.no_grad():

            mixture_spect = self.mFFT[:, 0, :, :] ** 2 + self.mFFT[:, 1, :, :] ** 2   # (num_block, num_frame, num_bin)
            mixture_spect = torch.clamp(mixture_spect, min=self.numerical_protection)
            mixture_spect = torch.unsqueeze(mixture_spect, 1)                         # ( num_block, 1, num_frame , num_bin )
    
            if self.mask_args.spect_type.lower() == 'fbank':
                self.mixture_spect = self.fbankNet(torch.sqrt(mixture_spect))         # ( num_block, 1, num_frame , num_bin )
            else:
                self.mixture_spect = mixture_spect ** (0.5 * self.compressed_scale)   # ( num_block, 1, num_frame , num_bin )
            
            if self.emask is not None and self.mask_args.mask_type.lower() != 'rfft_mask':
                emask      = self.emask[:, 0, :, :] ** 2 + self.emask[:, 1, :, :] ** 2   # (num_block, num_frame, num_bin)
                emask      = torch.clamp(emask, min=self.numerical_protection)
                emask      = torch.unsqueeze(emask, 1)                                   # ( num_block, 1, num_frame , num_bin )
                self.emask = emask ** (0.5 * self.compressed_scale)                       # ( num_block, 1, num_frame , num_bin )

    def forward(self):
        # input: (num_block, num_frame, num_dim)
        # mFTT:  (num_block, 2, num_frame, num_bin)
        # sFFT:  (num_block, 2, num_frame, num_bin)
        if self.netE is not None:
            netE_out = self.netE(self.input)           # (num_block, num_frame, output_size)
        else:
            netE_out = None

        if self.spe_mask_prj is not None and netE_out is not None:
            emask = self.spe_mask_prj(netE_out) # (num_block, num_frame, num_bin) or (num_block, num_frame, num_bin * 2)
        else:
            emask = None

        if self.mask_args.mask_type.lower() == 'rfft_mask':
            self.emask        = torch.unsqueeze(emask, 1)  # (num_block, 1, num_frame, num_bin)
            self.spe_enh_fft  = self.emask * self.mFFT     # (num_block, 2, num_frame, num_bin)
        elif self.mask_args.mask_type.lower() == 'cfft_mask':
            self.emask = torch.cat((torch.unsqueeze(emask[:, :, :self.num_bin], 1), torch.unsqueeze(emask[:, :, self.num_bin:], 1)), 1) # (num_block, 2, num_frame, num_bin)
            self.spe_enh_fft = self.emask * self.mFFT                                                                                   # (num_block, 2, num_frame, num_bin)
        else:
            self.emask = torch.cat((torch.unsqueeze(emask[:, :, :self.num_bin], 1), torch.unsqueeze(emask[:, :, self.num_bin:], 1)), 1) # (num_block, 2, num_frame, num_bin)
            spe_enh_rfft = self.emask[:,0,:,:] * self.mFFT[:,0,:,:] - self.emask[:,1,:,:] * self.mFFT[:,1,:,:]
            spe_enh_ifft = self.emask[:,0,:,:] * self.mFFT[:,1,:,:] + self.emask[:,1,:,:] * self.mFFT[:,0,:,:]
            self.spe_enh_fft = torch.cat((torch.unsqueeze(spe_enh_rfft, 1), torch.unsqueeze(spe_enh_ifft, 1)), 1)                    # (num_block, 2, num_frame, num_bin)

        if self.mask_args.fft_consistency and self.spe_enh_fft is not None:
            self.spe_enh_fft = self.spe_enh_fft.permute([0, 2, 3, 1])              # (num_block, num_frame, num_bin, 2)
            self.spe_enh_fft = torch.ifft(self.spe_enh_fft, 3)
            self.spe_enh_fft = torch.fft(self.spe_enh_fft, 3)
            self.spe_enh_fft = self.spe_enh_fft.permute([0, 3, 1, 2])              # ( num_block, 2, num_frame, num_bin )

        if self.spe_enh_fft is not None:
            spe_enh_spect = torch.unsqueeze(self.spe_enh_fft[:,0,:,:] ** 2 + self.spe_enh_fft[:,1,:,:] ** 2, 1)  # ( num_block, 1, num_frame, num_bin )
            spe_enh_spect = torch.clamp(spe_enh_spect, min=self.numerical_protection)
            if self.mask_args.spect_type.lower() == 'fbank':
                self.spe_enh_spect = self.fbankNet(torch.sqrt(spe_enh_spect))
            else:
                self.spe_enh_spect = spe_enh_spect ** (0.5 * self.compressed_scale)
        else:
            self.spe_enh_spect = None

        if self.sFFT is not None:
            spe_spect          = torch.unsqueeze(self.sFFT[:,0,:,:] ** 2 + self.sFFT[:,1,:,:] ** 2, 1) # ( num_block, 1, num_frame, num_bin )
            spe_spect          = torch.clamp(spe_spect, min=self.numerical_protection)
        else:
            spe_spect = None
        if self.nFFT is not None:
            noi_spect          = torch.unsqueeze(self.nFFT[:,0,:,:] ** 2 + self.nFFT[:,1,:,:] ** 2, 1) # ( num_block, 1, num_frame, num_bin )
            noi_spect          = torch.clamp(noi_spect, min=self.numerical_protection)
        else:
            noi_spect = None

        if self.mask_args.mask_weight > 0.0:
            self.mask = (spe_spect / (spe_spect + noi_spect)) ** 0.5   # ( num_block, 1, num_frame, num_bin )
            self.mask = torch.clamp(self.mask, min=1.0e-3)             # ( num_block, 1, num_frame, num_bin )
        else:
            self.mask = None

        if self.mask is not None:
            self.sFFT = self.mask * self.mFFT                          # ( num_block, 1, num_frame, num_bin )
            spe_spect = torch.unsqueeze(self.sFFT[:,0,:,:] ** 2 + self.sFFT[:,1,:,:] ** 2, 1) # ( num_block, 1, num_frame, num_bin )
            spe_spect = torch.clamp(spe_spect, min=self.numerical_protection)
        
        self.lowf_scale  = 1.0
        self.highf_scale = 1.0
        if self.compressed_scale != 1.0 and spe_spect is not None:
            #scale = (spe_spect + self.numerical_protection) ** (0.5 * (self.compressed_scale - 1.0))
            highf_scale = (spe_spect ** (0.5 * self.compressed_scale) / (1.0e-8 + spe_spect ** 0.5))
            highf_scale = torch.clamp(highf_scale, min=0.001, max = 100.0)
            self.highf_scale = torch.cat((highf_scale, highf_scale), dim = 1)

            lowf_scale = spe_spect ** (0.5 * self.compressed_scale)  # ( num_block, 1, num_bin, num_frame )
            self.lowf_scale = torch.cat((lowf_scale, lowf_scale), dim = 1)

        if spe_spect is not None:
            if self.mask_args.spect_type.lower() == 'fbank':
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
        
        fft_weight     = self.mask_args.fft_weight
        spect_weight   = self.mask_args.spect_weight
        mask_weight    = self.mask_args.mask_weight

        # loss_names = ['Spect', 'FFT', 'Mask']
        if fft_weight > 0.0 and self.spe_enh_fft is not None:
            self.loss_FFT = 0.1 * self.criterionL1(self.highf_scale * self.spe_enh_fft, self.highf_scale * self.sFFT) * self.num_bin * 2 + 0.9 * self.criterionL1(self.lowf_scale * self.spe_enh_fft, self.lowf_scale * self.sFFT) * self.num_bin * 2
            #self.loss_FFT = self.criterionL1(self.spe_enh_fft, self.sFFT) * self.num_bin * 2
            if losses is None:
                losses = fft_weight * self.loss_FFT
            else:
                losses = losses + fft_weight * self.loss_FFT
        else:
            self.loss_FFT   = 0.0

        if spect_weight > 0.0 and self.spe_enh_spect is not None:
            self.loss_Spect = self.criterionL1(self.spe_enh_spect, self.spe_spect) * self.num_bin * 2
            if losses is None:
                losses = spect_weight * self.loss_Spect
            else:
                losses = losses + spect_weight * self.loss_Spect
        else:
            self.loss_Spect = 0.0
        
        if mask_weight > 0.0 and self.emask is not None and self.mask is not None:
            self.loss_Mask   = self.criterionL1(self.emask, self.mask) * self.num_bin * 2      
            if losses is None:
                losses = mask_weight * self.loss_Mask
            else:
                losses = losses + mask_weight * self.loss_Mask
        else:
            self.loss_Mask = 0.0

        if perform_backward and losses is not None:
            losses.backward()

    def zero_grad(self):
        if self.netE_optimizer is not None:
            self.netE_optimizer.zero_grad()

        if self.spe_mask_prj_optimizer is not None:
            self.spe_mask_prj_optimizer.zero_grad()
    
    def step(self):
        ## netE_optimizer step
        if self.netE_optimizer is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.netE.parameters(), self.mask_args.max_norm)
            if math.isnan(grad_norm) or math.isinf(grad_norm):
                print('grad norm is nan. Do not update singleEnh.')  
            else:
                self.netE_optimizer.step()

        ## spe_mask_prj_optimizer step
        if self.spe_mask_prj_optimizer is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.spe_mask_prj.parameters(), self.mask_args.max_norm)
            if math.isnan(grad_norm) or math.isinf(grad_norm):
                print('grad norm is nan. Do not update multipleEnh.')  
            else:
                self.spe_mask_prj_optimizer.step()

    def optimize_parameters(self):
        # Perform forward 
        self.forward()

        # Set zero_grad to the optimizer
        self.zero_grad()

        # Perform backward
        self.backward_E()

        # Update the weight
        self.step()
