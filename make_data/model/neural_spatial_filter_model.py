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
from .networks import Neural_Spatial_Filter
from .conv_stft import ConvSTFT, ConviSTFT
from .fbanklayer import KaldiFbankModel, FbankModel

class neural_spatial_filter(BaseModel):
    def __init__(self, basic_args, multiple_args):
            """Initialize the DNN EM class.
            Parameters:
                basic_args:
                    num_channel: number of channels, Default: 2
                    sample_rate: sample rate of audio, Default: 16000
                    mic_pos: position of microphone below fomat, egs: [(x1, y1), (x2, y2), ...] 
                    sound_speed: speed of sound wave in the air
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
                    mask_weight: weight of mask loss: Default: 0.5
                    
                    opt_type: type of optimizer, adadelta | adam | SGD
                    init_type: type of network initialization, normal | xavier | kaiming | orthogonal
                    init_gain: scaling factor for normal, xavier and orthogonal.
                    max_norm: Norm cutoff to prevent explosion of gradients

                    num_beam: number of beamformer for enhance directions of dcf
                    dpr_targ_bf: beamformer filter coefficients for enhance directions of dcf
                    dpr_fix: If set to True, the dcf layer will not be learn and updated. Default: True

                    pair_id: pair id for computing directionary features
                    do_IPD: If set to True, using IPD feature, otherwise False, Default: True
                    do_cosIPD: If set to True, using cosIPD feature, otherwise False, Default: True
                    do_sinIPD: If set to True, using sinIPD feature, otherwise False, Default: True
                    do_ILD: If set to True, using ILD feature, otherwise False, Default: True

                    rnn_layer_size: size of rnn hidden layer, Default: [257 * 4, 512, 512, 512]
                    rnn_type: type of rnn, gru | lstm | rnn | None
                    bidirectional: If set to True, RNN is bidirectional, otherwise False, Default: False
                    
                    mask_layer_size: size of mask net, Default: [512, 256, 257]
                    mask_act_type: type of mask activation, None|relu|sigmoid|softmax|tanh|softplus|prelu|leakyrelu

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
            self.multiple_args = multiple_args

            self.num_bin              = basic_args.num_bin
            self.num_channel          = basic_args.num_channel
            self.nfft                 = basic_args.nfft
            self.compressed_scale     = basic_args.compressed_scale
            self.numerical_protection = basic_args.numerical_protection

            if hasattr(basic_args, 'cmvn_file') and basic_args.cmvn_file is not None and os.path.exists(basic_args.cmvn_file):
                print("Load cmvn from %s" % (basic_args.cmvn_file))
                self.input_cmvn  = torch.load(basic_args.cmvn_file)
                self.input_cmvn  = self.input_cmvn.to(self.device)
                if math.isnan(self.input_cmvn[0, 0]) or math.isinf(self.input_cmvn[0, 0]):
                    self.input_cmvn[0, 0] = 0.0
                if math.isnan(self.input_cmvn[1, 0]) or math.isinf(self.input_cmvn[1, 0]):
                    self.input_cmvn[1, 0] = 0.0
            else:
                self.input_cmvn  = None
            
            spect_weight = self.multiple_args.spect_weight
            fft_weight   = self.multiple_args.fft_weight
            mask_weight  = self.multiple_args.mask_weight
            
            #self.model_names = ['multipleEnh', 'convstft']
            self.model_names = []
            if spect_weight > 0.0 or fft_weight > 0.0 or mask_weight > 0.0:
                self.model_names.append('multipleEnh')
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

            ## define and initialize the DCFGSCEnhNet
            num_bin                      = self.num_bin
            num_channel                  = self.num_channel

            dpr_config                   = {}
            dpr_config['num_bin']        = num_bin
            dpr_config['num_channel']    = num_channel
            dpr_config['num_beam']       = multiple_args.num_beam
            dpr_config['targ_bf_weight'] = multiple_args.dpr_targ_bf
            dpr_config['fix']            = multiple_args.dpr_fix
            
            df_config                    = {}
            df_config['num_bin']         = self.num_bin
            df_config['num_channel']     = self.num_channel
            df_config['mic_pos']         = self.basic_args.mic_pos
            df_config['sound_speed']     = self.basic_args.sound_speed
            df_config['sample_rate']     = self.basic_args.sample_rate
            df_config['pair_id']         = self.multiple_args.pair_id
            df_config['do_IPD']          = self.multiple_args.do_IPD
            df_config['do_cosIPD']       = self.multiple_args.do_cosIPD
            df_config['do_sinIPD']       = self.multiple_args.do_sinIPD
            df_config['do_ILD']          = self.multiple_args.do_ILD
            
            net_config = {}
            net_config['spect_feat_size'] = 0
            net_config['rnn_layer_size']  = multiple_args.rnn_hlayer_size
            net_config['rnn_type']        = multiple_args.rnn_type
            net_config['bidirectional']   = multiple_args.bidirectional
            net_config['mask_layer_size'] = multiple_args.mask_hlayer_size
            net_config['mask_act_type']   = multiple_args.mask_act_type
            net_config['batch_norm']      = multiple_args.batch_norm
            net_config['bias']            = multiple_args.bias
            net_config['dropout']         = multiple_args.dropout

            if hasattr(multiple_args, 'spect_feats') and multiple_args.spect_feats is not None:
                net_config['spect_feat_size'] = self.basic_args.fbank_size if self.multiple_args.spect_feats.lower() == 'fbank' else num_bin
            else:
                net_config['spect_feat_size']  = 0
                self.multiple_args.spect_feats = None
            self.spect_feat_size = net_config['spect_feat_size']
            
            if spect_weight > 0.0 or fft_weight > 0.0 or mask_weight > 0.0:
                self.multipleEnh = Neural_Spatial_Filter(dpr_config = dpr_config, df_config = df_config, net_config = net_config)
                self.multipleEnh = init_net(self.multipleEnh, multiple_args.init_type, multiple_args.init_gain, self.device)
            else:
                self.multipleEnh = None

            self.num_tdoa = multiple_args.num_beam

            # define and initialize fbank network
            if self.multiple_args.spect_type.lower() == 'fbank' or ( self.multiple_args.spect_feats is not None and self.multiple_args.spect_feats.lower() == 'fbank' ):
                self.fbankNet = KaldiFbankModel(nFFT = self.basic_args.nfft, nbank = self.basic_args.fbank_size, samplerate = self.basic_args.sample_rate, fixed = True)
                self.fbankNet.to(self.device)
                self.model_names.append('fbankNet')
            else:
                self.fbankNet = None
            
            self.loss_names = ['FFT', 'Spect', 'Mask']
            
            ## visual_names = ['mixture_spect', 'mask', 'multiple_enh_spect', 'spe_spect']
            self.visual_names = ['mixture_spect']
            if self.multipleEnh is not None:
                self.visual_names.append('emask')
                self.visual_names.append('multiple_enh_spect')
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
            self.setup(basic_args)

    @classmethod
    def load_model(cls, model_path, continue_from_name = 'best', given_basic_args = None, given_multiple_args = None, gpu_ids = [1], isTrain = False):
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
        
        basic_args.model_dir          = model_path
        basic_args.continue_from_name = continue_from_name
        basic_args.gpu_ids            = gpu_ids
        basic_args.isTrain            = isTrain
        basic_args.steps              = model_configure['tr_steps']
        
        model = cls(basic_args, multiple_args)
        model.load_networks(continue_from_name)
        model.steps =  basic_args.steps

        model_state = {
            'epoch': model_configure['epoch'],
            'tr_steps': model_configure['tr_steps'],
            'val_steps': model_configure['val_steps'],
            'tr_FFT_loss': model_configure['tr_FFT_loss'],
            'tr_Spect_loss': model_configure['tr_Spect_loss'],
            'val_FFT_loss': model_configure['val_FFT_loss'],
            'val_Spect_loss': model_configure['val_Spect_loss'],
            'tr_Mask_loss': model_configure['tr_Mask_loss'],
            'val_Mask_loss': model_configure['val_Mask_loss']
        }
        return model, model_state, basic_args, multiple_args

    def save_model(self, suffix_name, epoch, val_steps, tr_FFT_loss = None, tr_Spect_loss = None, val_FFT_loss = None, val_Spect_loss = None, tr_Mask_loss = None, val_Mask_loss = None):
        configure_package = {
            'epoch': epoch,
            'tr_steps': self.steps,
            'val_steps': val_steps,
            'tr_FFT_loss': tr_FFT_loss,
            'tr_Spect_loss': tr_Spect_loss,
            'val_FFT_loss': val_FFT_loss,
            'val_Spect_loss': val_Spect_loss,
            'tr_Mask_loss': tr_Mask_loss,
            'val_Mask_loss': val_Mask_loss,
            'basic_args': self.basic_args,
            'multiple_args': self.multiple_args
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

        self.tdoas = tdoas

        ## set the input feats
        input_audio  = input_audio.to(self.device)
        target_audio = target_audio.to(self.device)                                # (num_block, 1, num_sample)
        input_audio  = input_audio.view(self.num_block * self.num_channel, 1, -1)  # (num_block * num_channel, 1, num_sample)

        mFFT = self.convstft(input_audio)  # (num_block * num_channel, num_bin * 2, num_frame)
        sFFT = self.convstft(target_audio) # (num_block, num_bin * 2, num_frame)
        self.mmFFT = mFFT                  # (num_block * num_channel, num_bin * 2, num_frame)

        self.num_frame = min(mFFT.size(2), sFFT.size(2))
        mFFT = mFFT[:, :, :self.num_frame]  # (num_block * num_channel, num_bin * 2, num_frame)
        sFFT = sFFT[:, :, :self.num_frame]  # (num_block, num_bin * 2, num_frame)

        mFFT   = mFFT.view(self.num_block, self.num_channel, self.num_bin * 2, -1) # (num_block, num_channel, num_bin * 2, num_frame)

        nFFT   = mFFT[:, 0, :, :] - sFFT      # (num_block, num_bin * 2, num_frame)

        mFFT_r = mFFT[:, :, :self.num_bin, :] #( num_block, num_channel, num_bin, num_frame)
        mFFT_i = mFFT[:, :, self.num_bin:, :] #( num_block, num_channel, num_bin, num_frame)

        mFFT_r = mFFT_r.permute([0, 3, 2, 1]).contiguous() #( num_block, num_frame, num_bin, num_channel)
        mFFT_i = mFFT_i.permute([0, 3, 2, 1]).contiguous() #( num_block, num_frame, num_bin, num_channel)

        mFFT_r = mFFT_r.view(self.num_block * self.num_frame, self.num_bin, self.num_channel) # ( num_block * num_frame, num_bin, num_channel)
        mFFT_i = mFFT_i.view(self.num_block * self.num_frame, self.num_bin, self.num_channel) # ( num_block * num_frame, num_bin, num_channel)
        
        self.mFFT = torch.cat([torch.unsqueeze(mFFT_r, 1), torch.unsqueeze(mFFT_i, 1)], dim = 1) # ( num_block * num_frame, 2, num_bin, num_channel )

        # (num_block, num_bin * 2, num_frame) --> ( num_block, 2, num_bin, num_frame )
        self.sFFT = torch.cat([torch.unsqueeze(sFFT[:, :self.num_bin, :], 1), torch.unsqueeze(sFFT[:, self.num_bin:, :], 1)], dim = 1) # ( num_block, 2, num_bin, num_frame )

        self.nFFT = torch.cat([torch.unsqueeze(nFFT[:, :self.num_bin, :], 1), torch.unsqueeze(nFFT[:, self.num_bin:, :], 1)], dim = 1) # ( num_block, 2, num_bin, num_frame )

    def compute_input_cmvn(self, input_audio, tdoas, sum_all = None, sum_square_all = None, frame_count = 0, is_finished = False):
        
        num_block, num_channel, num_sample = input_audio.size()
                
        self.tdoas = tdoas

        if self.input_cmvn is None:
            self.input_cmvn = torch.zeros(2, self.spect_feat_size, dtype = torch.float32)

        if sum_all is None or sum_square_all is None or frame_count < 1:
            sum_all         = torch.zeros(1, self.spect_feat_size, dtype = torch.float64)
            sum_square_all  = torch.zeros(1, self.spect_feat_size, dtype = torch.float64)
        
        sum_all         = sum_all.to(self.device)
        sum_square_all  = sum_square_all.to(self.device)
        self.input_cmvn = self.input_cmvn.to(self.device)

        with torch.no_grad():
            input_audio  = input_audio.to(self.device)
            input_audio  = input_audio.view(num_block * num_channel, 1, -1)

            mFFT = self.convstft(input_audio)  # (num_block * num_channel, num_bin * 2, num_frame)

            num_frame = mFFT.size(2)

            mFFT   = mFFT.view(num_block, num_channel, self.num_bin * 2, -1) # (num_block, num_channel, num_bin * 2, num_frame)
            
            mFFT_r = mFFT[:, :, :self.num_bin, :] #( num_block, num_channel, num_bin, num_frame)
            mFFT_i = mFFT[:, :, self.num_bin:, :] #( num_block, num_channel, num_bin, num_frame)

            mFFT_r = mFFT_r.permute([0, 3, 2, 1]).contiguous() #( num_block, num_frame, num_bin, num_channel)
            mFFT_i = mFFT_i.permute([0, 3, 2, 1]).contiguous() #( num_block, num_frame, num_bin, num_channel)

            mFFT_r = mFFT_r.view(num_block * num_frame, self.num_bin, num_channel) # ( num_block * num_frame, num_bin, num_channel)
            mFFT_i = mFFT_i.view(num_block * num_frame, self.num_bin, num_channel) # ( num_block * num_frame, num_bin, num_channel)

            mFFT = torch.cat([torch.unsqueeze(mFFT_r, 1), torch.unsqueeze(mFFT_i, 1)], dim = 1) # ( num_block * num_frame, 2, num_bin, num_channel )

            if self.multiple_args.spect_feats is not None:
                ehFFT = self.multipleEnh.fix_bf_enhance(mFFT, self.tdoas) #( num_block, num_frame, 2, num_bin )

                spect_feats   = torch.unsqueeze(ehFFT[:, :, 0, :] ** 2 + ehFFT[:, :, 1, :] ** 2, 1) # ( num_block, 1, num_frame, num_bin )
                spect_feats   = torch.clamp(spect_feats, min=self.numerical_protection)             # ( num_block, 1, num_frame, num_bin )
                
                if self.multiple_args.spect_feats.lower() == 'fbank':
                    spect_feats = self.fbankNet(torch.sqrt(spect_feats))        # ( num_block, 1, num_frame, spect_feat_size )
                else:
                    #spect_feats = spect_feats ** (0.5 * self.compressed_scale) # ( num_block, 1, num_frame, spect_feat_size )
                    spect_feats = torch.log(spect_feats)                        # ( num_block, 1, num_frame, spect_feat_size )
                
                spect_feats = spect_feats.view(-1, self.spect_feat_size)        # ( num_block * num_frame, spect_feat_size )
            else:
                spect_feats = None
                return self.input_cmvn, sum_all, sum_square_all, frame_count

            frame_count += spect_feats.size(0)

            sum_all = sum_all + torch.sum(spect_feats, dim = 0)
            sum_square_all = sum_square_all + torch.sum(spect_feats**2, dim = 0)
        
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
            self.tdoas = tdoas

            ## set the input feats
            input_audio  = input_audio.to(self.device)                       # (num_block, num_channel, num_sample)
            input_audio  = input_audio.view(num_block * num_channel, 1, -1)  # (num_frame, 1, num_sample)

            mFFT = self.convstft(input_audio)  # (num_block * num_channel, num_bin * 2, num_frame)
    
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
                cmvn            = self.input_cmvn
                fbank_extractor = self.fbankNet if (self.multiple_args.spect_feats is not None and self.multiple_args.spect_feats.lower()) == 'fbank' else None
                multiple_enh_fft, emask = self.multipleEnh(mFFT, self.tdoas, cmvn = cmvn, fbank_extractor = fbank_extractor) # ( num_block, num_frame, 2, num_bin), ( num_block, num_frame, num_bin )
                
                multiple_enh_fft = multiple_enh_fft.permute([0, 2, 3, 1]).contiguous() # ( num_block, 2, num_bin, num_frame )

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

        return multiple_enh_wav

    def compute_visuals(self):
        # self.visual_names = ['mixture_spect', 'multiple_enh_spect', 'single_enh_spect', 'spe_spect']        
        with torch.no_grad():
            
            self.mmFFT = self.mmFFT.view(self.num_block, self.num_channel, self.num_bin * 2, -1)[:, :, :, :self.num_frame] # (num_block, num_channel, num_bin * 2, num_frame)
            
            mixture_spect       = self.mmFFT[:, 0, :self.num_bin, :] ** 2 + self.mmFFT[:, 0, self.num_bin:, :] ** 2 # (num_block, num_bin, num_frame)
            mixture_spect       = torch.clamp(mixture_spect, min=self.numerical_protection)
            mixture_spect       = mixture_spect.permute([0, 2, 1])                                                  # (num_block, num_frame , num_bin)
            mixture_spect       = torch.unsqueeze(mixture_spect, 1)                                                 # ( num_block, 1, num_frame , num_bin )
    
            if self.multiple_args.spect_type.lower() == 'fbank':
                self.mixture_spect = self.fbankNet(torch.sqrt(mixture_spect))                    # ( num_block, 1, num_frame , num_bin )
            else:
                self.mixture_spect = mixture_spect ** (0.5 * self.compressed_scale)              # ( num_block, 1, num_frame , num_bin )
    
    def forward(self):
        # mFFT: (num_block * num_frame, 2, num_bin, num_channel)
        if self.multipleEnh is not None:
            cmvn            = self.input_cmvn
            fbank_extractor = self.fbankNet if ( self.multiple_args.spect_feats is not None and self.multiple_args.spect_feats.lower() ) else None
            multiple_enh_fft, emask = self.multipleEnh(self.mFFT, self.tdoas, cmvn = cmvn, fbank_extractor = fbank_extractor) # ( num_block, num_frame, 2, num_bin), ( num_block, num_frame, num_bin )

            self.multiple_enh_fft   = multiple_enh_fft.permute([0, 2, 3, 1]).contiguous() # ( num_block, 2, num_bin, num_frame )
            self.emask              = emask.unsqueeze(1)                                  # ( num_block, 1, num_frame, num_bin )
        else:
            self.multiple_enh_fft  = None
            self.emask             = None
        
        if self.multiple_args.fft_consistency and self.multiple_enh_fft is not None:
            self.multiple_enh_fft = self.multiple_enh_fft.permute([0, 3, 2, 1])           # [num_block, num_frame, num_bin, 2]
            self.multiple_enh_fft = torch.ifft(self.multiple_enh_fft, 3)
            self.multiple_enh_fft = torch.fft(self.multiple_enh_fft, 3)
            self.multiple_enh_fft = self.multiple_enh_fft.permute([0, 3, 2, 1])           # ( num_block, 2, num_bin, num_frame )

        if self.multiple_enh_fft is not None:
            multiple_enh_spect = torch.unsqueeze(self.multiple_enh_fft[:,0,:,:] ** 2 + self.multiple_enh_fft[:,1,:,:] ** 2, 1)  # ( num_block, 1, num_bin, num_frame )
            multiple_enh_spect = torch.clamp(multiple_enh_spect, min=self.numerical_protection)
        else:
            multiple_enh_spect = None
        if self.sFFT is not None:
            spe_spect          = torch.unsqueeze(self.sFFT[:,0,:,:] ** 2 + self.sFFT[:,1,:,:] ** 2, 1) # ( num_block, 1, num_bin, num_frame )
            spe_spect          = torch.clamp(spe_spect, min=self.numerical_protection)
        else:
            spe_spect = None
        if self.nFFT is not None:   # ( num_block, 2, num_bin, num_frame )
            noi_spect          = torch.unsqueeze(self.nFFT[:,0,:,:] ** 2 + self.nFFT[:,1,:,:] ** 2, 1) # ( num_block, 1, num_bin, num_frame )
            noi_spect          = torch.clamp(noi_spect, min=self.numerical_protection)
        else:
            noi_spect = None
        
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
        if spe_spect is not None:
            spe_spect          = spe_spect.permute([0, 1, 3, 2])                          # ( num_block, 1, num_frame, num_bin )
        if noi_spect is not None:
            noi_spect          = noi_spect.permute([0, 1, 3, 2])                          # ( num_block, 1, num_frame, num_bin )
        
        if self.multiple_args.mask_weight > 0.0:
            self.mask = (spe_spect / (spe_spect + noi_spect)) ** 0.5                # ( num_block, 1, num_frame, num_bin )
            #self.mask = spe_spect ** 0.5 / (spe_spect ** 0.5 + noi_spect ** 0.5)   # ( num_block, 1, num_frame, num_bin )
            self.mask = torch.clamp(self.mask, min=1.0e-4)                          # ( num_block, 1, num_frame, num_bin )
        else:
            self.mask = None
        
        if multiple_enh_spect is not None:
            if self.multiple_args.spect_type.lower() == 'fbank':
                self.multiple_enh_spect = self.fbankNet(torch.sqrt(multiple_enh_spect))
            else:
                self.multiple_enh_spect = multiple_enh_spect ** (0.5 * self.compressed_scale)
        else:
            self.multiple_enh_spect = None
        
        if spe_spect is not None:
            if self.multiple_args.spect_type.lower() == 'fbank':
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
        
        # loss_names = ['FFT', 'Spect', 'mask']
        spect_weight = self.multiple_args.spect_weight
        fft_weight   = self.multiple_args.fft_weight
        mask_weight  = self.multiple_args.mask_weight

        if fft_weight > 0.0 and self.multiple_enh_fft is not None:
            #self.loss_FFT = 0.1 * self.criterionL1(self.highf_scale * self.multiple_enh_fft, self.highf_scale * self.sFFT) * self.num_bin * 2 + 0.9 * self.criterionL1(self.lowf_scale * self.multiple_enh_fft, self.lowf_scale * self.sFFT) * self.num_bin * 2

            self.loss_FFT = self.criterionL1(self.lowf_scale * self.multiple_enh_fft, self.lowf_scale * self.sFFT) * self.num_bin * 2
            if losses is None:
                losses = fft_weight * self.loss_FFT
            else:
                losses = losses + fft_weight * self.loss_FFT
        else:
            self.loss_FFT   = 0.0

        if spect_weight > 0.0 and self.multiple_enh_spect is not None:
            self.loss_Spect = self.criterionL1(self.multiple_enh_spect, self.spe_spect) * self.num_bin * 2
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
        if self.multipleEnh_optimizer is not None:
            self.multipleEnh_optimizer.zero_grad()
    
    def step(self):
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
