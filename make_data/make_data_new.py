#!/usr/bin/python3
# coding:utf-8

from __future__ import print_function
import io
import sys
import os
import codecs
import math
import numpy as np
from multiprocessing import Pool
from py_utils import file_parse
from py_utils.audioparser import AudioParser
import random
import argparse
import librosa
from scipy.io import wavfile
import sys
import scipy.io as sio
import torch
import re

from model.conv_stft import ConvSTFT, ConviSTFT
from model.beamformer import Beamformor, NullBeamformor

import pyroomacoustics as pra

def load_data(args):
    spe_utt_ids = None
    scp_file = os.path.join(args.dataroot, 'wav.scp')
    if not os.path.exists(scp_file):
        exit("IOError: %s not existed" % (scp_file))
    else:
        with open(scp_file) as f:
            spe_utt_ids = f.readlines()
        spe_utt_ids = [x.strip().split() for x in spe_utt_ids]
        spe_utt_size = len(spe_utt_ids)
        print("num_speech = %d" % spe_utt_size)

    if not os.path.exists(args.interference_scp):
        exit("IOError: %s not existed" % (args.interference_scp))
    else:
        with open(args.interference_scp) as f:
            noise_utt_ids = f.readlines()
        noise_utt_ids = [x.strip() for x in noise_utt_ids]
        noise_utt_size = len(noise_utt_ids)
        print("num_noise = %d" % noise_utt_size)

    if not os.path.exists(args.diffuse_scp):
        diffuse_utt_ids = None
        print("num_diffuse = 0")
    else:
        with open(args.diffuse_scp) as f:
            diffuse_utt_ids = f.readlines()
        diffuse_utt_ids = [x.strip() for x in diffuse_utt_ids]
        diffuse_utt_size = len(diffuse_utt_ids)
        print("num_diffuse = %d" % diffuse_utt_size)

    utt2spk_file = os.path.join(args.dataroot, 'utt2spk')
    if not os.path.exists(utt2spk_file):
        utt2spk_dict = None
        print("num_utt2spk = 0")
    else:
        with open(utt2spk_file) as f:
            utt2spk_dict = f.readlines()
        utt2spk_dict = [x.strip().split() for x in utt2spk_dict]
        print("num_utt2spk = %d" % len(utt2spk_dict))
    
    utt2data_file = os.path.join(args.dataroot, 'utt2data')
    if not os.path.exists(utt2data_file):
        utt2data_dict = None
        print("num_utt2data = 0")
    else:
        with open(utt2data_file) as f:
            utt2data_dict = f.readlines()
        utt2data_dict = [x.strip().split() for x in utt2data_dict]
        print("num_utt2data = %d" % len(utt2data_dict))
    
    text_file = os.path.join(args.dataroot, 'text')
    if not os.path.exists(text_file):
        text_dict = None
        print("num_text = 0")
    else:
        text_dict = []
        with open(text_file) as f:
            for line in f.readlines():
                line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
                splits = line.split()
                if len(splits) < 2:
                    continue
                utt_id = splits[0]
                text_value = " ".join(splits[1:])
                text_dict.append((utt_id, text_value))
        print("num_text = %d" % len(text_dict))
    
    return spe_utt_ids, noise_utt_ids, diffuse_utt_ids, text_dict, utt2spk_dict, utt2data_dict

# {'snr': 30, 'sir': iSIR, 'n_src': num_interf + 1, 'n_tgt': 1, 'ref_mic': 0}
def callback_mix(premix, snr=0, sir=0, ref_mic=0, n_src=None, n_tgt=None):

    # first normalize all separate recording to have unit power at microphone one
    pre_alpha = np.random.rand(1, n_src-1) + 0.1
    pre_alpha /= np.sum(pre_alpha)
    for it in range(0, n_src-1):
        premix[n_tgt+it:n_tgt+it +1,:,:] *= pre_alpha[0][it] # (num_source, num_channel, num_sample)

    p_mic_ref  = np.std(premix[:,ref_mic,:], axis=1) # (num_source)
    p_var_ref  = np.var(premix[:,ref_mic,:], axis=1) # (num_source)

    targ_Power   = max(p_var_ref[0], 1e-50)
    itf_Power    = max(np.sum(p_var_ref) - targ_Power, 1e-50)

    # premix /= p_mic_ref[:,None,None]

    # now compute the power of interference signal needed to achieve desired SIR
    sigma_i = np.sqrt(10 ** (- sir / 10) *targ_Power/itf_Power)
    premix[n_tgt:n_src,:,:] *= sigma_i
    # compute noise variance
    sigma_n = np.sqrt( 10.0 ** (- snr / 10.0) ) * p_mic_ref[0,None]

    # Mix down the recorded signals
    mix = np.sum(premix[:n_src,:], axis=0) + sigma_n * np.random.randn(*premix.shape[1:])
    # mix *= p_mic_ref[0,None]
    return mix
    
def make_noisy(args, thread_id, num_make_utts):
    
    spe_utt_ids, noise_utt_ids, diffuse_utt_ids, text_dict, utt2spk_dict, utt2data_dict = load_data(args)

    audio_parser      = AudioParser()

    spe_utt_size     = len(spe_utt_ids) if spe_utt_ids is not None else 0
    noise_utt_size   = len(noise_utt_ids) if noise_utt_ids is not None else 0
    diffuse_utt_size = len(diffuse_utt_ids) if diffuse_utt_ids is not None else 0
    
    noisy_scp_list   = []
    noisy_utt2spk    = []
    noisy_text_dict  = []
    mix2info         = []
    num_utts         = 0

    all_angle           = 360.0
    Targ_Ang_Num        = args.num_targ_ang
    Targ_Ang_Resolution = all_angle / Targ_Ang_Num if Targ_Ang_Num > 0 else 0.0

    save_mix    = args.save_mix
    save_reverb = args.save_reverb
    save_clean  = args.save_clean
    while True:
        ## Random a room
        room_x   = random.uniform(args.min_room_length, args.max_room_length)
        room_y   = random.uniform(args.min_room_weidth, args.max_room_weidth)
        room_z   = random.uniform(args.min_room_height, args.max_room_height)
        room_dim = [room_x, room_y, room_z]

        ## Create the room
        T60                   = random.uniform(args.min_T60, args.max_T60)
        absorption, max_order = pra.inverse_sabine(T60, room_dim)
        if save_mix:
            room_mix   = pra.ShoeBox(room_dim, fs = args.sample_rate, materials=pra.Material(absorption), max_order=max_order, sigma2_awgn = None)
        else:
            room_mix   = None
        if save_reverb:
            room_ref   = pra.ShoeBox(room_dim, fs = args.sample_rate, materials=pra.Material(absorption), max_order=max_order, sigma2_awgn = None)
        else:
            room_mix   = None
        if save_clean:
            room_dir   = pra.ShoeBox(room_dim, fs = args.sample_rate, materials=pra.Material(0.99999), max_order=max_order, sigma2_awgn = None)
        else:
            room_dir = None
        
        ## Random the position of microphone array
        mic_x  = random.uniform(args.min_mic_x, room_x - args.min_mic_x)
        mic_y  = random.uniform(args.min_mic_y, room_y - args.min_mic_y)
        mic_z  = random.uniform(args.min_mic_z, max(min(room_z - args.min_mic_z, 2.0), args.min_mic_z + 0.5))

        ## Compute The position of microphones
        mic_xyz = []
        for m in range(args.num_mic):
            mic_pos   = args.mic_pos[m]
            x         = mic_x + mic_pos[0]
            y         = mic_y + mic_pos[1]
            z         = mic_z
            mic_xyz.append([x, y, z])
        mic_xyz = np.array(mic_xyz) # ( 6, 3 )
        mic_xyz = mic_xyz.T			# ( 3, 6 )

        ## Add micphone array
        mic_array = pra.MicrophoneArray(mic_xyz, args.sample_rate)
        if room_mix is not None:
            room_mix  = room_mix.add_microphone_array(mic_array)
        if room_ref is not None:
            room_ref  = room_ref.add_microphone_array(mic_array)
        if room_dir is not None:
            room_dir  = room_dir.add_microphone_array(mic_array)

        ##print("room = [%.2f %.2f %.2f], micro = [%.2f %.2f %.2f]" % (room_x, room_y, room_z, mic_x, mic_y, mic_z))
        
        ## Add target sources to room_mix and room_ref
        target_source = None
        while True:
            if args.num_targ_ang <= 0.0:
                targ_ang = random.randint( 0, int(all_angle) )
            else:
                targ_ang = int(random.randint(0, Targ_Ang_Num - 1) * Targ_Ang_Resolution)

            targ_theta  = np.pi * targ_ang / 180.0
            targ_dist   = random.uniform(args.min_targ_distance, args.max_targ_distance)
            
            targ_x      = mic_x + np.cos(targ_theta) * targ_dist
            targ_y      = mic_y + np.sin(targ_theta) * targ_dist
            targ_z      = mic_z

            target_source = [targ_x, targ_y, targ_z]

            if (targ_x < (room_x - 0.5) and targ_x > 0.5) and (targ_y < (room_y - 0.5) and targ_y > 0.5):
                break
            
        if target_source is None and not room_mix.is_inside(target_source):
            continue
        
        ##print("room = [%.2f %.2f %.2f], target_source = [%.2f %.2f %.2f]" % (room_x, room_y, room_z, target_source[0], target_source[1], target_source[2]))
        ##print("targ_ang = %d, targ_dist %.2f" % (targ_ang, targ_dist))
        targ_tdoa = targ_ang
        if args.is_linear_mic and targ_tdoa > 180:
            targ_tdoa = 360.0 - targ_tdoa
        
        ## Add interference sources to room_mix
        num_interf    = min(random.randint(1, args.max_num_interf), 1)
        interf_angs   = []
        interf_dists  = []
        interf_source = []
        
        while True:
            interf_ang  = random.randint(0, int(all_angle))
            interf_tdoa = interf_ang
            if args.is_linear_mic and interf_tdoa > 180:
                interf_tdoa = 360.0 - interf_tdoa
            if np.abs(targ_tdoa - interf_tdoa) < args.minAD:
                continue
            interf_theta = np.pi * interf_ang / 180.0
            interf_dist  = random.uniform(args.min_interf_distance, args.max_interf_distance)

            interf_x      = mic_x + np.cos(interf_theta) * interf_dist
            interf_y      = mic_y + np.sin(interf_theta) * interf_dist
            interf_z      = mic_z

            ainterf_source = [interf_x, interf_y, interf_z]
            if (interf_x < (room_x - 0.5) and interf_x > 0.5) and (interf_y < (room_y - 0.5) and interf_y > 0.5):
                interf_angs.append(interf_ang)
                interf_dists.append(interf_dist)
                interf_source.append(ainterf_source)
            
            if len(interf_source) >= num_interf:
                break
                
        ##print("interf_ang = %d, interf_dist %.2f, num_interf = %d" % (interf_ang, interf_dist, len(interf_source)))

        for sim in range(args.nutt_per_room):
            if room_mix is not None:
                room_mix.sources = []
            if room_ref is not None:
                room_ref.sources = []
            if room_dir is not None:
                room_dir.sources = []
            
            ## Add Speech to microphone array
            while True:
                spe_idx = random.randint(0, spe_utt_size - 1)
                spe_key, spe_path = spe_utt_ids[spe_idx]

                spe_wav = audio_parser.WaveData(spe_path, sample_rate = args.sample_rate)
                if spe_wav is None or spe_wav.shape[0] < args.sample_rate:
                    continue
                spe_wav = np.squeeze(spe_wav)
                if np.mean(np.abs(spe_wav)) > 0:
                    break
            
            spe_length 	   = spe_wav.shape[0]
            spe_wav        = pra.normalize(spe_wav)
            spe_wav        = pra.highpass(spe_wav, args.sample_rate, 50)
            
            if room_mix is not None and room_mix.is_inside(target_source):
                room_mix = room_mix.add_source(target_source, signal = spe_wav, delay = 0)
            else:
                print("target_source not in room_mix")
                continue
            if room_ref is not None and room_ref.is_inside(target_source):
                room_ref = room_ref.add_source(target_source, signal = spe_wav, delay = 0)
            else:
                print("target_source not in room_ref")
            if room_dir is not None and room_dir.is_inside(target_source):
                room_dir = room_dir.add_source(target_source, signal = spe_wav, delay = 0)
            else:
                print("target_source not in room_dir")
                        
            if room_mix is not None and len(room_mix.sources) < 1:
                print("target_source not in room_mix")
                break
            if room_ref is not None and len(room_ref.sources) < 1:
                print("target_source not in room_ref")
                break
            if room_dir is not None and len(room_dir.sources) < 1:
                print("target_source not in room_dir")
                break
            
            ## Add Interference to microphone array
            for it in range(0, num_interf):
                while True:
                    inf_idx = random.randint(0, noise_utt_size - 1)
                    inf_path = noise_utt_ids[inf_idx]

                    inf_wav = audio_parser.WaveData(inf_path, sample_rate = args.sample_rate)
                    if inf_wav is None or inf_wav.shape[0] < args.sample_rate:
                        continue
                    inf_wav = np.squeeze(inf_wav)
                    if np.mean(np.abs(inf_wav)) > 0:
                        break
                
                inf_length = inf_wav.shape[0]
                inf_wav = pra.normalize(inf_wav)
                inf_wav = pra.highpass(inf_wav, args.sample_rate, 50)

                while(inf_length < spe_length):
                    inf_wav    = np.concatenate((inf_wav, inf_wav), axis = 0)
                    inf_length = inf_wav.shape[0]
                inf_wav = inf_wav[:spe_length]
                
                if room_mix is not None and room_mix.is_inside(interf_source[it]):
                    room_mix = room_mix.add_source(interf_source[it], signal = inf_wav, delay = 0)
                else:
                    print("interf_source not in room_mix")
                    continue

            if room_mix is not None and len(room_mix.sources) < 1:
                break

            ## Make the far-field mixture audio
            iSIR  = random.uniform(args.lowSIR, args.upSIR)
            room_mix.simulate(callback_mix = callback_mix, callback_mix_kwargs = {'snr': 30, 'sir': iSIR, 'n_src': num_interf + 1, 'n_tgt': 1, 'ref_mic': 0})
            
            mix_wav 				= room_mix.mic_array.signals.T	# (nchannel, nsample)
            mix_length, num_channel = mix_wav.shape
            
            ## Read diffuse noise
            if diffuse_utt_ids is not None:
                while True:
                    diff_idx = random.randint(0, diffuse_utt_size - 1)
                    diff_path = diffuse_utt_ids[diff_idx]

                    diff_wav = audio_parser.WaveData(diff_path, sample_rate = args.sample_rate, id_channel = list(range(0, num_channel)))
                    if diff_wav is None or diff_wav.shape[0] < args.sample_rate:
                        continue
                    if np.mean(np.abs(diff_wav)) > 0:
                        break
                
                dif_length, num_channel = diff_wav.shape
                '''
                for i in range(int(num_channel / 2)):
                    ch_wav = diff_wav[:, i]
                    diff_wav[:, i] = diff_wav[:, num_channel - i -1]
                    diff_wav[:, num_channel - i -1] = ch_wav
                '''
                
                ## Add diffuse noise into mix
                while( dif_length < mix_length ):
                    diff_wav    = np.concatenate((diff_wav, diff_wav), axis = 0)
                    dif_length = diff_wav.shape[0]
                diff_wav = diff_wav[0:mix_length, :]
                
                iSNR    = random.uniform(args.lowSNR, args.upSNR)
                mix_wav = audio_parser.MixWave(mix_wav, diff_wav, snr = iSNR)

            ## Adapt gain of mixture audio by given gain
            gain     = random.uniform(args.lowGain, args.upGain)
            scale	 = gain / np.max(np.abs(mix_wav))
            mix_wav  = mix_wav * scale
            mix_wav  = mix_wav * 32767.0
            mix_wav  = mix_wav.astype(np.int16)

            if room_dir is not None:
                ## Simulate directional signals
                room_dir.simulate()
                dir_wav = room_dir.mic_array.signals[0,:].T # (spe_length)
                dir_wav = dir_wav * scale
                dir_wav = dir_wav * 32767.0
                dir_wav = dir_wav.astype(np.int16)
            else:
                dir_wav = None

            if room_ref is not None:
                ## Simulate the clean far-field signal to make ref signal for compute metrics
                room_ref.simulate()
                ref_wav = room_ref.mic_array.signals 		 # (num_channel, spe_length)
                ref_wav = ref_wav * scale			  		 # (num_channel, spe_length)
            else:
                ref_wav = None
            
            if ref_wav is not None:
                if args.targ_bf is not None:
                    num_block = 1
                    ref_wav   = ref_wav[np.newaxis, :, :]    	 			 # [ num_block, num_channel, spe_length ]
                    ref_wav   = torch.FloatTensor(ref_wav)   	 		     # [ num_block, num_channel, spe_length ]
                    ref_wav   = ref_wav.view(num_block * num_channel, 1, -1) # [ num_block * num_channel, 1, spe_length ]

                    input_audio  = ref_wav.to(args.device)     		 # (num_block * num_channel, 1, spe_length)

                    mFFT  = args.convstft(input_audio)                # (num_block * num_channel, num_bin * 2, num_frame)

                    num_frame = mFFT.size(2)
                    mFFT   = mFFT.view(num_block, num_channel, num_bin * 2, -1) #( num_block, num_channel, num_bin * 2, num_frame)
                    mFFT_r = mFFT[:, :, :num_bin, :] 							#( num_block, num_channel, num_bin, num_frame)
                    mFFT_i = mFFT[:, :, num_bin:, :] 							#( num_block, num_channel, num_bin, num_frame)

                    mFFT_r = mFFT_r.permute([0, 3, 2, 1]).contiguous() 		    #( num_block, num_frame, num_bin, num_channel)
                    mFFT_i = mFFT_i.permute([0, 3, 2, 1]).contiguous()          #( num_block, num_frame, num_bin, num_channel)

                    mFFT_r = mFFT_r.view(num_block * num_frame, num_bin, num_channel) # ( num_block * num_frame, num_bin, num_channel)
                    mFFT_i = mFFT_i.view(num_block * num_frame, num_bin, num_channel) # ( num_block * num_frame, num_bin, num_channel)

                    mFFT = torch.cat([torch.unsqueeze(mFFT_r, 1), torch.unsqueeze(mFFT_i, 1)], dim = 1) # ( num_block * num_frame, 2, num_bin, num_channel )

                    # Compute the BF bf_direction_resolution
                    targ_tdoa = targ_ang
                    if num_channel == 2 or args.is_linear_mic:
                        if targ_tdoa > 180:
                            targ_tdoa = 360.0 - targ_tdoa
                    bf_beam = targ_tdoa / args.bf_direction_resolution + 0.5
                    bf_beam = int(bf_beam) % args.num_beam
                    print("tdoa = %d, beam = %d" % (targ_ang, bf_beam))

                    rFFT = args.targ_bf(mFFT, bf_beam) 				            # (num_block * num_frame, 2, num_bin, 1)
                    rFFT = rFFT[:, :, :, 0].view([num_block, -1, 2, num_bin])   # (num_block, num_frame, 2, num_bin)

                    rFFT    = rFFT.permute([0, 2, 3, 1]).contiguous()    # ( num_block, 2, num_bin, num_frame )
                    est_fft = torch.cat([rFFT[:,0], rFFT[:,1]], 1) 	     # ( num_block, num_bin * 2, num_frame )
                    ref_wav = args.convistft(est_fft)                    # ( num_block, 1, num_sample)
                    ref_wav = torch.squeeze(ref_wav, 1)                  # ( num_block, num_sample)
                    ref_wav = ref_wav[0, :]								 # ( num_sample)
                    ref_wav = ref_wav.data.cpu().numpy() 				 # ( num_sample)
                else:
                    ref_wav = ref_wav[0, :]								 # ( num_sample)
                
                ref_wav = ref_wav * 32767.0
                ref_wav = ref_wav.astype(np.int16)
            else:
                ref_wav = None
            
            ## Align mix_wav, ref_wav and dir_wav
            nsample = min(mix_wav.shape[0], ref_wav.shape[0], dir_wav.shape[0])
            mix_wav = mix_wav[:nsample]
            if ref_wav is not None:
                ref_wav = ref_wav[:nsample]
            if dir_wav is not None:
                dir_wav = dir_wav[:nsample]

            num_utts += 1

            _, spe_name, _ = file_parse.getFileInfo(spe_path)

            out_path = os.path.join(args.out_path, 'wav')
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            
            if utt2data_dict is not None:
                data_key, data_id = utt2data_dict[spe_idx]
                out_path = os.path.join(out_path, data_id)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
            else:
                data_id = 'data01'

            if utt2spk_dict is not None:
                spk_key, spk_id = utt2spk_dict[spe_idx]
                out_path = os.path.join(out_path, spk_id)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
            else:
                spk_id = 'spk01'
                out_path = os.path.join(out_path, 'wav')
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
            
            spe_key = spe_key.replace('_', '').replace('-', '').replace('.', '')
            spk_id  = spk_id.replace('_', '').replace('-', '').replace('.', '')
            #utt_id = spk_id + "_" + spe_key + "%02d%07d" % (thread_id, num_utts)
            utt_id = spk_id + "_" + "%02d%07d" % (thread_id, num_utts)
            
            if mix_wav is not None:
                ## Write the mixture audio
                filename = "%s_id%02d%07d_Doa%d_SIR%.1f_SNR%.1f" % (spe_key, thread_id, num_utts, targ_ang, iSIR, iSNR)
                mix_path = os.path.join(out_path, '%s.wav' % (filename) )
                audio_parser.WriteWave(mix_path, mix_wav, args.sample_rate)
            else:
                mix_path = None

            if dir_wav is not None:
                filename = "%s_id%02d%07d_Doa%d_DS" % (spe_key, thread_id, num_utts, targ_ang)
                ds_path = os.path.join(out_path, '%s.wav' % (filename) )
                audio_parser.WriteWave(ds_path, dir_wav, args.sample_rate)
            else:
                ds_path = None
            
            if ref_wav is not None:
                filename = "%s_id%02d%07d_Doa%d_Ref" % (spe_key, thread_id, num_utts, targ_ang)
                ref_path = os.path.join(out_path, '%s.wav' % (filename) )
                audio_parser.WriteWave(ref_path, ref_wav, args.sample_rate)
            else:
                ref_path = None

            if text_dict is not None:
                text_key, text_value = text_dict[spe_idx]
            else:
                text_value = ' '
            
            noisy_scp_list.append((utt_id, mix_path, ds_path, ref_path, targ_ang, targ_dist, iSIR, iSNR, scale))
            noisy_utt2spk.append(spk_id)
            noisy_text_dict.append(text_value)

            info = (utt_id, spe_key, mix_path, ds_path, ref_path, targ_ang, targ_dist, interf_angs, interf_dists, iSIR, iSNR, scale)

            mix2info.append(info)
            
            print("%d / %d: %s" % (num_utts, num_make_utts, mix_path))

            if num_utts >= num_make_utts:
                return noisy_scp_list, noisy_utt2spk, noisy_text_dict, mix2info

if __name__ == '__main__':
    '''
    Function describtion:
        Make noisy audio with diffuse noise, interference and reverberations, and generate wav.scp text utt2spk spk2utt mixinfo et al.
    '''
    parser = argparse.ArgumentParser(description = 'Make Noisy Audio')
    parser.add_argument('--dataroot', metavar='DIR', help='path to the clean wav (wav.scp text utt2spk spk2utt)', default='/home/snie/works/data/speech/train/')
    parser.add_argument('--out_path', metavar='DIR', help='the output path to save the noisy data', default='/home/snie/works/data/noisy_data/train/')
    
    parser.add_argument('--num_workers', type=int, default=1, help='the number of the thread to make noisy data')
    parser.add_argument('--num_utterances', type=int, default=1000000, help='the number of utterances that you want to make')

    parser.add_argument('--diffuse_scp', default='/data/HOME/snie/data/noise/diffuse_noise/train/diffuse_noise.scp', type=str, help='File to diffuse noise')
    parser.add_argument('--interference_scp', default='/data/HOME/snie/data/noise/interference/train/interference.scp', type=str, help='File to interference noise')

    parser.add_argument('--targ_bf', default=None, type = str, help='targ_bf.mat File to target beamformer')
    parser.add_argument('--num_null', type=int, default=0, help='num of null direction for targ_bf')
    parser.add_argument('--num_targ_ang', type=int, default=0, help='the number of direction for target source')
    
    parser.add_argument('--nutt_per_room', type=int, default=10, help='the number of utterances that you want to make in one simulated room')
	
    parser.add_argument('--sample_rate', default = 16000, type=int, help='sample rate of audio, Default: 16000')

    parser.add_argument('--num_mic', type=int, default=1, help='the number of microphones')
    parser.add_argument('--mic_pos', default = '(0.020000, 0.000000) (-0.020000, 0.000000)', type=str, help='position of microphones')
    parser.add_argument('--is_linear_mic', type=bool, default=True, help='Is Microphone array')
    
    parser.add_argument('--min_targ_distance', type=float, default=0.5, help='min distance between target speech and mic')
    parser.add_argument('--max_targ_distance', type=float, default=5.5, help='max distance between target speech and mic')

    parser.add_argument('--min_interf_distance', type=float, default=1.0, help='1.0 min distance of room')
    parser.add_argument('--max_interf_distance', type=float, default=7.0, help='7.0 min weidth of room')

    parser.add_argument('--min_mic_x', type=float, default=0.5, help='min x of mic position')
    parser.add_argument('--min_mic_y', type=float, default=0.5, help='min y of mic position')
    parser.add_argument('--min_mic_z', type=float, default=1.0, help='min z of mic position')

    parser.add_argument('--min_room_length', type=float, default=3.0, help='3.0 min length of room')
    parser.add_argument('--max_room_length', type=float, default=8.0, help='9.0 max length of room')
    parser.add_argument('--min_room_weidth', type=float, default=2.5, help='2.5 min weidth of room')
    parser.add_argument('--max_room_weidth', type=float, default=5.5, help='6.5 max weidth of room')
    parser.add_argument('--min_room_height', type=float, default=2.5, help='2.5 min height of room')
    parser.add_argument('--max_room_height', type=float, default=4.5, help='4.0 max height of room')

    parser.add_argument('--min_T60', type=float, default=0.15, help='min reverberation time')
    parser.add_argument('--max_T60', type=float, default=0.50, help='max reverberation time')
    parser.add_argument('--sigma2_awgn', type=float, default=1.0e-5, help='the noise from Microphone hardware')
	
    parser.add_argument('--max_num_interf', type=int, default=3, help='the number of interference sources')
    parser.add_argument('--minAD', type=float, default=0.0, help='min angle difference between targe source and interference source')
    parser.add_argument('--lowSIR', type=float, default=-5.0, help='-5 lowSIR for interference')
    parser.add_argument('--upSIR', type=float, default=15.0, help='15 upSNR for interference')
    parser.add_argument('--lowSNR', type=float, default=5.0, help='0.0 lowSNR for diffuse noise')
    parser.add_argument('--upSNR', type=float, default=20.0, help='20.0 upSNR for diffuse noise')

    parser.add_argument('--lowGain', type=float, default=0.01, help='lowGain = 0.01 * 32767')
    parser.add_argument('--upGain', type=float, default=0.7, help='upGain = 0.7 * 32767')

    parser.add_argument('--save_mix', type=bool, default=True, help='whether to save the mixture wav')
    parser.add_argument('--save_reverb', type=bool, default=True, help='whether to save the noise-free reverb wav')
    parser.add_argument('--save_clean', type=bool, default=True, help='whether to save the noise-free reverb-free clean wav')

    args = parser.parse_args()
    
    # Prepare path and file
    if not os.path.exists(args.out_path):
        try:
            os.makedirs(args.out_path)
        except IOError:
            exit("IOError: Cann't makedir %s" % (args.out_path))
    
    gpu_ids = ['0']
    args.device  = torch.device("cuda:{}".format(gpu_ids[0]) if len(gpu_ids) > 0 and torch.cuda.is_available() else "cpu")
    if len(gpu_ids) > 0:
        torch.backends.cudnn.benchmark = True

    ## parse the mic_pos (0.0, 0.0) (0.0, 0.0) (0.0, 0.0) (0.0, 0.0) (0.0, 0.0)
    mic_pos_str = re.findall(r'[(](.*?)[)]', args.mic_pos)
    mic_pos = []
    for pos_str in mic_pos_str:
        pos_str = pos_str.split(',')
        mic_pos.append((float(pos_str[0]), float(pos_str[1]))) 
    args.mic_pos = mic_pos
    print(args.is_linear_mic)
    print(mic_pos)
    args.num_mic = len(mic_pos)

    ## Load gsc targ beamformor
    if args.targ_bf is not None and os.path.exists(args.targ_bf):
        ## Define and initialize the targ beamformor
        print("targ_bf = %s" % (args.targ_bf))
        targ_bf = args.targ_bf
        key = sio.whosmat(targ_bf)[0][0]
        data = sio.loadmat(targ_bf)
        if key in data:
            targ_bf = data[key]  # MUST BE [1, 2, num_bin, num_channel] egs. [20, 2, 257, 5]
            print(targ_bf.shape)
        else:
            targ_bf = None
            exit("Load targ_bf Failed!")
        num_beam, num_dim, num_bin, num_channel = targ_bf.shape
        assert num_channel >= args.num_mic , "illegal targ_bf, the required targ_bf should be for %d num_channel, but got %d" % (args.num_mic, num_channel)

        print("num_beam = %d, num_channel = %d" % (num_beam, num_channel))
        if args.num_null > 0:
            num_beam = int(num_beam / args.num_null)
        if args.num_null > 0:
            args.targ_bf = NullBeamformor(num_beam = num_beam, num_null = args.num_null, num_bin = num_bin, num_channel = num_channel, weight_init = targ_bf, fix = True)
        else:
            args.targ_bf = Beamformor(num_beam = num_beam, num_bin = num_bin, num_channel = num_channel, weight_init = targ_bf, fix = True)
        args.targ_bf.to(args.device)
        if num_channel == 2 or args.is_linear_mic:
            args.bf_direction_resolution = 180.0 / (num_beam - 1)
        else:
            args.bf_direction_resolution = 360.0 / num_beam
        print("bf_direction_resolution = %f" % (args.bf_direction_resolution))
        args.num_beam = num_beam

        ## Define and initialize the convstft and convistft
        win_len   = 512
        win_inc   = 256 
        fft_len   = 512
        win_type  = 'hamming'
        args.convstft  = ConvSTFT(win_len, win_inc, fft_len, win_type = win_type, feature_type='complex', fix = True)
        args.convstft.to(args.device)
        args.convistft = ConviSTFT(win_len, win_inc, fft_len, win_type = win_type, feature_type = 'complex', fix = True)
        args.convistft.to(args.device)
    else:
        args.num_beam  = 0
        args.targ_bf   = None
        args.convstft  = None
        args.convistft = None

    print("start make_noisy")
    noisy_scp_list, noisy_utt2spk, noisy_text_dict, mix2info = make_noisy(args, 1, args.num_utterances)

    out_scp_file      = os.path.join(args.out_path, 'wav.scp')
    out_mix_file      = os.path.join(args.out_path, 'mix.scp')
    out_text_file     = os.path.join(args.out_path, 'text')
    out_utt2info_file = os.path.join(args.out_path, 'utt2info')
    out_utt2spk_file  = os.path.join(args.out_path, 'utt2spk')
    out_spk2utt_file  = os.path.join(args.out_path, 'spk2utt')

    scp_writer = codecs.open(out_scp_file, 'w', 'utf-8')
    mix_writer = codecs.open(out_mix_file, 'w', 'utf-8')
    text_writer = codecs.open(out_text_file, 'w', 'utf-8')
    utt2info_writer = codecs.open(out_utt2info_file, 'w', 'utf-8')
    utt2spk_writer = codecs.open(out_utt2spk_file, 'w', 'utf-8')
    spk2utt_writer = codecs.open(out_spk2utt_file, 'w', 'utf-8')

    data2utt_dict = {}
    spk2utt_dict = {}

    for n in range(len(noisy_scp_list)):
        
        utt_id, mix_path, ds_path, ref_path, targ_ang, targ_dist, iSIR, iSNR, scale = noisy_scp_list[n]
        text_value = noisy_text_dict[n]
        spk_id = noisy_utt2spk[n]
        
        scp_writer.write("%s %s\n" % (utt_id, mix_path))
        mix_writer.write("%s %s %s %s %d %.1f %.1f %.1f, %.1f\n" % (utt_id, mix_path, ds_path, ref_path, targ_ang, targ_dist, iSIR, iSNR, scale))
        text_writer.write("%s %s\n" % (utt_id, text_value))
        utt2spk_writer.write("%s %s\n" % (utt_id, spk_id))

        utt_id, spe_key, mix_path, ds_path, ref_path, targ_ang, targ_dist, interf_angs, interf_dists, iSIR, iSNR, scale = mix2info[n]
        interf_info = ""
        for m in range(len(interf_angs)):
            interf_info = interf_info + " %d %.1f" % (interf_angs[m], interf_dists[m])
        utt2info_writer.write("%s, %s, %s, %s, %s, %d, %.1f, %s, %.1f, %.1f, %.1f\n" % (utt_id, spe_key, mix_path, ds_path, ref_path, targ_ang, targ_dist, interf_info, iSIR, iSNR, scale))

        spk2utt = utt_id
        if spk_id in spk2utt_dict.keys():
            spk2utt = spk2utt_dict[spk_id] + " " + spk2utt
        spk2utt_dict[spk_id] = spk2utt
            
    for spk_id, utt_ids in spk2utt_dict.items():
        spk2utt_writer.write("%s %s\n" % (spk_id, utt_ids))
    
    scp_writer.close()
    mix_writer.close()
    text_writer.close()
    utt2info_writer.close()
    utt2spk_writer.close()
    spk2utt_writer.close()
    exit("Succed to Finish!")
