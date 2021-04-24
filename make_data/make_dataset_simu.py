'''
This is a script for generating UCA microphone data.
'''

from __future__ import division, print_function

import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

import sys
import scipy.io as sio
import numpy as np
import torch

from model.conv_stft import ConvSTFT, ConviSTFT
from model.beamformer import Beamformor
import pyroomacoustics as pra

def make_mixture(speech, noise, snr = 0):

	if speech is None or noise is None:
		return None
	
	spelen = speech.shape[0]
	
	exnoise = noise
	while exnoise.shape[0] < spelen:
		exnoise = np.concatenate([exnoise, noise], 0)
	noise = exnoise
	noilen = noise.shape[0]

	elen = noilen - spelen - 1
	if elen > 1:
		s = int(np.round(random.randint(0, elen - 1)))
	else:
		s = 0
	e = s + spelen

	noise = noise[s:e]
	
	try:
		spe_pow = np.sum(np.var(speech, axis=0))
		noi_pow = np.sum(np.var(noise, axis=0))
		if spe_pow <= 0.0 or noi_pow <= 0.0:
			return None
		
		noi_scale = math.sqrt(spe_pow / (noi_pow * (10 ** (snr / 10.0))))
	except:
		return None

	mixture = speech + noise * noi_scale
	return mixture


# {'snr': 30, 'sir': iSIR, 'n_src': interf_num + 1, 'n_tgt': 1, 'ref_mic': 0}
def callback_mix(premix, snr=0, sir=0, ref_mic=0, n_src=None, n_tgt=None):

    # first normalize all separate recording to have unit power at microphone one
    pre_alpha = np.random.rand(1, n_src-1) + 0.1
    pre_alpha /= np.sum(pre_alpha)
    for it in range(0, n_src-1):
        premix[n_tgt+it:n_tgt+it +1,:,:] *= pre_alpha[0][it]

    p_mic_ref  = np.std(premix[:,ref_mic,:], axis=1)
    p_var_ref  = np.var(premix[:,ref_mic,:], axis=1)

    targ_Power   = max(p_var_ref[0], 1e-50)
    itf_Power    = max(np.sum(p_var_ref) - targ_Power, 1e-50)

    # premix /= p_mic_ref[:,None,None]

    # now compute the power of interference signal needed to achieve desired SIR
    sigma_i = np.sqrt(10 ** (- sir / 10) *targ_Power/itf_Power)
    premix[n_tgt:n_src,:,:] *= sigma_i
    # compute noise variance
    sigma_n = np.sqrt(10 ** (- snr / 10))*p_mic_ref[0,None]

    # Mix down the recorded signals
    mix = np.sum(premix[:n_src,:], axis=0) + sigma_n * np.random.randn(*premix.shape[1:])
    # mix *= p_mic_ref[0,None]
    return mix

def build_utt_scp(data_dir, utt_scp='wav.scp'):
    # utt_id mix_path spe_path spe_scale db
    wav_scp_file = os.path.join(data_dir, utt_scp)
    utts_scp = []
    with open(wav_scp_file) as f:
        for line in f.readlines():
            line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
            splits = line.split()

            utt_key = splits[0]
            mix_path = splits[1]

            utts_scp.append((utt_key, mix_path))
    utts_size = len(utts_scp)
    print("load %d utts from %s" % (utts_size, data_dir))
    return utts_scp, utts_size

def build_text_scp(data_dir, utt_scp='text'):
    wav_scp_file = os.path.join(data_dir, utt_scp)
    utts_scp = []
    with open(wav_scp_file,"r",encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
            splits = line.split()

            utt_key = splits[0]
            utt_text = line.replace(utt_key,'')

            utts_scp.append((utt_key, utt_text))
    utts_size = len(utts_scp)
    print("load %d utts from %s" % (utts_size, data_dir))
    return utts_scp, utts_size

def find_utt_text(utts_scp, utts_size, utt_key):
    utt_text = ''
    ut       = 0
    for ut in range(0, utts_size):
       # print(str(ut))
        if utt_key ==  utts_scp[ut][0]:
            utt_text = utts_scp[ut][1]
            break
    return utt_text

def build_noise_scp(data_dir, utt_scp='wav.scp'):
    wav_scp_file = os.path.join(data_dir, utt_scp)
    utts_scp = []
    with open(wav_scp_file) as f:
        for line in f.readlines():
            line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
            splits = line.split()

            mix_path = splits[0]
            utts_scp.append(mix_path)
    utts_size = len(utts_scp)
    print("load %d utts from %s" % (utts_size, data_dir))
    return utts_scp, utts_size

def file_name(pathName):
    fName = []
    pathName = pathName.strip().replace('/', ' ')
    splits = pathName.split()
    tName = splits[-1]
    fName = tName[0:-4]
    return fName


# Some simulation parameters
Fs 			  = 16000
t0 			  = 1. / ( Fs * np.pi * 1e-2 )
max_order_sim = 2
sigma2_n      = 5e-8

#setting the simulation parameters#
min_delta_ang  = 45
min_absorption = 0.2
max_absorption = 0.7

min_iSNR       = 15
max_iSNR       = 25

interf_num     = 1

min_iSIR       = -5
max_iSIR       = 8

rm_min_x       = 3.5        # the min value of the room dimension in x
rm_min_y       = 3.5        # the min value of the room dimension in y
rm_min_ht      = 2          # the min value of the room dimension in z

min_pos        = 0.6

dist_min_ms    = 0.5        # the min distance between the microphone and each source
dist_max_ms    = 2.0        # the max distance between the microphone and each source

delay          = 0

rt60_min       = 0.2        # the min length of rt60(sec)
rt60_max       = 0.4        # the max length of rt60(sec)

min_Gain       = 0.05
max_Gain       = 0.5

num_mic     = 5            # number of microphones
radius_mics = 0.05		   # the radius of microphones arrays

simu_num       = 5000              	# the sentences number
rm_num         = 1000                	# the room number for this simulation
rm_per_sim     = int(simu_num / rm_num) #for per room we genereate <rm_per_sim> mixture

gpu_ids = ['2']
device  = torch.device("cuda:{}".format(gpu_ids[0]) if len(gpu_ids) > 0 and torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

## Load gsc targ beamformor
filter_path = '/data/HOME/snie/deep_enhance/train_enhance/egs/trans_paper/exp/'
gsc_targ_bf = os.path.join(filter_path, 'gsc_targ_bf.mat')
key = sio.whosmat(gsc_targ_bf)[0][0]
data = sio.loadmat(gsc_targ_bf)
if key in data:
	gsc_targ_bf = data[key]  # MUST BE [1, 2, num_bin, num_channel] egs. [20, 2, 257, 5]
	print(gsc_targ_bf.shape)
else:
	gsc_targ_bf = None
	exit("Load gsc_targ_bf Failed!")
num_beam, num_dim, num_bin, num_channel = gsc_targ_bf.shape

targ_bf = Beamformor(num_beam = num_beam, num_bin = num_bin, num_channel = num_channel, weight_init = gsc_targ_bf, fix=True)
targ_bf.to(device)

## Tne number of beam
Targ_Ang_Num   		= 5 * num_beam
Targ_Ang_Resolution = 360 / Targ_Ang_Num

## Define and initialize the convstft and convistft
win_len   = 512
win_inc   = 256 
fft_len   = 512
win_type  = 'hamming'
convstft  = ConvSTFT(win_len, win_inc, fft_len, win_type = win_type, feature_type='complex', fix = True)
convstft.to(device)
convistft = ConviSTFT(win_len, win_inc, fft_len, win_type = win_type, feature_type = 'complex', fix = True)
convistft.to(device)

## Main script Enter
speech_scp_path  = '/exdata/HOME/sliang/DCF_GSC_NNTraining/clean/'
noise_scp_path   = '/exdata/HOME/sliang/DCF_GSC_NNTraining/dfnoise/dfn_m5_d50/'
#sav_file_dir     = '/home/snie/works/deep_enhance/train_enhance/egs/trans_paper/testdata/testset4-3mic-5.0cm/wav/'
#wrt_dir          = '/home/snie/works/deep_enhance/train_enhance/egs/trans_paper/testdata/testset4-3mic-5.0cm/'
sav_file_dir     = '/exdata/HOME/sliang/DCF_GSC_NNTraining/testsets/TestD4/wav/'
wrt_dir          = '/exdata/HOME/sliang/DCF_GSC_NNTraining/testsets/TestD4/'
if not os.path.isdir(wrt_dir):
    try:
        os.makedirs(wrt_dir)
    except OSError:
        exit("ERROR: %s is not a dir" % (wrt_dir))
if not os.path.isdir(sav_file_dir):
    try:
        os.makedirs(sav_file_dir)
    except OSError:
        exit("ERROR: %s is not a dir" % (sav_file_dir))

noisy_scp_file       = open(wrt_dir + 'wav.scp', 'w')

sp_utts_scp, sp_utts_size = build_utt_scp(data_dir = speech_scp_path, utt_scp = 'wav.scp')
df_utts_scp, df_utts_size = build_utt_scp(data_dir = noise_scp_path, utt_scp = 'wav.scp')

start_spe_id   = sp_utts_size - 3000
end_spe_id	   = sp_utts_size - 1

start_dif_id   = 0
end_dif_id     = df_utts_size - 1

num_utt   	   = 0
for rm in range(0, rm_num):
	## Random a room
    rm_x        = rm_min_x  + random.randint(0, 6) / 2.0
    rm_y        = rm_min_y  + random.randint(0, 6) / 2.0
    rm_ht       = rm_min_ht + random.randint(0, 4) / 2.0
	
	## Random the position of microphone array
    mc_x        = random.randint( 0, 10 * rm_x )  / 10
    mc_y        = random.randint( 0, 10 * rm_y )  / 10
    mc_ht       = random.uniform( 0, 0.6 * rm_ht )
	
    mc_x        = max(mc_x, min_pos)
    mc_x        = min(mc_x, rm_x - min_pos)
    mc_y        = max(mc_y, min_pos)
    mc_y        = min(mc_y, rm_y - min_pos)
    mc_ht       = max(mc_ht, min_pos)
	
	## The position of microphones
    mic_xyz = [] 
    for m in range(num_mic):
	    mic_theta = np.pi * ( m * ( 360.0 / num_mic ) / 180.0 )
	    x         = mc_x + radius_mics * np.cos(mic_theta)
	    y         = mc_y + radius_mics * np.sin(mic_theta)
	    z         = mc_ht
	    mic_xyz.append([x, y, z])
    mic_xyz = np.array(mic_xyz) # (5, 3)
    mic_xyz = mic_xyz.T			# (3, 5)
	
    ## Create the room
    absorption = random.uniform(min_absorption, max_absorption)
    rt60_tgt   = rt60_min + random.randint(0, 10) / 10.0*(rt60_max-rt60_min) 
    room_dim   = [rm_x, rm_y, rm_ht]
    absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    room_mix   = pra.ShoeBox(room_dim, fs=16000, materials=pra.Material(absorption), max_order=max_order, sigma2_awgn = sigma2_n)
    room_ref   = pra.ShoeBox(room_dim, fs=16000, materials=pra.Material(absorption), max_order=max_order, sigma2_awgn = None)
    # d_absorption, d_max_order = pra.inverse_sabine(0.2, room_dim)
    #room_dir   = pra.ShoeBox(room_dim, absorption =0.99999,  fs =16000, max_order = max_order, sigma2_awgn = None)
	
    ## Add micphone array
    mic_array = pra.MicrophoneArray(mic_xyz, Fs)
    room_mix.add_microphone_array(mic_array)
    room_ref.add_microphone_array(mic_array)
    #room_dir.add_microphone_array(mic_array)
	
    print("room = [%.2f %.2f %.2f], micro = [%.2f %.2f %.2f]" % (rm_x, rm_y, rm_ht, mc_x, mc_y, mc_ht))

	## Add sources to room_mix, room_ref and room_dir
    while True:
        targ_ang    = random.randint(0, Targ_Ang_Num - 1) * Targ_Ang_Resolution
        #targ_ang    = 0
        targ_theta  = np.pi * targ_ang / 180.0

        targ_dist   = random.uniform(dist_min_ms, dist_max_ms)
        
        targ_x      = mc_x + np.cos(targ_theta) * targ_dist
        targ_y      = mc_y + np.sin(targ_theta) * targ_dist
        
        if targ_x < (rm_x-0.1) and targ_x > 0.1 and targ_y < (rm_y-0.1) and targ_y > 0.1:
        #if room_mix.is_inside([targ_x, targ_y, mc_ht], include_borders=True):
            break

    target_source = np.array([targ_x, targ_y, mc_ht])
    room_mix.add_source(target_source)
    room_ref.add_source(target_source)
    #room_dir.add_source(target_source)
	
    print("targ_ang = %d, targ_dist %.2f" % (targ_ang, targ_dist))

	## Add sources to room_mix
    interf_source = []
    for it in range(0, interf_num):
        while True:
            interf_ang = random.randint(0, 360)
            if np.abs(targ_ang - interf_ang) < min_delta_ang:
                continue
            interf_theta = np.pi * interf_ang / 180.0

            interf_dist   = random.uniform(dist_min_ms, dist_max_ms)
            
            interf_x      = mc_x + np.cos(interf_theta) * interf_dist
            interf_y      = mc_y + np.sin(interf_theta) * interf_dist
            
            if interf_x <(rm_x-0.1) and interf_x > 0.1 and interf_y < (rm_y-0.1) and interf_y > 0.1:
            # if room_mix.is_inside([interf_x, interf_y, mc_ht], include_borders=True):
                break
        
        print("interf_ang = %d, interf_dist %.2f" % (interf_ang, interf_dist))
       
        interf_source.append(np.array([interf_x, interf_y, mc_ht]))
        room_mix.add_source(interf_source[it])
	
    room_mix.compute_rir()
    room_ref.compute_rir()
    #room_dir.compute_rir()
	
    for sim in range(0, rm_per_sim):
        room_mix.sources.clear()
        #room_dir.sources.clear()
        room_ref.sources.clear()
        
        ## Read target speech audio
        while True:
            spe_id 	   = random.randint(start_spe_id, end_spe_id)
            utt_key    = sp_utts_scp[spe_id][0]
            spe_path   = sp_utts_scp[spe_id][1]
            spe_name   = file_name(pathName= spe_path)
            sample_rate, spe_wav = wavfile.read(spe_path)
            if len(spe_wav.shape) > 1:
                spe_wav		     = np.mean(spe_wav, 1)
            spe_wav				 = spe_wav.astype(np.float)
            if np.mean(np.abs(spe_wav)) > 0:
                break
        
        spe_length 	   = spe_wav.shape[0]
        spe_wav        = pra.normalize(spe_wav)
        spe_wav        = pra.highpass(spe_wav, Fs, 50)
        
        room_mix.add_source(target_source, signal = spe_wav, delay = delay)
        room_ref.add_source(target_source, signal = spe_wav, delay = delay)
        #room_dir.add_source(target_source, signal = spe_wav, delay = delay)
        
        ## Read interfere speech audio
        for it in range(0, interf_num):
            while True:
                while True:
                    inf_id 	   = random.randint(start_spe_id, end_spe_id)
                    if np.abs(spe_id - inf_id) > 500:
                        break
                inf_path   = sp_utts_scp[inf_id][1]
                sample_rate, inf_wav = wavfile.read(inf_path) # (nsample, nchannel)
                if len(inf_wav.shape) > 1:
                    inf_wav		     = np.mean(inf_wav, 1)
                inf_wav				 = inf_wav.astype(np.float)
                if np.mean(np.abs(inf_wav)) > 0:
                    break
            
            inf_length = inf_wav.shape[0]
            inf_wav = pra.normalize(inf_wav)
            inf_wav = pra.highpass(inf_wav, Fs, 50)

            while(inf_length < spe_length):
                inf_wav    = np.concatenate((inf_wav, inf_wav), axis = 0)
                inf_length = inf_wav.shape[0]
            inf_wav = inf_wav[:spe_length]
            
            room_mix.add_source(interf_source[it][:], signal = inf_wav, delay = delay)
        
        ## Make the far-field mixture audio
        iSIR  = random.randint(min_iSIR, max_iSIR)
        room_mix.simulate(callback_mix = callback_mix, callback_mix_kwargs = {'snr': 30, 'sir': iSIR, 'n_src': interf_num + 1, 'n_tgt': 1, 'ref_mic': 0})
        
        mix_wav 				= room_mix.mic_array.signals.T	# (nchannel, nsample)
        mix_length, num_channel = mix_wav.shape
        
        ## Read diffuse noise
        while True:
            dif_id 	   = random.randint(start_dif_id, end_dif_id)
            dif_path   = df_utts_scp[dif_id][1]
            
            sample_rate, dif_wav = wavfile.read(dif_path)
            dif_wav				 = dif_wav.astype(np.float)
            if np.mean(np.abs(dif_wav)) > 0:
                break
        dif_length, num_channel = dif_wav.shape
        
        ## Add diffuse noise into mix
        while( dif_length < mix_length ):
            dif_wav    = np.concatenate((dif_wav, dif_wav), axis = 0)
            dif_length = dif_wav.shape[0]
        dif_wav = dif_wav[0:mix_length, :]
        
        iSNR    = random.randint(min_iSNR, max_iSNR)
        mix_wav = add_dfNoise(mix_wav, dif_wav, snr = iSNR)
        
        ## Adapt gain of mixture audio by given gain
        gain     = random.uniform(min_Gain, max_Gain)
        scale	 = gain / np.max(np.abs(mix_wav))
        mix_wav  = mix_wav * scale
        
        mix_wav = mix_wav * 32767.0
        mix_wav = mix_wav.astype(np.int16)
        
        ## Simulate the clean far-field signal to make ref signal for compute metrics
        room_ref.simulate()
        ref_wav  = room_ref.mic_array.signals 		 # (num_channel, spe_length)
        ref_wav  = ref_wav * scale			  		 # (num_channel, spe_length)
        
        num_block = 1
        ref_wav   = ref_wav[np.newaxis, :, :]    	 			 # [ num_block, num_channel, spe_length ]
        ref_wav   = torch.FloatTensor(ref_wav)   	 		     # [ num_block, num_channel, spe_length ]
        ref_wav   = ref_wav.view(num_block * num_channel, 1, -1) # [ num_block * num_channel, 1, spe_length ]

        input_audio  = ref_wav.to(device)     		 # (num_block * num_channel, 1, spe_length)

        mFFT  = convstft(input_audio)                # (num_block * num_channel, num_bin * 2, num_frame)

        num_frame = mFFT.size(2)
        mFFT   = mFFT.view(num_block, num_channel, num_bin * 2, -1) #( num_block, num_channel, num_bin * 2, num_frame)
        mFFT_r = mFFT[:, :, :num_bin, :] 							#( num_block, num_channel, num_bin, num_frame)
        mFFT_i = mFFT[:, :, num_bin:, :] 							#( num_block, num_channel, num_bin, num_frame)

        mFFT_r = mFFT_r.permute([0, 3, 2, 1]).contiguous() 		    #( num_block, num_frame, num_bin, num_channel)
        mFFT_i = mFFT_i.permute([0, 3, 2, 1]).contiguous()          #( num_block, num_frame, num_bin, num_channel)

        mFFT_r = mFFT_r.view(num_block * num_frame, num_bin, num_channel) # ( num_block * num_frame, num_bin, num_channel)
        mFFT_i = mFFT_i.view(num_block * num_frame, num_bin, num_channel) # ( num_block * num_frame, num_bin, num_channel)

        mFFT = torch.cat([torch.unsqueeze(mFFT_r, 1), torch.unsqueeze(mFFT_i, 1)], dim = 1) # ( num_block * num_frame, 2, num_bin, num_channel )

        beam_id = targ_ang / ( 360.0 / num_beam ) + 0.5
        beam_id = int(beam_id) % num_beam

        rFFT = targ_bf(mFFT, beam_id) 						 # (num_block * num_frame, 2, num_bin, 1)
        rFFT = rFFT.view(num_block, num_frame, 2, num_bin)   # (num_block, num_frame, 2, num_bin)

        rFFT    = rFFT.permute([0, 2, 3, 1]).contiguous()    # ( num_block, 2, num_bin, num_frame )
        est_fft = torch.cat([rFFT[:,0], rFFT[:,1]], 1) 	     # ( num_block, num_bin * 2, num_frame )
        ref_wav = convistft(est_fft)                         # ( num_block, 1, num_sample)
        ref_wav = torch.squeeze(ref_wav, 1)                  # ( num_block, num_sample)
        ref_wav = ref_wav[0, :]								 # ( num_sample)
        ref_wav = ref_wav.data.cpu().numpy() 				 # ( num_sample)
        
        ref_wav = ref_wav * 32767.0
        ref_wav = ref_wav.astype(np.int16)
        
        ## Simulate the direct signal
        #room_dir.simulate()
        dir_wav = room_ref.mic_array.signals[0,:].T # (spe_length)
        dir_wav = dir_wav * scale
        dir_wav = dir_wav * 32767.0
        dir_wav = dir_wav.astype(np.int16)
        
        num_utt += 1
        
        print("[%d] / [%d]" % (num_utt, simu_num))

        nsample = min(mix_wav.shape[0], ref_wav.shape[0], dir_wav.shape[0])
        mix_wav = mix_wav[:nsample]
        ref_wav = ref_wav[:nsample]
        dir_wav = dir_wav[:nsample]

        ## Write the mixture audio
        utt_key       = utt_key + '_Id' + str(num_utt)
        wt_f_Name     = sav_file_dir + spe_name+'_id'+str(num_utt) +'_Doa'+ str(targ_ang) + '_SIR' + str(iSIR) + 'db_SNR' + str(iSNR) + 'db.wav'
        wavfile.write(wt_f_Name, Fs, mix_wav)
        
        ## Write the direct audio
        cln_f_Name    = sav_file_dir + spe_name+'_id'+str(num_utt) +'_Doa'+ str(targ_ang) + '_DS.wav'
        wavfile.write(cln_f_Name, Fs, dir_wav)
        
        ## Write the direct audio
        ref_f_Name    = sav_file_dir + spe_name+'_id'+str(num_utt) +'_Doa'+ str(targ_ang) + '_Ref.wav'
        wavfile.write(ref_f_Name, Fs, ref_wav)
        
        noisy_scp_file.write(utt_key+ ' '+ wt_f_Name+ ' '+ cln_f_Name + ' '+ ref_f_Name + ' '+ str(1)+' '+ str(iSIR)+' '+ str(iSNR)+' '+str(gain * 32767.0)+'\n' )

noisy_scp_file.close()
