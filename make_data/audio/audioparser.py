import os
import subprocess
import gzip
import librosa
import numpy as np
import torch
import random
from audio.audio_io import load_audio
from audio.audio_io import load_mat
import math
from audio import kaldi_io

from scipy import signal
import scipy.io.wavfile as wav

windows = {'hamming': signal.hamming, 'hann': signal.hann, 'blackman': signal.blackman,
           'bartlett': signal.bartlett}

audio_conf_example = dict(sample_rate=8000,
                  num_channel=1,
                  num_uChannel=1,
                  window_size=512,
                  window_shift=256,
                  window='hamming',
                  mat_key='wav')

def add_delta(utt, delta_order):
    num_frames = utt.shape[0]
    feat_dim = utt.shape[1]

    utt_delta = np.zeros(shape=[num_frames, feat_dim * (1 + delta_order)], dtype=np.float32)

    # firsr order part is just the uttarnce max_offset+1
    utt_delta[:, 0:feat_dim] = utt

    scales = [[1.0], [-0.2, -0.1, 0.0, 0.1, 0.2], [0.04, 0.04, 0.01, -0.04, -0.1, -0.04, 0.01, 0.04, 0.04]]

    delta_tmp = np.zeros(shape=[num_frames, feat_dim], dtype=np.float32)
    for i in range(1, delta_order + 1):
        max_offset = (len(scales[i]) - 1) / 2
        for j in range(-max_offset, 0):
            delta_tmp[-j:, :] = utt[0:(num_frames + j), :]
            for k in range(-j):
                delta_tmp[k, :] = utt[0, :]
            scale = scales[i][j + max_offset]
            if scale != 0.0:
                utt_delta[:, i * feat_dim:(i + 1) * feat_dim] += scale * delta_tmp

        scale = scales[i][max_offset]
        if scale != 0.0:
            utt_delta[:, i * feat_dim:(i + 1) * feat_dim] += scale * utt

        for j in range(1, max_offset + 1):
            delta_tmp[0:(num_frames - j), :] = utt[j:, :]
            for k in range(j):
                delta_tmp[-(k + 1), :] = utt[(num_frames - 1), :]
            scale = scales[i][j + max_offset]
            if scale != 0.0:
                utt_delta[:, i * feat_dim:(i + 1) * feat_dim] += scale * delta_tmp
    return utt_delta

def splice(utt, left_context_width, right_context_width):
    """
    splice the utterance
    Args:
        utt: numpy matrix containing the utterance features to be spliced
        context_width: how many frames to the left and right should
            be concatenated
    Returns:
        a numpy array containing the spliced features, if the features are
        too short to splice None will be returned
    """
    # return None if utterance is too short
    if utt.shape[0] < 1 + left_context_width + right_context_width:
        return None

    #  create spliced utterance holder
    utt_spliced = np.zeros(
        shape=[utt.shape[0], utt.shape[1] * (1 + left_context_width + right_context_width)],
        dtype=np.float32)

    #  middle part is just the utterance
    utt_spliced[:, left_context_width * utt.shape[1]:
                   (left_context_width + 1) * utt.shape[1]] = utt

    for i in range(left_context_width):
        #  add left context
        utt_spliced[i + 1:utt_spliced.shape[0],
        (left_context_width - i - 1) * utt.shape[1]:
        (left_context_width - i) * utt.shape[1]] = utt[0:utt.shape[0] - i - 1, :]

    for i in range(right_context_width):
        # add right context
        utt_spliced[0:utt_spliced.shape[0] - i - 1,
        (left_context_width + i + 1) * utt.shape[1]:
        (left_context_width + i + 2) * utt.shape[1]] = utt[i + 1:utt.shape[0], :]

    return utt_spliced

class AudioParser(object):
    '''def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError'''

    def WaveData(self, audio_path, sample_rate=16000, id_channel=[0]):
        if audio_path is None or not os.path.exists(audio_path):
            return None
        if '.mat' == os.path.splitext(audio_path)[1]:
            y = load_mat(audio_path)                 # [num_sample, num_channel]
        else:
            y = load_audio(audio_path, sample_rate)  # [num_sample, num_channel]
        if y is None:
            return None
        if len(y.shape) == 1:
            y = y[:, np.newaxis]                     # [num_sample, 1]
        return y[:, id_channel]                      # [num_sample, use_channel]

    def nSamples(self, audio_path, sample_rate=16000, id_channel=[0]):
        y = self.WaveData(audio_path, id_channel, sample_rate)
        if y is None:
            return 0
        return y.shape[0]

    def FileSize(self, audio_path):
        return os.path.getsize(audio_path)

    def WriteWave(self, audio_path, data, sample_rate=16000):
        if data is not None:
            wav.write(audio_path, sample_rate, data)

    def RMRData(self, rmr_path, mat_key='rir'):
        y = load_mat(rmr_path, mat_key)
        return y

    def MakeMixture(self, speech, noise, db):
        if speech is None or noise is None:
            # print("MakeMixture: speech is None or noise is None")
            return None, None, None
        if np.sum(np.square(noise)) < 1.0e-6:
            # print("MakeMixture: np.sum(np.square(noise)) < 1.0e-6")
            return None, None, None

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
            noi_pow = np.sum(np.square(noise))
            if noi_pow > 0:
                noi_scale = math.sqrt(np.sum(np.square(speech)) / (noi_pow * (10 ** (db / 10.0))))
            else:
                return None, None, None
        except:
            return None, None, None

        nnoise = noise * noi_scale
        mixture = speech + nnoise
        mixture = mixture.astype('float32')
        speech = speech.astype('float32')
        nnoise = nnoise.astype('float32')
        return mixture, speech, nnoise
        
    def MixVoice(self, speech, noise, mix_rate = 0.5):
        if speech is None or noise is None:
            print("MakeMixture: speech is None or noise is None")
            return None
        #if np.sum(np.square(noise)) < 1.0e-20:
        #    print("MakeMixture: np.sum(np.square(noise)) < 1.0e-6")
        #    return None

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

        nnoise = noise * mix_rate
        mixture = speech + nnoise
        mixture = mixture.astype('float32')
        return mixture

    def Make_Reverberation(self, speech, rmr, use_fast=False):

        if speech is None or rmr is None:
            # print("Make_Reverberation: speech is None or rmr is None")
            return None
        nsamples = len(speech)
        speech = signal.convolve(speech, rmr, mode='full')
        speech = speech[0:nsamples]
        return speech

    def Gain_Control(self, wave, Gain):

        if wave is None:
            # print("Gain_Control: wave is None")
            return None, None

        max_sample = np.max(np.fabs(wave))

        if max_sample <= 0:
            # print("Gain_Control: np.fabs(wave) is 0")
            return None, None

        wave = wave / max_sample

        wave = wave * Gain

        wave = wave.astype('float32')

        return wave, Gain / max_sample

    def Make_Noisy_Wave(self, speech, noise, spe_rmr, noi_rmr, SNR):
        if speech is None or noise is None:
            # print("Make_Noisy_Wave:speech is None or noise is None")
            return None

        if spe_rmr is not None:
            speech = self.Make_Reverberation(speech, spe_rmr)
        if noi_rmr is not None:
            noise = self.Make_Reverberation(noise, noi_rmr)

        noisy = self.MakeMixture(speech, noise, SNR)
        return noisy

class FFTSpectAndMaskParser(AudioParser):
    def __init__(self, audio_conf = audio_conf_example, gpu = False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        """
        super(FFTSpectAndMaskParser, self).__init__()
        self.num_uChannel = audio_conf['num_uChannel']
        self.num_channel = audio_conf['num_channel']
        self.mat_key = audio_conf['mat_key']
        self.window_shift = audio_conf['window_shift']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.gpu = gpu

    def FFT(self, audio_path, sample_rate=16000, id_channel=[0]):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: FFT coefficient of audio
        """
        y = self.WaveData(audio_path, sample_rate, id_channel)
        if y is None:
            return None
        
        n_fft = self.window_size
        win_length = n_fft
        hop_length = self.window_shift

        ffts = []

        for ch in range(len(id_channel)):
            D = librosa.stft(y[:, ch], n_fft=n_fft, hop_length=hop_length,
                             win_length=win_length, window=self.window)
            D = np.transpose(D)[:, 0:int(n_fft/2)]
            ffts.append(D)
        return ffts

    def WavScaleFFT(self, audio_path, sample_rate=16000, id_channel=[0], scale = 8000.0, max_norm = True):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: FFT coefficient of audio
        """
        y = self.WaveData(audio_path, sample_rate, id_channel)
        if y is None:
            return None, None
        if max_norm:
            scale = scale * 1.0 / np.amax(np.abs(y))
        y = scale * y 
        
        n_fft = self.window_size
        win_length = n_fft
        hop_length = self.window_shift

        ffts = []

        for ch in range(len(id_channel)):
            D = librosa.stft(y[:, ch], n_fft=n_fft, hop_length=hop_length,
                             win_length=win_length, window=self.window)
            D = np.transpose(D)[:, 0:int(n_fft/2)]
            ffts.append(D)
        return ffts, scale

    def FFTSpect(self, audio_path, sample_rate=16000, id_channel=[0]):
        """
                :param audio_path: Path where audio is stored from the manifest file
                :return: FFT coefficient of audio
                """
        y = self.WaveData(audio_path, sample_rate, id_channel)
        if y is None:
            return None

        n_fft = self.window_size
        win_length = n_fft
        hop_length = self.window_shift

        Spects = []

        for ch in range(len(id_channel)):
            D = librosa.stft(y[:, ch], n_fft=n_fft, hop_length=hop_length,
                             win_length=win_length, window=self.window)
            D = np.transpose(D)
            spect, phase = librosa.magphase(D)
            Spects.append(spect)
        return Spects

    def FFTLogSpect(self, audio_path, sample_rate=16000, id_channel=[0]):
        """
                :param audio_path: Path where audio is stored from the manifest file
                :return: FFT coefficient of audio
                """
        y = self.WaveData(audio_path, sample_rate, id_channel)
        if y is None:
            return None
        
        n_fft = self.window_size
        win_length = n_fft
        hop_length = self.window_shift

        LogSpects = []
        for ch in range(len(id_channel)):
            D = librosa.stft(y[:, ch], n_fft=n_fft, hop_length=hop_length,
                             win_length=win_length, window=self.window)
            D = np.transpose(D)
            spect = np.log(np.multiply(np.conjugate(D), D).real + 1.0e-7)
            LogSpects.append(spect)
        return LogSpects

    def FFTPowSpect(self, audio_path, sample_rate=16000, id_channel=[0]):
        """
                :param audio_path: Path where audio is stored from the manifest file
                :return: FFT coefficient of audio
                """
        y = self.WaveData(audio_path, sample_rate, id_channel)
        if y is None:
            return None

        n_fft = self.window_size
        win_length = n_fft
        hop_length = self.window_shift

        PowSpects = []

        for ch in range(len(id_channel)):
            D = librosa.stft(y[:, ch], n_fft=n_fft, hop_length=hop_length,
                             win_length=win_length, window=self.window)
            D = np.transpose(D)
            spect = np.multiply(np.conjugate(D), D).real
            PowSpects.append(spect)
        return PowSpects

    def FFTMask(self, mFFTSpect, sFFTSpect, beta = 1.0):
        """
        :param mfft: the fft coeffiences of mixture signals
        :param sfft: the fft coeffiences of clean speech signals
        :param beta: the exponent coeffiences used for computing mask
        :return: Audio in training/testing format
        """

        #mask = np.divide(np.square(sFFTSpect), np.square(mFFTSpect) + 1.0e-13)
        mask = np.divide(sFFTSpect, mFFTSpect)
        mask = np.power(mask, beta)
        return mask


def read_gzip(target_path):
    """
    read the file containing the state alignments
    Args:
        target_path: path to the alignment file
    Returns:
        A dictionary containing
            - Key: Utterance ID
            - Value: The state alignments as a space seperated string
    """
    target_dict = {}
    with gzip.open(target_path, 'rb') as fid:
        for line in fid:
            split_line = line.decode('utf-8').strip().split(' ')
            if len(split_line) < 2:
                continue
            target_dict[split_line[0]] = split_line[1:]
    return target_dict

class FeatLabelParser(object):
    def __init__(self, label_file = None):
        if label_file is not None:
            self.target_dict = read_gzip(label_file)
        else:
            self.target_dict = None
        super(FeatLabelParser, self).__init__()

    def parse_label(self, label_path, num_frame, given_label = 0):
        
        encoded_targets = self.load_label(label_path)
        if encoded_targets is None:
            encoded_targets = np.zeros(shape = (1, num_frame), dtype = np.int32) + np.int32(given_label)
        else:
            encoded_targets = self.resample_label(encoded_targets, num_frame)
        return encoded_targets

    def resample_label(self, label, trg_frames):
        if label is None:
            return None

        if len(label.shape) >= 2:
            num_utt, src_frames = label.shape[0], label.shape[1]
            new_label = np.zeros(shape=(num_utt, trg_frames), dtype = np.int32)
            for k in range(num_utt):
                for i in range(trg_frames):
                    new_label[k, i] = label[k, int(np.round(i * src_frames / trg_frames))]
        else:
            src_frames = label.shape[0]
            new_label = np.zeros(shape=(trg_frames), dtype = np.int16)
            for i in range(trg_frames):
                new_label[i] = label[int(np.round(i * src_frames / trg_frames))]
        
        return new_label
     
    def load_label(self, label_path):
        if label_path is None:
            return None
        try:
            if "ark:" in label_path:
                encoded_target = kaldi_io.read_mat(label_path) # (1, num_frame)
                encoded_target = np.int32(encoded_target) 
            else:
                if self.target_dict is not None and label_path is not None and label_path in self.target_dict:
                    targets = self.target_dict[label_path]
                    num_frame = len(targets)
                    encoded_target = [np.int32(targets[i]) for i in range(num_frame)]
                    encoded_target = np.array(encoded_target, dtype = np.int32)
                    encoded_target = encoded_target[np.newaxis, :]
                else:
                    encoded_target = None
        except:
            print('{} has error'.format(label_path))
            encoded_target = None
        return encoded_target
        
    def parse_feat(self, feat_path, delta_order = 0, cmvn = None, left_context_width = 0, right_context_width = 0, step = 0):
        if feat_path is None:
            return None
        feat = self.load_feat(feat_path, delta_order)
        if feat is None:
            return None

        if cmvn is not None:
            feat = (feat + cmvn[0, :]) * cmvn[1, :]
        
        feat_size = feat.shape[1]
        
        if step > 0:
            left_context_width = left_context_width * (step + 1)
            right_context_width = right_context_width * (step + 1)
        
        if left_context_width > 0 or right_context_width > 0:
            feat = splice(feat, left_context_width, right_context_width)
        
        if step > 0 and feat is not None:
            #idxs = list(range(0, left_context_width, step + 1)) + [left_context_width] + list(range(left_context_width + 1, left_context_width + 1 + right_context_width, step + 1))
            idxs = list(range(0, left_context_width, step + 1)) + [left_context_width] + list(range(left_context_width + 2, left_context_width + 1 + right_context_width, step + 1))
            # 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
            select_idx = []
            for idx in idxs:
                select_idx = select_idx + list(range(idx * feat_size, (idx + 1) * feat_size))
            feat = feat[:, select_idx]
            
        return feat
    
    def load_feat(self, feat_path, delta_order = 0):
        try:
            
            if "ark:" in feat_path:
                feat = kaldi_io.read_mat(feat_path)
            else:
                feat = np.load(feat_path)
            if feat is not None and delta_order > 0:
                feat = add_delta(feat, delta_order)
        except:
            print('{} has error'.format(feat_path))
            feat = None
        return feat
