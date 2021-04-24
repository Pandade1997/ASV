import os
import gzip
import numpy as np
import struct
import random
import math
import scipy.signal
import librosa
import torch
import scipy.io as sio
#import torchaudio
import scipy.io.wavfile as wav
from scipy import signal

from py_utils.wav_io import readwav, writewav
from py_utils.utils import convfft

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}
audio_conf_example = dict(sample_rate=16000,
                          num_channel=5,
                          num_uChannel=1,
                          window_size=512,
                          window_shift=256,
                          window='hamming',
                          mat_key='wav')

def FileSize(audio_path):
    return os.path.getsize(audio_path)

'''
def load_audio(path):
    try:
        if os.path.exists(path) and FileSize(path) > 15000:
            rate, sound = wav.read(path)
            if len(sound.shape) > 1:
                if sound.shape[1] == 1:
                    sound = sound.squeeze()
                else:
                    sound = sound.mean(axis=1)
            if rate != 16000:
                sound = signal.resample(sound, len(sound) * 16000 / rate)
        else:
            return None

        return sound
    except:
        print('load_audio {} failed'.format(path))
        return None
'''

def load_audio(path):
    try:
        if os.path.exists(path) and FileSize(path) > 4000:
            #rate, sound = wav.read(path)
            sound, rate = librosa.load(path, sr=16000)
            
            if len(sound.shape) > 1:
                if sound.shape[1] == 1:
                    sound = sound.squeeze()
                else:
                    sound = sound.mean(axis=1)
            if rate != 16000:
                #sound = signal.resample(sound, len(sound) * 16000 / rate)
                sound = librosa.resample(sound, rate, 16000)
        else:
            return None

        ##sound = sound / 65536.
        return sound
    except:
        print('load_audio {} failed'.format(path))
        return None

def load_mat(filename, key='feat'):
    if os.path.exists(filename):
        key = sio.whosmat(filename)[0][0]
        ##print('key = %s' % key)
        data = sio.loadmat(filename)
        if key in data:
            mat = data[key]
        else:
            return None
        '''
        if mat.shape[0] > 1:
            mat = mat[:, 0];
            mat = mat.reshape(mat.shape[0])
        elif mat.shape[1] > 1:
            mat = mat[0, :];
            mat = mat.reshape(mat.shape[1])
        '''
        if np.isnan(mat).sum() > 0:
            return None
        return mat
    else:
        print('load_mat {} failed'.format(filename))
        return None


def load_audio_feat_len(audio_path):
    path, pos = audio_path.split(':')
    ark_read_buffer = open(path, 'rb')
    ark_read_buffer.seek(int(pos), 0)
    header = struct.unpack('<xcccc', ark_read_buffer.read(5))
    if header[0].decode('utf-8') != "B":
        raise Exception("Input .ark file is not binary")
    if header[1].decode('utf-8') == "C":
        raise Exception("Input .ark file is compressed")

    _, rows = struct.unpack('<bi', ark_read_buffer.read(5))

    return rows


def load_npy(filename):
    if os.path.exists(filename):
        mat = np.load(filename)
        if np.isnan(mat).sum() > 0:
            return None
        return mat
    else:
        print('load_npy {} failed'.format(filename))
        return None


def target_trans(target, frame_num, word_dict):
    target_size = len(target)
    if target_size == frame_num:
        final_target = list(int(x) for x in list(target) if x is not None)
    else:
        final_target = list(int(target[int(x * target_size / frame_num)]) for x in range(frame_num))

    return final_target


class Targetcounter(object):
    def __init__(self, target_path, label_num):
        self.target_dict = read_target_file(target_path)
        self.label_num = label_num

    def compute_target_count(self):
        encoded_targets = np.concatenate(
            [self.encode(targets)
             for targets in self.target_dict.values()])

        #  count the number of occurences of each target
        count = np.bincount(encoded_targets,
                            minlength=self.label_num)
        return count

    def encode(self, targets):
        """ encode a target sequence """
        encoded_targets = list([int(x) for x in list(targets)])
        return np.array(encoded_targets, dtype=np.int)


def read_target_file(target_path):
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
            target_dict[split_line[0]] = split_line[1:]
    return target_dict


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


def add_delta(utt, delta_order):
    num_frames = utt.shape[0]
    feat_dim = utt.shape[1]

    utt_delta = np.zeros(
        shape=[num_frames, feat_dim * (1 + delta_order)],
        dtype=np.float32)

    #  first order part is just the utterance max_offset+1
    utt_delta[:, 0:feat_dim] = utt

    scales = [[1.0], [-0.2, -0.1, 0.0, 0.1, 0.2],
              [0.04, 0.04, 0.01, -0.04, -0.1, -0.04, 0.01, 0.04, 0.04]]

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


class AudioParser(object):
    '''def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError'''

    def WaveData(self, audio_path):
        y = load_audio(audio_path)
        return y

    def WriteWave(self, audio_path, data, sample_rate=16000):
        if data is not None:
            wav.write(audio_path, sample_rate, data)

    def RMRData(self, rmr_path, mat_key='rir', mic = 0):
        y = load_mat(rmr_path, mat_key)
        if y.shape[0] > 100:
            y = y[:, mic]
        elif y.shape[1] > 100:
            y = y[mic, :]
        y = y.squeeze()
        return y

    def MakeMixture(self, speech, noise, db):
        if speech is None or noise is None:
            # print("MakeMixture: speech is None or noise is None")
            return None
        if np.sum(np.square(noise)) < 1.0e-13:
            #print("MakeMixture: np.sum(np.square(noise)) < 1.0e-6")
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
            noi_pow = np.sum(np.square(noise))
            if noi_pow > 0:
                noi_scale = math.sqrt(np.sum(np.square(speech)) / (noi_pow * (10 ** (db / 10.0))))
            else:
                return None
        except:
            return None

        nnoise = noise * noi_scale
        mixture = speech + nnoise
        mixture = mixture.astype('float32')
        return mixture

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
        nsample = len(speech)
        #speech = signal.convolve(speech, rmr, mode='same')
        speech = signal.convolve(speech, rmr, mode='full')
        speech = speech[0:nsample]
        return speech

    def Gain_Control(self, wave, Gain):

        if wave is None:
            # print("Gain_Control: wave is None")
            return None

        max_sample = np.max(np.fabs(wave))

        if max_sample <= 0:
            # print("Gain_Control: np.fabs(wave) is 0")
            return None

        wave = wave / max_sample

        wave = wave * Gain

        wave = wave.astype('float32')

        return wave

    def Make_Noisy_Wave(self, speech, noise, spe_rmr, noi_rmr, SNR):
        if speech is None or noise is None:
            # print("Make_Noisy_Wave:speech is None or noise is None")
            return None

        if spe_rmr is not None:
            speech = self.Make_Reverberation(speech, spe_rmr)
        if noi_rmr is not None:
            noise = self.Make_Reverberation(noise, noi_rmr)

        noisy = self.MakeMixture(speech, noise, SNR)

        if noisy is not None:
            noisy = noisy.astype('float32')
            return noisy
        else:
            return None

