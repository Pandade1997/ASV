import os
import math
import gzip
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data.sampler import Sampler

from options.config import TrainOptions
from data.base_data_loader import BaseDataset, make_utt2spk, dump_to_text


class DeepSpeakerUttSeqDataset(BaseDataset):
    def __init__(self, opt, data_dir):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        utt_id /path/to/audio.wav
        ...
        :param data_scp: Path to scp as describe above
        :param label_file : Dictionary containing the delta_order, context_width, normalize_type and max_num_utt_cmvn
        :param audio_conf: Dictionary containing the sample_rate, num_channel, window_size window_shift
        """    
        self.feats_scp = []
        feats_scp_file = os.path.join(data_dir, 'feats.scp')        
        with open(feats_scp_file, 'r', encoding='utf-8') as scp_reader:
            for line in scp_reader:
                line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
                splits = line.split()
                if len(splits) < 2:
                    continue
                utt_id = splits[0]
                utt_path = splits[1]
                self.feats_scp.append((utt_id, utt_path))                    
        self.spe_size = len(self.feats_scp)
        self.num_utt = len(self.feats_scp)
        
        utt2spk_file = os.path.join(data_dir, 'utt2spk')
        self.utt2spk_ids, spk2ids = make_utt2spk(utt2spk_file, self.feats_scp)
        dump_to_text(spk2ids, os.path.join(opt.exp_path, 'spk2ids'))
        self.class_nums = len(spk2ids)
        print('The dataset have {} utts and {} speakers'.format(self.spe_size, self.class_nums))

        for n in range(self.spe_size):
            feat_path = self.feats_scp[n][1]
            raw_in_feat = self.load_feat(feat_path)
            raw_delta_in_feat = self.load_feat(feat_path, opt.delta_order)
            in_feat = self.parse_feat(feat_path, opt.delta_order, None, None, opt.left_context_width, opt.right_context_width)    
            if in_feat is not None and raw_in_feat is not None:
                self.raw_feat_size = np.shape(raw_in_feat)[1]
                self.feat_size = np.shape(raw_delta_in_feat)[1]
                self.in_size = np.shape(in_feat)[1]                
                break
        print('feat_size {}'.format(self.feat_size))
        if self.feat_size < 0:
            raise Exception('Wrong feat_size {}'.format(self.feat_size))
                    
        super(DeepSpeakerUttSeqDataset, self).__init__(opt, data_dir)

    def __getitem__(self, index):
        while True:
            utt_id, utt_path = self.feats_scp[index]
            spk_id = self.utt2spk_ids[utt_id]
            try:
                vad_idx = self.utt2vad[utt_id]
            except:
                vad_idx = None
            in_feat = self.parse_feat(utt_path, self.delta_order, vad_idx, self.cmvn, self.left_context_width,
                                      self.right_context_width)
            
            if in_feat is None:
                index = random.randint(0, self.spe_size - 1)
            else:
                break
      
        return utt_id, spk_id, torch.FloatTensor(in_feat)
                
    def __len__(self):
        return self.spe_size


def rand_seq_segment(feat, segment_length, segment_shift, num_segment_utt):
    if feat is None:
        return None
    num_frame = feat.shape[0]
    start_frame = 0
    end_frame = num_frame
                
    num_frame_needed = (num_segment_utt - 1) * (segment_length - segment_shift) + segment_length
    if num_frame_needed + 1 < num_frame:
        start_frame = np.random.randint(0, num_frame - num_frame_needed - 1)
        end_frame = start_frame + num_frame_needed
    
    all_feat = feat
    while all_feat.shape[0] < segment_length:
        all_feat = torch.cat([all_feat, feat], 0)
        
        seg_len = all_feat.shape[0]
        elen = seg_len - segment_length - 1
        if elen > 1:
            start_frame = np.round(random.randint(0, elen - 1))
        else:
            start_frame = 0
        end_frame = start_frame + segment_length
                
    segment_num = 0
    feat_mats = None
    for start in range(start_frame, end_frame, segment_shift):
        end = start + segment_length
        if end <= all_feat.shape[0] and segment_num < num_segment_utt:
            feat_mat_slice = all_feat[start:end, :].unsqueeze(0) 
            if feat_mats is None:
                feat_mats = feat_mat_slice
            else:
                feat_mats = torch.cat((feat_mats, feat_mat_slice), dim=0)
            segment_num += 1
    
    return feat_mats, segment_num     


# Prepare the parameters
opt = TrainOptions().parse()
min_segment_length, max_segment_length = opt.min_segment_length, opt.max_segment_length
min_num_segment, max_num_segment = opt.min_num_segment, opt.max_num_segment
segment_shift_rate = opt.segment_shift_rate

def _collate_fn(batch):
    if min_segment_length <= max_segment_length and max_segment_length > 0:
        segment_win = np.random.randint(min_segment_length, max_segment_length)
        segment_shift = int(np.round(segment_win * segment_shift_rate))
    else:
        segment_win = batch[0][2].size(0)
    freq_size = batch[0][2].size(1)
    minibatch_size = len(batch)
    inputs = torch.zeros(minibatch_size, segment_win, freq_size)
    inputs = torch.zeros(0, segment_win, freq_size)
    inputs = None
    segment_num = None
            
    utt_ids = []
    spk_ids = []
    segment_nums = []
    segment_targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        utt_id = sample[0]
        spk_id = int(sample[1])
        spect = sample[2]
                         
        utt_ids.append(utt_id)
        spk_ids.append(spk_id)
        num_segment_utt = np.random.randint(min_num_segment, max_num_segment)
        spect_slice, segment_num = rand_seq_segment(spect, segment_win, segment_shift, num_segment_utt)
        segment_nums.append(segment_num)
        segment_targets.extend([spk_id] * segment_num)
        if inputs is None:
            inputs = spect_slice
        else:
            inputs = torch.cat((inputs, spect_slice), dim=0)

    targets = torch.LongTensor(spk_ids)
    segment_nums = torch.LongTensor(segment_nums)
    segment_targets = torch.LongTensor(segment_targets)
    return utt_ids, inputs, segment_nums, targets, segment_targets
    
    
class DeepSpeakerUttSeqDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(DeepSpeakerUttSeqDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

