import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from options.config import TrainOptions
from model.model import model_select, load
from data.utt_data_loader import DeepSpeakerUttDataset, DeepSpeakerUttDataLoader
from data import kaldi_io
 

def extract_feature(opt, inputs, model, segment_length, segment_shift):
    input_data = None
    length = inputs.size(1)
    seq_len = 0
    for x in range(0, length, segment_shift):
        end = x + segment_length
        if end < length:
            feature_mat = inputs[:, x:end, :]
        else:
            if x == 0:
                input_data = inputs
                seq_len += 1
            break
        seq_len += 1
        if input_data is None:
            input_data = feature_mat
        else:
            input_data = torch.cat((input_data, feature_mat), 0) 
    seq_len = torch.LongTensor([seq_len]).to(opt.device) 
    output, w, b = model(input_data, seq_len)
    output_mean = torch.mean(output, dim=0)
    output_mean_norm = output_mean / torch.sqrt(torch.sum(output_mean**2, dim=-1, keepdim=True) + 1e-6)   
    return output_mean_norm


def evaluate(opt, model, val_loader):
    if isinstance(model, DistributedDataParallel):
        model = model.module
    model.eval()
    
    vector_dir = os.path.join(opt.works_dir, opt.exp_path, 'xvector')
    if not os.path.exists(vector_dir):
        os.makedirs(vector_dir)

    mag_ark_scp_output = 'ark:| copy-vector ark:- ark,scp:{0}/xvector{1}.ark,{0}/xvector{1}.scp'.format(vector_dir, opt.thread_num) 

    segment_length = int((opt.min_segment_length + opt.max_segment_length) / 2)
    segment_shift = int(opt.segment_shift_rate * segment_length)
    with kaldi_io.open_or_fd(mag_ark_scp_output,'wb') as f_mag:
        for i, (data) in enumerate(val_loader, start=0):
            with torch.no_grad():
                utt_ids, inputs, targets = data
                #print(utt_ids)            
                inputs = inputs.to(opt.device)
                output_mean_norm = extract_feature(opt, inputs, model, segment_length, segment_shift)
                output_mean_norm = output_mean_norm.detach().cpu().numpy()     
                #print(utt#_id, enroll_mat.shape, enroll_outputs.shape, enroll_output.shape)
                kaldi_io.write_vec_flt(f_mag, output_mean_norm, key=utt_ids[0]) 
                if i % 100 == 0:
                    print(i)
                
    print('finish')

if __name__ == "__main__":        
    # Prepare the parameters
    opt = TrainOptions().parse()
    
    # Configure the distributed training
    if opt.cuda:
        opt.device = torch.device("cuda")
    else:
        opt.device = torch.device("cpu")
    
    ## Data Prepare ##
    print("Building dataset")
    val_dataset = DeepSpeakerUttDataset(opt, opt.dataroot)
    val_loader = DeepSpeakerUttDataLoader(val_dataset, batch_size=1, num_workers=opt.num_workers, shuffle=False)
    opt.in_size = val_dataset.in_size
    print('opt.in_size = {}'.format(opt.in_size))  
    print("Building dataset Sucessed")
    
    ##  Building Model ##
    print("Building Model")
    seq_training = False
    if opt.loss_type.split('_')[0] == 'class-seq' or opt.loss_type.split('_')[0] == 'seq':
        seq_training=True    
    model = model_select(opt, seq_training)
    if opt.resume:
        model, opt.total_iters = load(model, opt.resume, 'state_dict')
    else:
        raise Exception('wrong opt.resume {}'.format(opt.resume))    
    model.to(opt.device)
    print(model)
    print("Building Model Sucessed")
    
    ## model testing ##
    evaluate(opt, model, val_loader)

