from __future__ import print_function

from config import TrainOptions

import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from model import DeepSpeakerModel, DeepSpeakerSeqModel, DeepSpeakerCnnModel, DeepSpeakerCnnSeqModel, load, normalize
from data_loader import DeepSpeakerDataset, DeepSpeakerDataLoader, DeepSpeakerSeqDataset, DeepSpeakerSeqDataLoader

from utils import create_output_dir, processDataTable2

def compute_embedding(opt, model, data_mat):
    segment_length = int((opt.min_segment_length + opt.max_segment_length) / 2)
    segment_shift = int(opt.segment_shift_rate * segment_length)
    input_data = None
    seq_data = 0
    length = data_mat.size(0)
    #print('data_mat ', data_mat.shape)
    for x in range(0, length, segment_shift):
        end = x + segment_length
        if end < length:
            feature_mat = data_mat[x:end, :, :]
        else:
            if x == 0:
                input_data = data_mat
                seq_data += 1
            break
        seq_data += 1
        if input_data is None:
            input_data = feature_mat
        else:
            input_data = torch.cat((input_data, feature_mat), 1)
    input_data = input_data.to(opt.device)
    seq_data = torch.LongTensor([seq_data]).to(opt.device)

    #print('input_data ', input_data.shape, seq_data)
    if opt.seq_training == 'true':
        if opt.model_type == 'cnn':
            input_data = input_data.transpose(0, 1).unsqueeze(1).to(opt.device)
        out_data, w, b, _ = model(input_data, seq_data)   
    else:
        if opt.model_type == 'lstm' or opt.model_type == 'blstmp':
            out_data, w, b = model(input_data)
        elif opt.model_type == 'cnn':
            input_data = input_data.transpose(0, 1).unsqueeze(1).to(opt.device)
            out_data, w, b = model(input_data)

    out = torch.mean(out_data, 0)
    #print('out_data ', out_data.shape, out.shape)
    out = normalize(out)
    out = out.detach().cpu().numpy()
    out = out[np.newaxis, :]
    return out_data.detach().cpu().numpy(), out, w, b   

def evaluate(opt, model, val_loader, logging):
    if isinstance(model, DistributedDataParallel):
        model = model.module
    model.eval()
    
    vector_dir = os.path.join(opt.log_dir, 'vector')
    if not os.path.exists(vector_dir):
        os.makedirs(vector_dir)
    fwrite = open(os.path.join(opt.log_dir, 'results'), 'w')
    # show progress bar only on main process.
    valid_enum = tqdm(val_loader, desc='Valid')

    segment_length = int((opt.min_segment_length + opt.max_segment_length) / 2)
    segment_shift = int(opt.segment_shift_rate * segment_length)

    embedding_mean_probs = []
    score_mean_probs = []

    probs = []
    labels = []
    for i, (data) in enumerate(valid_enum, start=0):
        with torch.no_grad():
            utt_id_list, enroll_mat_list, test_0, label = data
            #print(utt_id_list)
            #utt_id_list = utt_id_list[0]
            x = 0 
            #print(utt_id_list, len(utt_id_list), len(enroll_mat_list))  
            enroll_output_list = []   
            for enroll_mat in enroll_mat_list:
                #print('enroll_mat', enroll_mat.shape)
                enroll_mat = enroll_mat.to(opt.device).transpose(0, 1)
                enroll_outputs, enroll_output, w, b = compute_embedding(opt, model, enroll_mat)
                X1 = np.concatenate((enroll_outputs, enroll_output), 0)
                np.save(os.path.join(vector_dir, utt_id_list[x][0]), X1)
                enroll_output_list.append(enroll_output)
                x += 1
            test_0 = test_0.to(opt.device).transpose(0, 1)
            test_0_outputs, test_0_output, w, b = compute_embedding(opt, model, test_0)
            X1 = np.concatenate((test_0_outputs, test_0_output), 0)
            np.save(os.path.join(vector_dir, utt_id_list[-1][0]), X1)
            
            vectors = np.concatenate(enroll_output_list, axis=0)
            vector_avg = np.mean(vectors, axis=0)
            #print('vector_avg ', vectors.shape, vector_avg.shape)
            vector_avg = vector_avg / np.sqrt(np.sum(vector_avg**2, axis=-1, keepdims=True) + 1e-6)

            prob = test_0_output * vector_avg
            prob = np.sum(prob)
            prob = np.abs(w.detach().cpu().numpy()) * prob + b.detach().cpu().numpy()
            #result = float(1.0 / (1.0 + np.exp(prob)))
            result = float(prob)
            embedding_mean_probs.append(result)
            labels.append(int(label.detach().cpu().numpy()))
            fwrite.write('{0} {1} {2:0.4f} {3}'.format(utt_id_list[0], utt_id_list[-1], result, labels[-1]) + '\n')
    embedding_mean_eer, embedding_mean_thresh = processDataTable2(np.array(labels), np.array(embedding_mean_probs))
    fwrite.write('embedding_mean_eer {} embedding_mean_thresh {}'.format(embedding_mean_eer, embedding_mean_thresh))
    fwrite.close()
    logging.info("embedding_mean_EER : %0.4f (thres:%0.4f)" % (embedding_mean_eer, embedding_mean_thresh))
    eer = embedding_mean_eer
    return eer
    
# Prepare the parameters
opt = TrainOptions().parse()

## set seed ##
if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
print('manual_seed = %d' % opt.manual_seed)
random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)

# Configure the distributed training
opt.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
opt.distributed = opt.num_gpus > 1
if opt.cuda:
    opt.device = torch.device("cuda")
else:
    opt.device = torch.device("cpu")

opt.main_proc = True

## prepara dir for training ##
if not os.path.isdir(opt.works_dir):
    try:
        os.makedirs(opt.works_dir)
    except OSError:
        print("ERROR: %s is not a dir" % (opt.works_dir))
        pass

opt.exp_path = os.path.join(opt.works_dir, 'exp')
if not os.path.isdir(opt.exp_path):
    os.makedirs(opt.exp_path)

opt.log_dir = os.path.join(opt.exp_path, opt.model_name)
if not os.path.exists(opt.log_dir):
    os.makedirs(opt.log_dir)

opt.model_dir = os.path.join(opt.exp_path, opt.model_name)
if not os.path.exists(opt.model_dir):
    os.makedirs(opt.model_dir)

if opt.cmvn_file is not None:
    opt.cmvn_file = os.path.join(opt.model_dir, opt.cmvn_file)
    
## Create a logger for loging the training phase ##
logging = create_output_dir(opt, opt.log_dir)

## Data Prepare ##
logging.info("Building dataset")
if opt.seq_training == 'true':
    opt.data_type = 'test'
    val_dataset = DeepSpeakerSeqDataset(opt, os.path.join(opt.dataroot, 'train'))
    val_loader = DeepSpeakerSeqDataLoader(val_dataset, batch_size=1, num_workers=opt.num_workers, shuffle=False,
                                          pin_memory=True)
else:
    opt.data_type = 'test'
    val_dataset = DeepSpeakerDataset(opt, os.path.join(opt.dataroot, 'train'))
    val_loader = DeepSpeakerDataLoader(val_dataset, batch_size=1, num_workers=opt.num_workers, shuffle=False,
                                       pin_memory=True)
logging.info("Building dataset Sucessed")

##  Building Model ##
logging.info("Building Model")
opt.in_size = val_dataset.in_size
if opt.seq_training == 'true':
    if opt.model_type == 'lstm' or opt.model_type == 'blstmp':
        model = DeepSpeakerSeqModel(opt)
    elif opt.model_type == 'cnn':
        model = DeepSpeakerCnnSeqModel(opt)
    else:
        raise Exception('wrong model_type {}'.format(opt.model_type))
else:
    if opt.model_type == 'lstm' or opt.model_type == 'blstmp':
        model = DeepSpeakerModel(opt)
    elif opt.model_type == 'cnn':
        model = DeepSpeakerCnnModel(opt)
    else:
        raise Exception('wrong model_type {}'.format(opt.model_type))

if opt.resume:
    model, opt.steps = load(model, opt.resume)
else:
    raise Exception('wrong opt.resume {}'.format(opt.resume))
    
model.to(opt.device)

print(model)
logging.info("Building Model Sucessed")

## model testing ##
evaluate(opt, model, val_loader, logging)

