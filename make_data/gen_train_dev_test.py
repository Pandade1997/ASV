#!/usr/bin/python3
# coding:utf-8

import io
import sys
import os
import codecs
import math
import numpy as np

from py_utils import file_parse

if __name__ == '__main__':
    '''
    Function describtion
    Divide the dataset into train, dev and test, all of train, dev and test contain spk2utt, text, utt2spk and wav.scp

    Input parameters
        param: mix_scp: /home/snie/works/projects/train_asr/data/local/mix.scp
        param: train_dev_test: 8.0:1.0:1.0
        param: out_path: /home/snie/works/projects/train_asr/data
    '''
    mix_scp = sys.argv[1]
    train_dev_test = sys.argv[2]
    out_path = sys.argv[3]
    
    dataset_path, dataset_name, dataset_type = file_parse.getFileInfo(mix_scp)
    
    train_dev_test = train_dev_test.split(':')
    if len(train_dev_test) != 3:
        print("Error: The input parameters is not supported!")
        print("Example: python gen_train_dev_test.py wav.scp text 8:1:1 data")
        exit(1)

    train = float(train_dev_test[0])
    dev = float(train_dev_test[1])
    test = float(train_dev_test[2])
    train_rate = train / (train + dev + test)
    dev_rate = dev / (train + dev + test)
    test_rate = test / (train + dev + test)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    train_path = os.path.join(out_path, 'train')
    dev_path = os.path.join(out_path, 'dev')
    test_path = os.path.join(out_path, 'test')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(dev_path):
        os.makedirs(dev_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    scp_list = None
    if not os.path.exists(mix_scp):
        exit("IOError: %s not existed" % (mix_scp))
    else:
        with open(mix_scp) as f:
            scp_list = f.readlines()

    num_utt = len(scp_list)

    train_utt_num = math.floor(train_rate * num_utt)
    dev_utt_num = math.floor(dev_rate * num_utt)
    test_utt_num = num_utt - train_utt_num - dev_utt_num

    utt_idx = np.random.permutation(num_utt)

    train_utt_idx = utt_idx[0:train_utt_num]
    dev_utt_idx = utt_idx[train_utt_num:train_utt_num + dev_utt_num]
    test_utt_idx = utt_idx[train_utt_num + dev_utt_num:]

    ## Training set ##
    print("gen Training set: %d utterances" % (len(train_utt_idx)))
    out_scp_file = os.path.join(train_path, dataset_name + dataset_type)
    scp_writer = codecs.open(out_scp_file, 'w', 'utf-8')
    for i in train_utt_idx:
        scp_writer.write("%s" % (scp_list[i]))
    scp_writer.close()

    ## Dev set ##
    print("gen Dev set: %d utterances" % (len(dev_utt_idx)))
    out_scp_file = os.path.join(dev_path, dataset_name + dataset_type)
    scp_writer = codecs.open(out_scp_file, 'w', 'utf-8')
    for i in dev_utt_idx:
        scp_writer.write("%s" % (scp_list[i]))
    scp_writer.close()

    ## Training set ##
    print("gen Test set: %d utterances" % (len(test_utt_idx)))
    out_scp_file = os.path.join(test_path, dataset_name + dataset_type)
    scp_writer = codecs.open(out_scp_file, 'w', 'utf-8')
    for i in test_utt_idx:
        scp_writer.write("%s" % (scp_list[i]))
    scp_writer.close()

    print("Succed to Finish!")