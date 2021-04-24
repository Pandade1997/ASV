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
        param: in_path: /home/edison_su/works/projects/train_asr/data/local/
        param: train_dev_test: 8.0:1.0:1.0
        param: out_path: /home/edison_su/works/projects/train_asr/data
    '''
    in_path = sys.argv[1]
    train_dev_test = sys.argv[2]
    out_path = sys.argv[3]

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

    scp_file = os.path.join(in_path, 'wav.scp')
    utt2spk_file = os.path.join(in_path, 'utt2spk')
    spk2utt_file = os.path.join(in_path, 'spk2utt')
    utt2data_file = os.path.join(in_path, 'utt2data')
    data2utt_file = os.path.join(in_path, 'data2utt')
    try:
        encoder = file_parse.detectEncoding(scp_file)
        scp_reader = codecs.open(scp_file, 'r', encoding=encoder)
        encoder = file_parse.detectEncoding(utt2spk_file)
        utt2spk_reader = codecs.open(utt2spk_file, 'r', encoding=encoder)
        encoder = file_parse.detectEncoding(spk2utt_file)
        spk2utt_reader = codecs.open(spk2utt_file, 'r', encoding=encoder)
        encoder = file_parse.detectEncoding(utt2data_file)
        utt2data_reader = codecs.open(utt2data_file, 'r', encoding=encoder)
        encoder = file_parse.detectEncoding(data2utt_file)
        data2utt_reader = codecs.open(data2utt_file, 'r', encoding=encoder)
    except IOError:
        exit("Error: Cann't Open %s, %s and %s" % (scp_file, utt2spk_file, spk2utt_file))

    scp_list = []
    for line in scp_reader.readlines():
        line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
        splits = line.split()
        if len(splits) < 2:
            continue
        utt_id = splits[0]
        utt_path = splits[1]
        scp_list.append((utt_id, utt_path))



    utt2spk_dict = []
    for line in utt2spk_reader.readlines():
        line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
        splits = line.split()
        if len(splits) < 2:
            continue
        utt_id = splits[0]
        spk_id = splits[1]
        utt2spk_dict.append((utt_id, spk_id))

    utt2data_dict = []
    for line in utt2data_reader.readlines():
        line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
        splits = line.split()
        if len(splits) < 2:
            continue
        utt_id = splits[0]
        data_id = splits[1]
        utt2data_dict.append((utt_id, data_id))

    num_utt = len(scp_list)

    print("Dataset have %d utterances" % (num_utt))

    scp_reader.close()
    utt2spk_reader.close()
    spk2utt_reader.close()
    utt2data_reader.close()
    data2utt_reader.close()

    train_utt_num = int(math.floor(train_rate * num_utt))
    dev_utt_num = int(math.floor(dev_rate * num_utt))
    test_utt_num = int(num_utt - train_utt_num - dev_utt_num)

    utt_idx = np.random.permutation(num_utt)

    train_utt_idx = utt_idx[0:train_utt_num]
    dev_utt_idx = utt_idx[train_utt_num:train_utt_num + dev_utt_num]
    test_utt_idx = utt_idx[train_utt_num + dev_utt_num:]

    ## Training set ##
    print("gen Training set: %d utterances" % (len(train_utt_idx)))
    out_scp_file = os.path.join(train_path, 'wav.scp')
    out_utt2spk_file = os.path.join(train_path, 'utt2spk')
    out_spk2utt_file = os.path.join(train_path, 'spk2utt')
    out_utt2data_file = os.path.join(train_path, 'utt2data')
    out_data2utt_file = os.path.join(train_path, 'data2utt')

    scp_writer = codecs.open(out_scp_file, 'w', 'utf-8')
    utt2spk_writer = codecs.open(out_utt2spk_file, 'w', 'utf-8')
    spk2utt_writer = codecs.open(out_spk2utt_file, 'w', 'utf-8')
    utt2data_writer = codecs.open(out_utt2data_file, 'w', 'utf-8')
    data2utt_writer = codecs.open(out_data2utt_file, 'w', 'utf-8')

    data2utt_dict = {}
    spk2utt_dict = {}
    for i in train_utt_idx:
        utt_id, scp_path = scp_list[i]
        spk_utt_id, spk_id = utt2spk_dict[i]
        data_utt_id, data_id = utt2data_dict[i]
        
        if spk_utt_id == utt_id and data_utt_id == utt_id:
            scp_writer.write("%s %s\n" % (utt_id, scp_path))
            utt2spk_writer.write("%s %s\n" % (utt_id, spk_id))
            utt2data_writer.write("%s %s\n" % (utt_id, data_id))
    
            spk2utt = utt_id
            if spk_id in spk2utt_dict:
                spk2utt = spk2utt_dict[spk_id] + " " + spk2utt
            spk2utt_dict[spk_id] = spk2utt
    
            data2utt = utt_id
            if data_id in data2utt_dict:
                data2utt = data2utt_dict[data_id] + " " + data2utt
            data2utt_dict[data_id] = data2utt
        else:
            print("error: utt_id text_utt_id spk_utt_id and data_utt_id are not equal")
        
    for spk_id, utt_ids in spk2utt_dict.items():
        spk2utt_writer.write("%s %s\n" % (spk_id, utt_ids))
    for data_id, utt_ids in data2utt_dict.items():
        data2utt_writer.write("%s %s\n" % (data_id, utt_ids))

    scp_writer.close()
    utt2spk_writer.close()
    spk2utt_writer.close()
    utt2data_writer.close()
    data2utt_writer.close()

    ## Dev set ##
    print("gen Dev set: %d utterances" % (len(dev_utt_idx)))
    out_scp_file = os.path.join(dev_path, 'wav.scp')
    out_utt2spk_file = os.path.join(dev_path, 'utt2spk')
    out_spk2utt_file = os.path.join(dev_path, 'spk2utt')
    out_utt2data_file = os.path.join(dev_path, 'utt2data')
    out_data2utt_file = os.path.join(dev_path, 'data2utt')

    scp_writer = codecs.open(out_scp_file, 'w', 'utf-8')
    utt2spk_writer = codecs.open(out_utt2spk_file, 'w', 'utf-8')
    spk2utt_writer = codecs.open(out_spk2utt_file, 'w', 'utf-8')
    utt2data_writer = codecs.open(out_utt2data_file, 'w', 'utf-8')
    data2utt_writer = codecs.open(out_data2utt_file, 'w', 'utf-8')

    data2utt_dict = {}
    spk2utt_dict = {}
    for i in dev_utt_idx:
        utt_id, scp_path = scp_list[i]
        spk_utt_id, spk_id = utt2spk_dict[i]
        data_utt_id, data_id = utt2data_dict[i]
        
        if spk_utt_id == utt_id and data_utt_id == utt_id:
            scp_writer.write("%s %s\n" % (utt_id, scp_path))
            utt2spk_writer.write("%s %s\n" % (utt_id, spk_id))
            utt2data_writer.write("%s %s\n" % (utt_id, data_id))
    
            spk2utt = utt_id
            if spk_id in spk2utt_dict:
                spk2utt = spk2utt_dict[spk_id] + " " + spk2utt
            spk2utt_dict[spk_id] = spk2utt
    
            data2utt = utt_id
            if data_id in data2utt_dict:
                data2utt = data2utt_dict[data_id] + " " + data2utt
            data2utt_dict[data_id] = data2utt
        else:
            print("error: utt_id text_utt_id spk_utt_id and data_utt_id are not equal")
        
    for spk_id, utt_ids in spk2utt_dict.items():
        spk2utt_writer.write("%s %s\n" % (spk_id, utt_ids))
    for data_id, utt_ids in data2utt_dict.items():
        data2utt_writer.write("%s %s\n" % (data_id, utt_ids))

    scp_writer.close()
    utt2spk_writer.close()
    spk2utt_writer.close()
    utt2data_writer.close()
    data2utt_writer.close()

    ## Test set ##
    print("gen Test set: %d utterances" % (len(test_utt_idx)))
    out_scp_file = os.path.join(test_path, 'wav.scp')
    out_utt2spk_file = os.path.join(test_path, 'utt2spk')
    out_spk2utt_file = os.path.join(test_path, 'spk2utt')
    out_utt2data_file = os.path.join(test_path, 'utt2data')
    out_data2utt_file = os.path.join(test_path, 'data2utt')

    scp_writer = codecs.open(out_scp_file, 'w', 'utf-8')
    utt2spk_writer = codecs.open(out_utt2spk_file, 'w', 'utf-8')
    spk2utt_writer = codecs.open(out_spk2utt_file, 'w', 'utf-8')
    utt2data_writer = codecs.open(out_utt2data_file, 'w', 'utf-8')
    data2utt_writer = codecs.open(out_data2utt_file, 'w', 'utf-8')

    data2utt_dict = {}
    spk2utt_dict = {}
    for i in test_utt_idx:
        utt_id, scp_path = scp_list[i]
        spk_utt_id, spk_id = utt2spk_dict[i]
        data_utt_id, data_id = utt2data_dict[i]
        
        if spk_utt_id == utt_id and data_utt_id == utt_id:
            scp_writer.write("%s %s\n" % (utt_id, scp_path))
            utt2spk_writer.write("%s %s\n" % (utt_id, spk_id))
            utt2data_writer.write("%s %s\n" % (utt_id, data_id))
    
            spk2utt = utt_id
            if spk_id in spk2utt_dict:
                spk2utt = spk2utt_dict[spk_id] + " " + spk2utt
            spk2utt_dict[spk_id] = spk2utt
    
            data2utt = utt_id
            if data_id in data2utt_dict:
                data2utt = data2utt_dict[data_id] + " " + data2utt
            data2utt_dict[data_id] = data2utt
        else:
            print("error: utt_id text_utt_id spk_utt_id and data_utt_id are not equal")
            
    for spk_id, utt_ids in spk2utt_dict.items():
        spk2utt_writer.write("%s %s\n" % (spk_id, utt_ids))
    for data_id, utt_ids in data2utt_dict.items():
        data2utt_writer.write("%s %s\n" % (data_id, utt_ids))

    scp_writer.close()
    utt2spk_writer.close()
    spk2utt_writer.close()
    utt2data_writer.close()
    data2utt_writer.close()

    print("Succed to Finish!")
