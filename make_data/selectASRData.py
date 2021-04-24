#!/usr/bin/python3
# coding:utf-8

import io
import sys
import os
import codecs
import random
from py_utils import file_parse
from py_utils.audio_io import check_audio
import jieba

if __name__ == '__main__':
    '''
    Function describtion
    Given the dataset list,choose the utterance to train asr model. spk2utt, text, utt2spk and wav.scp will
    be generated in the output path
    
    Input parameters
    param: scp_lst: /mnt/nlpr/DATA/Audio/Chinese/data_scp/code/scp.lst
           aishell2 1.0 /mnt/nlpr/DATA/Audio/Chinese/data_scp/data/AISHELL-2/wav.scp path=-5:-4:-3:-2
           aishell1 1.0 /mnt/nlpr/DATA/Audio/Chinese/data_scp/data/AiShell/wav.scp path=-3:-2
           ... ...
     param: out_path: /mnt/nlpr/DATA/Audio/Chinese/863_2/
    '''
    scp_lst = sys.argv[1]
    vocab_file = sys.argv[2]
    out_path = sys.argv[3]
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    try:
        encoder = file_parse.detectEncoding(scp_lst)
        scp_lst_reader = codecs.open(scp_lst, 'r', encoding=encoder)
        log_file = os.path.join(out_path, 'log')
        log_writer = codecs.open(log_file, 'w', encoding='utf-8')
    except IOError:
        exit("Error: Cann't Open %s" % (scp_lst_reader))

    jieba.set_dictionary(vocab_file)

    scp_file = os.path.join(out_path, 'wav.scp')
    utt2spk_file = os.path.join(out_path, 'utt2spk')
    spk2utt_file = os.path.join(out_path, 'spk2utt')
    utt2data_file = os.path.join(out_path, 'utt2data')
    data2utt_file = os.path.join(out_path, 'data2utt')

    scp_writer = codecs.open(scp_file, 'w', 'utf-8')
    utt2spk_writer = codecs.open(utt2spk_file, 'w', 'utf-8')
    spk2utt_writer = codecs.open(spk2utt_file, 'w', 'utf-8')
    utt2data_writer = codecs.open(utt2data_file, 'w', 'utf-8')
    data2utt_writer = codecs.open(data2utt_file, 'w', 'utf-8')

    num_utts = 0
    data2utt_dict = {}
    spk2utt_dict = {}
    for line in scp_lst_reader.readlines():
        line = line.replace('\n', '').replace('\r', '').replace('\t', ' ')
        splits = line.split()
        if len(splits) < 4:
            continue

        data_id = splits[0]
        data_rate = float(splits[1])

        path_idx = []
        name_idx = []
        for utt2spk_info in splits[3:]:
            if "path=" in utt2spk_info:
                utt2spk_info = utt2spk_info.split('=')[1]
                path_location = utt2spk_info.split(':')
                for idx in path_location:
                    path_idx.append(int(idx))
            elif "name=" in utt2spk_info:
                utt2spk_info = utt2spk_info.split('=')[1]
                name_location = utt2spk_info.split(':')
                for idx in name_location:
                    name_idx.append(int(idx))

        in_path, _, _, _ = file_parse.getFileDetailInfo(splits[2], -2)

        in_scp_file = os.path.join(in_path, 'wav.scp')
        encoder = file_parse.detectEncoding(in_scp_file)
        scp_reader = codecs.open(in_scp_file, 'r', encoder)
        for scp in scp_reader.readlines():
            if random.random() > data_rate:
                continue
            scp = scp.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
            splits = scp.split()
            if len(splits) >= 2:
                key = splits[0]
                file_path = splits[1].strip().replace('\n', '').replace('\r', '')
            else:
                continue

            # Check the wav
            if not check_audio(file_path, 4000):
                print("%s is not existed or is too small\n" % (file_path))
                log_writer.write("%s is not existed or is too small\n" % (file_path))
                continue

            spk_id = ""
            for idx in path_idx:
                _, spk_info, _, _ = file_parse.getFileDetailInfo(file_path, idx)
                spk_id = spk_id + spk_info
            _, _, filename, _ = file_parse.getFileDetailInfo(file_path, -2)
            sub_parts = filename.split('_')
            for idx in name_idx:
                spk_id = spk_id + sub_parts[idx]

            spk_id = data_id + spk_id
            spk_id = spk_id.strip().replace('_', '').replace('-', '').replace('.', '')
            # spk_id_len = len(spk_id)
            # spk_id_len = (5 - spk_id_len % 5) + spk_id_len
            spk_id = spk_id.ljust(40, '0')
            num_utts = num_utts + 1
            filename = "%08d" % (num_utts)
            # filename = filename.strip().replace('_', '').replace('-', '').replace('.', '')
            utt_id = spk_id + "_" + filename

            print("%s %s\n" % (utt_id, file_path))
            utt2spk_writer.write("%s %s\n" % (utt_id, spk_id))
            utt2data_writer.write("%s %s\n" % (utt_id, data_id))
            scp_writer.write("%s %s\n" % (utt_id, file_path))

            spk2utt = utt_id
            if spk_id in spk2utt_dict.keys():
                spk2utt = spk2utt_dict[spk_id] + " " + spk2utt
            spk2utt_dict[spk_id] = spk2utt

            data2utt = utt_id
            if data_id in data2utt_dict.keys():
                data2utt = data2utt_dict[data_id] + " " + data2utt
            data2utt_dict[data_id] = data2utt
        scp_reader.close()

    for spk_id, utt_id in spk2utt_dict.items():
        print("%s %s\n" % (spk_id, utt_id))
        spk2utt_writer.write("%s %s\n" % (spk_id, utt_id))

    for data_id, utt_id in data2utt_dict.items():
        data2utt_writer.write("%s %s\n" % (data_id, utt_id))
    data2utt_writer.close()

    scp_lst_reader.close()
    scp_writer.close()
    utt2spk_writer.close()
    spk2utt_writer.close()
    utt2data_writer.close()
    data2utt_writer.close()
    log_writer.close()
    print("Succed to Finish!")
