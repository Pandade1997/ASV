#!/usr/bin/python3
# coding:utf-8

import io
import sys
import os
import codecs
import random
from py_utils import file_parse
import jieba

if __name__ == '__main__':
    '''
    Function describtion
    Given the wav_list, spk2utt, text, utt2spk and wav.scp will
    be generated in the output path
    
    Input parameters
     param: out_path: /mnt/nlpr/DATA/Audio/Chinese/863_2/
    '''
    in_text_file    = sys.argv[1]
    vocab_file      = sys.argv[2]
    out_path        = sys.argv[3]
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    jieba.set_dictionary(vocab_file)

    text_file       = os.path.join(out_path, 'ttext')
    text_writer     = codecs.open(text_file, 'w', 'utf-8')
    
    # prepare text dict
    encoder = file_parse.detectEncoding(in_text_file)
    if encoder is None:
        encoder = 'utf-8'
    text_reader = open(in_text_file, 'rb')
    for line in text_reader.readlines():
        line = file_parse.to_str(line, encoder)
        line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
        splits = line.split()
        if len(splits) < 2:
            continue

        text_key = splits[0]
        text_value = ''.join(splits[1:])
        words = jieba.cut(text_value, HMM=False)  # turn off new word discovery (HMM-based)
        text_value = " ".join(words)
        text_value = file_parse.cleantxt(text_value)
        print("%s %s\n" % (text_key, text_value))
        text_writer.write("%s %s\n" % (text_key, text_value))

    text_reader.close()
    text_writer.close()
    print("Succed to Finish!")