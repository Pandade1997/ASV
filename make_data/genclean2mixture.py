#!/usr/bin/python3
# coding:utf-8

import io
import sys
import os
import codecs
import random
from py_utils import file_parse

if __name__ == '__main__':
    '''
    Function describtion
        Generate the clean2noisy according to the utt2info
        clean2noisy format is as follow
            spe_key mix_key_1 mix_key_2 mix_key_2, ..., mix_key_n
    
    Input parameters
        param: utt2info: /home/edison_su/works/data/noisy_data/train/utt2info
        param: out_path: /home/edison_su/works/data/noisy_data/train/
    '''

    utt2info_file = sys.argv[1]
    out_path = sys.argv[2]
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    clean2noisy_file = os.path.join(out_path, 'clean2noisy')
    clean2noisy_writer = codecs.open(clean2noisy_file, 'w', 'utf-8')

    clean2noisy_dict = {}
    num_scp = 0
    encoder = file_parse.detectEncoding(utt2info_file)
    scp_reader = codecs.open(utt2info_file, 'r', encoder)
    for scp in scp_reader.readlines():
        
        scp = scp.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
        splits = scp.split(',')
        if len(splits) >= 2:
            mix_key = splits[0]
            spe_key = splits[1]
        else:
            continue
        num_scp = num_scp + 1
        print("[%d] %s" % (num_scp, mix_key))
        
        
        clean2noisy = mix_key
        if spe_key in clean2noisy_dict.keys():
            clean2noisy = clean2noisy_dict[spe_key] + " " + clean2noisy
        clean2noisy_dict[spe_key] = clean2noisy
    scp_reader.close()

    for spe_key, clean2noisy in clean2noisy_dict.items():
        clean2noisy_writer.write("%s %s\n" % (spe_key, clean2noisy))
    clean2noisy_writer.close()
    print("Succed to Finish!")
    
