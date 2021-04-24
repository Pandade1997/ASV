#!/usr/bin/env python
import os
import sys 
import numpy as np
import kaldi_io


if __name__ == "__main__":    
    if len(sys.argv) < 3:
        print('Usage: %s data_dir P:percentage_of_cv/T:num_utts_of_tr/V:num_utts_of_cv repeats' % sys.argv[0])
        exit(1)
    
    npy_file = sys.argv[1]
    kaldi_file = sys.argv[2]
    
    mag_ark_scp_output = 'ark:| copy-vector ark:- ark,scp:{0}.ark,{0}.scp'.format(kaldi_file)
    num = 0
    with open(npy_file, 'r', encoding='utf-8') as fread, kaldi_io.open_or_fd(mag_ark_scp_output,'wb') as f_mag:
        for line in fread:
            line = line.strip().replace('\n', '')
            utt_id, feat_path = line.split(' ')
            output = np.load(feat_path)    
            #print(utt#_id, enroll_mat.shape, enroll_outputs.shape, enroll_output.shape)
            kaldi_io.write_vec_flt(f_mag, output, key=utt_id) 
            if num % 100 == 0:
                print(num)
            num += 1
                
    print('finish')

