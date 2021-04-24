#!/usr/bin/env python


'''
This is a minimal version just barely working.
It splits data dir into tr and cv based on spk info,
but different from original that each spk's utts will
be splited into both tr and cv, with a constant proportion or num utts.
If second param is integer, then constant num utts; otherwise proportion.

e.g.
spk_1 has 10 utts, spk_2 has 20.
When %P is set to 10%, 1 utt of spk_1 and 2 utts of spk_2 will
go into cv, others tr.

We'll split only ivectors.scp, spk2utt and utt2spk.
We don't support random seeds for now.
'''


import os
import sys 
import kaldi_io
from random import shuffle, random, seed


def spk2utt_to_utt2spk(spk2utt):
    utt2spk = []
    for spk in spk2utt:
        for utt in spk2utt[spk]:
            utt2spk.append((utt, spk))
    return utt2spk


if len(sys.argv) < 3:
    print('Usage: %s data_dir P:percentage_of_cv/T:num_utts_of_tr/V:num_utts_of_cv repeats' % sys.argv[0])
    exit(1)

data_dir = sys.argv[1]
p = int(sys.argv[2])

seed(2333)                

mode = 'num_utts_tr'
fwrite = open(os.path.join(data_dir, 'enroll_pairs.txt'), 'w')
feats_scp = {}
with open(os.path.join(data_dir, 'feats.scp'), 'r') as fp_in:
    for line in fp_in:
        utt_key, utt_scp = line.strip().split(None, 1)
        #try:
            #feat = kaldi_io.read_mat(utt_scp)
        feats_scp[utt_key] = utt_scp   
        #except:
            #print(utt_scp, ' has error')
spk_list = []
spk2utt = {}
num = 0
with open(os.path.join(data_dir, 'spk2utt'), 'r') as fp_in:
    for line in fp_in:
        tokens = line.strip().split()
        spk = tokens[0]
        utts = tokens[1:]
        valid_utts = []
        if num >= 100:
            break
        for utt in utts:
            try:
                utt_scp = feats_scp[utt]
                feat = kaldi_io.read_mat(utt_scp)
            except:
                print(utt_scp, ' has error')
                continue
            if utt in feats_scp:
                valid_utts.append(utt)
        spk_list.append(spk)
        spk2utt[spk] = valid_utts
        num += 1



# data for this repeat
feats_scp_tr = []
feats_scp_cv = []
spk2utt_tr = {}
spk2utt_cv = {}

# mangled keys for trials
spk2utt_tr_mangled = {}

print('Positive ...')
spk_id = 0
for spk in spk_list:
    if spk_id % 10000 == 0:
        print('Done', spk_id)
    spk_id += 1

    utts = spk2utt[spk]
    if len(utts) < (p + 1):
        #print 'Too few utt for spk %s: %d' % (spk, len(utts))
        continue
    len_of_cv = len(utts) - p
    if len_of_cv <= 0:
        len_of_cv = 1
    
    # shuffle and extract
    shuffle(utts)
    utts_tr = utts[len_of_cv:]
    utts_cv = utts[:len_of_cv]
    spk2utt_tr[spk] = utts_tr
    spk2utt_cv[spk] = utts_cv 
    out_line_tr = ''
    for utt_key in utts_tr:
        out_line_tr += utt_key + ' ' + feats_scp[utt_key] + ' ' 
    spk2utt_tr_mangled[spk] = out_line_tr 
    for utt_key in utts_cv:
        out_line = spk2utt_tr_mangled[spk] + utt_key + ' ' + feats_scp[utt_key] + ' 1' + '\n'
        fwrite.write(out_line)     

print('Negative ...')
# make more negative trials
#neg_sample_mode = 'legacy'
# select same number utts as positive ones from random speakers
non_target_spks = list(spk2utt_cv.keys())
num_non_target_spks = len(non_target_spks)
spk_id = 0
for spk in spk2utt_cv.keys():
    if spk_id % 10000 == 0:
        print('Done', spk_id)
    spk_id += 1

    num_neg_sample = len(spk2utt_cv[spk])
    num_neg_utts_this_spk = 0
    while True:
        other_spk_id = int(random() * num_non_target_spks)
        other_spk = non_target_spks[other_spk_id]
        if other_spk != spk:
            other_spk_utts = spk2utt_cv[other_spk]
            utt_id = int(random() * len(other_spk_utts))
            utt_key = other_spk_utts[utt_id]            
            out_line = spk2utt_tr_mangled[spk] + utt_key + ' ' + feats_scp[utt_key] + ' 0' + '\n'
            fwrite.write(out_line)
            num_neg_utts_this_spk += 1
            if num_neg_utts_this_spk >= num_neg_sample:
                break

fwrite.close()
