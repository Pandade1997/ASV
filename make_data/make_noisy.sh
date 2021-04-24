#!/bin/bash

. ./cmd.sh
. ./path.sh


#datalst=$1
#datadir=$2
#stage=$3

out_dir=$1
speech_dir=$2
noise_dir=$3
music_dir=$4
rmr_dir=$5
train=$6
dev=$7
test=$8
stage=$9

tools=make_data

#if [ $stage -le 0 ]; then
  
  data=train
  python ${tools}/make_reverb_noisy_data.py --dataroot $speech_dir/$data --noise_scp $noise_dir/$data/noise.scp --music_scp $music_dir/$data/music_wind.scp --rmr_scp $rmr_dir/rir.scp --out_path $out_dir/$data --num_utterances $train
  
  data=dev
  python ${tools}/make_reverb_noisy_data.py --dataroot $speech_dir/$data --noise_scp $noise_dir/$data/noise.scp --music_scp $music_dir/$data/music_wind.scp --rmr_scp $rmr_dir/rir.scp --out_path $out_dir/$data --num_utterances $dev
    
  data=test
  python ${tools}/make_reverb_noisy_data.py --dataroot $speech_dir/$data --noise_scp $noise_dir/$data/noise.scp --music_scp $music_dir/$data/music_wind.scp --rmr_scp $rmr_dir/rir.scp --out_path $out_dir/$data --num_utterances $test
  
#fi

#if [ $stage -le 1 ]; then 
  for data in train dev test; do
    python3 ${tools}/genclean2mixture.py $out_dir/$data/utt2info $out_dir/$data
  done
#fi

exit 0;
