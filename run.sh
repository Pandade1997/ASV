#!/bin/bash

datadir=/exdata/HOME/snie/code/v2_speaker_asvspoof/dataset_min/speaker/data/
expdir=exp
stage=2

loss_type="class_softmax"
margin_type="Softmax"
att_type="multi_attention"
model_type="cnn_Res20_IR"
segment_type="all"
speaker_num=32
utter_num=8
resume="none"

delta_order=0
rnn_hidden_size=256
embedding_size=512
segment_shift_rate=0.50
min_segment_length=240
max_segment_length=280
min_num_segment=5
max_num_segment=15
normalize_type=0
batch_size=128

lr=0.002
optim_type="adam"
dist_url="tcp://127.0.0.1:15502"

. ./utils/parse_options.sh

if [ $stage -le 2 ]; then
  # Multiple GPU
  export NGPUS=2
  CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train_speaker.py --dist-url ${dist_url} --delta_order ${delta_order} --agument_feat --normalize_type ${normalize_type} --batch_size ${batch_size} --rnn_hidden_size ${rnn_hidden_size} --embedding_size ${embedding_size} --cuda --batch_size ${batch_size} --dataroot $datadir --loss_type ${loss_type} --margin_type ${margin_type} --att_type $att_type --model_type $model_type --segment_type $segment_type --speaker_num $speaker_num --utter_num ${utter_num} --resume ${resume} --lr ${lr} --optim_type ${optim_type} --segment_shift_rate $segment_shift_rate --min_segment_length $min_segment_length --max_segment_length $max_segment_length --min_num_segment $min_num_segment --max_num_segment $max_num_segment
  
  #speaker_seq_"$seq_training"_"$model_type"_"$train_type"_"$segment_type"
  # Single GPU
  #CUDA_VISIBLE_DEVICES=3 python3 local/train_speaker.py --cuda --dataroot $datadir --seq_training $seq_training --train_type $train_type --model_type $model_type --segment_type $segment_type --speaker_num $speaker_num --model_name speaker_DSAE_LSTM_MultiAttention_fromRandomInit_All --lr 0.0001 --segment_shift_rate $segment_shift_rate --min_segment_length $min_segment_length --max_segment_length $max_segment_length --min_num_segment $min_num_segment --max_num_segment $max_num_segment --cmvn_file cmvn.npy --resume /home1/cyr/nworks/speaker/train_speaker/exp/speaker_DSAE_LSTM_MultiAttention_fromRandomInit_All/model_best.pth --vad
fi

exit 0;

#--dist-url "tcp://127.0.0.1:15502" --delta_order 0 --agument_feat --normalize_type 0 --batch_size 128 --rnn_hidden_size 256 --embedding_size 512 --cuda --batch_size 128 --dataroot /exdata/HOME/snie/code/v2_speaker_asvspoof/dataset_min/speaker/data/ --loss_type "class_softmax" --margin_type "Softmax" --att_type ”multi_attention“ --model_type "cnn_Res20_IR" --segment_type "all" --speaker_num 32 --utter_num 8 --resume "none" --lr 0.002 --optim_type "adam" --segment_shift_rate 0.50 --min_segment_length 240 --max_segment_length 280 --min_num_segment 5 --max_num_segment 15
