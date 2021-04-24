#!/bin/bash
works_dir=$(pwd)
feats_scp=/data/HOME/bliu/workspace/speaker/data/vox/test/feats.scp_100
expdir=exp/xvector/
stage=1
nj=4

loss_type="class_softmax"
margin_type="Softmax"
att_type="multi_attention"
model_type="cnn_Res50_IR"
segment_type="all"

segment_shift_rate=0.50
min_segment_length=180
max_segment_length=140

resume="exp/speaker_cnn_Res50_IR-class_softmax-multi_attention-Softmax/newest_model.pth"

. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh


if [ $stage -le 3 ]; then
  files=$(for n in `seq $nj`; do echo ${expdir}/split${nj}/$n/feats.scp; done)
  directories=$(for n in `seq $nj`; do echo ${expdir}/split${nj}/$n; done)
  for n in `seq $nj`; do
    mkdir -p ${expdir}/split${nj}/$n
  done
  utils/split_scp.pl ${feats_scp} $files   
  for n in `seq $nj`; do
    cat ${expdir}/split${nj}/$n/feats.scp | awk '{print $1 " " $1}' > ${expdir}/split${nj}/$n/utt2spk
  done
fi


if [ $stage -le 4 ]; then
  mkdir -p ${expdir}/log
  # Multiple GPU
  $train_cmd JOB=1:$nj $expdir/log/extract.JOB.log \
    CUDA_VISIBLE_DEVICES=2 python3 extract_vector.py --cuda --works_dir ${works_dir} --exp_path ${expdir}/ --dataroot ${expdir}/split${nj}/JOB/ --thread_num JOB --loss_type ${loss_type} --margin_type ${margin_type} --att_type $att_type --model_type $model_type --segment_type $segment_type --resume ${resume} --segment_shift_rate $segment_shift_rate --min_segment_length $min_segment_length --max_segment_length $max_segment_length || exit 1;      
    
   cat $expdir/xvector/xvector*.scp > $expdir/vectors.scp
fi

exit 0;

