#!/bin/bash

#datalst=$1
#datadir=$2
#stage=$3
CUDA_DEVICES=2
stage=0
datalst=/exdata/HOME/snie/DeepNoiseTrackingNetwork/data/speech/speech.lst
datadir=/exdata/HOME/snie/DeepNoiseTrackingNetwork/data/

words_vocab_file=word_seg_vocab.txt
# Prepare wav.scp, text, utt2spk, spk2utt, utt2data and data2utt from a dataset_lst
if [ $stage -le 0 ]; then
  echo "step-1: Prepare wav.scp, text, utt2spk, spk2utt, utt2data and data2utt from a dataset_lst"
  python selectASRData.py $datalst $words_vocab_file $datadir/speech/all || exit 1;
fi

# Prepare train, dev and test for speech
if [ $stage -le 1 ]; then 
  echo "step-2: prepare train, dev and test for speech"
  python genASR_train_dev_test_new.py $datadir/speech/all 99.0:0.50:0.50 $datadir/speech/ || exit 1;
fi

# Prepare train, dev and test for noise
if [ $stage -le 2 ]; then 
  echo "step-3: prepare train, dev and test for noise"
  python gen_train_dev_test.py ${datadir}/noise/all/interference/interference.scp 99.0:0.5:0.5 ${datadir}/noise/interference/ || exit 1;

  python gen_train_dev_test.py ${datadir}/noise/all/diffuse_noise/diffuse_noise.scp 90.0:5.0:5.0 ${datadir}/noise/diffuse_noise/ || exit 1;
fi

# Prepare noisy speech 
:<<BLOCK
BLOCK

mic_pos="(0.0200,0.000000)(-0.0200,0.000000)"
outname=2mic_noisy
num_null=0
num_mic=2
targ_bf=/exdata/HOME/snie/DeepNoiseTrackingNetwork/train_enhance_cfeat/egs/2mic/exp/bf_targ_bf_2mic_4.0cm.mat


if [ $stage -le 3 ]; then
  echo "step-3: prepare train noisy speech"
  train_num=300000
  data=train
  num_works=1

  start=0
  end=$(expr $start + $num_works)
  for i in $(seq $start $end)
  do
  {
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python make_data_new.py --dataroot ${datadir}/speech/$data --diffuse_scp ${datadir}/noise/diffuse_noise/$data/diffuse_noise.scp --interference_scp ${datadir}/noise/interference/$data/interference.scp --out_path ${datadir}/${outname}/$data/$i --num_utterances $train_num --targ_bf $targ_bf --num_null $num_null --num_mic $num_mic --is_linear_mic True --min_targ_distance 0.5 --max_targ_distance 5.5 --min_interf_distance 1.0 --max_interf_distance 7.0 --max_num_interf 3 --lowSIR -5.0 --upSIR 15.0 --mic_pos $mic_pos
  } &
  done

  wait

  for i in $(seq $start $end)
  do
    cat ${datadir}/${outname}/$data/$i/wav.scp >> ${datadir}/${outname}/$data/wav.scp
    cat ${datadir}/${outname}/$data/$i/mix.scp >> ${datadir}/${outname}/$data/mix.scp
    cat ${datadir}/${outname}/$data/$i/text >> ${datadir}/${outname}/$data/text
    cat ${datadir}/${outname}/$data/$i/utt2spk >> ${datadir}/${outname}/$data/utt2spk
    cat ${datadir}/${outname}/$data/$i/utt2info >> ${datadir}/${outname}/$data/utt2info
    cat ${datadir}/${outname}/$data/$i/spk2utt >> ${datadir}/${outname}/$data/spk2utt
  done

  python genclean2mixture.py ${datadir}/${outname}/$data/utt2info ${datadir}/${outname}/$data/
fi

if [ $stage -le 4 ]; then
  echo "step-4: prepare dev noisy speech"
  dev_num=5000
  data=dev
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python make_data_new.py --dataroot ${datadir}/speech/$data --diffuse_scp ${datadir}/noise/diffuse_noise/$data/diffuse_noise.scp --interference_scp ${datadir}/noise/interference/$data/interference.scp --out_path ${datadir}/${outname}/$data --num_utterances $dev_num --targ_bf $targ_bf --num_null $num_null --num_mic $num_mic --is_linear_mic True --min_targ_distance 0.5 --max_targ_distance 5.5 --min_interf_distance 1.0 --max_interf_distance 7.0 --max_num_interf 3 --lowSIR -5.0 --upSIR 15.0 --mic_pos $mic_pos
  
  python genclean2mixture.py ${datadir}/${outname}/$data/utt2info ${datadir}/${outname}/$data/
fi

if [ $stage -le 5 ]; then
  echo "step-5: prepare test noisy speech"
  test_num=500
  data=test
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python make_data_new.py --dataroot ${datadir}/speech/$data --diffuse_scp ${datadir}/noise/diffuse_noise/$data/diffuse_noise.scp --interference_scp ${datadir}/noise/interference/$data/interference.scp --out_path ${datadir}/${outname}/$data --num_utterances $test_num --targ_bf $targ_bf --num_null $num_null --num_mic $num_mic --is_linear_mic True --min_targ_distance 0.5 --max_targ_distance 5.5 --min_interf_distance 1.0 --max_interf_distance 7.0 --max_num_interf 3 --lowSIR -5.0 --upSIR 15.0 --mic_pos $mic_pos

  python genclean2mixture.py ${datadir}/${outname}/$data/utt2info ${datadir}/${outname}/$data/
fi

exit 0;

