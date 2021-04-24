#!/bin/bash

#datalst=$1
#datadir=$2
#stage=$3
CUDA_DEVICES=1
stage=3
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
  #python gen_train_dev_test.py ${datadir}/noise/all/interference/interference.scp 99.0:0.5:0.5 ${datadir}/noise/interference/ || exit 1;

  python gen_train_dev_test.py ${datadir}/noise/all/diffuse_noise/diffuse_noise.scp 90.0:5.0:5.0 ${datadir}/noise/diffuse_noise/ || exit 1;
fi

# Prepare noisy speech
if [ $stage -le 3 ]; then
  echo "step-3: prepare train noisy speech"
  outname=5mic_noisy
  num_null=0
  num_mic=5
  mic_radius=0.055
  targ_bf=/exdata/HOME/snie/DeepNoiseTrackingNetwork/train_enhance/egs/5mic/exp/bf_targ_bf_5mic_5.5cm.mat

  train_num=40000
  data=train
  num_works=4

  start=5
  end=$(expr $start + $num_works)
  for i in $(seq $start $end)
  do
  {
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python make_data.py --dataroot ${datadir}/speech/$data --diffuse_scp ${datadir}/noise/5mic_diffuse_noise/$data/diffuse_noise.scp --interference_scp ${datadir}/noise/interference/$data/interference.scp --out_path ${datadir}/${outname}/$data/$i --num_utterances $train_num --targ_bf $targ_bf --num_null $num_null --num_mic $num_mic --mic_radius $mic_radius
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


:<<BLOCK

  dev_num=5000
  data=dev
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python make_data.py --dataroot ${datadir}/speech/$data --diffuse_scp ${datadir}/noise/5mic_diffuse_noise/$data/diffuse_noise.scp --interference_scp ${datadir}/noise/interference/$data/interference.scp --out_path ${datadir}/${outname}/$data --num_utterances $dev_num --targ_bf $targ_bf --num_null $num_null --num_mic $num_mic --mic_radius $mic_radius
  
  python genclean2mixture.py ${datadir}/${outname}/$data/utt2info ${datadir}/${outname}/$data/


  test_num=5000
  data=test
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python make_data.py --dataroot ${datadir}/speech/$data --diffuse_scp ${datadir}/noise/5mic_diffuse_noise/$data/diffuse_noise.scp --interference_scp ${datadir}/noise/interference/$data/interference.scp --out_path ${datadir}/${outname}/$data --num_utterances $test_num --targ_bf $targ_bf --num_null $num_null --num_mic $num_mic --mic_radius $mic_radius

  python genclean2mixture.py ${datadir}/${outname}/$data/utt2info ${datadir}/${outname}/$data/

BLOCK

fi
exit 0;
