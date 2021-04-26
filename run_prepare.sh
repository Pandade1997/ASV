aishell_data=/exdata/HOME/snie/code/v2_speaker_asvspoof/dataset_min/speaker/speech

## make dataset
featdir=/exdata/HOME/snie/code/v2_speaker_asvspoof/dataset_min/speaker/fbank
datadir=/exdata/HOME/snie/code/v2_speaker_asvspoof/dataset_min/speaker/data
for x in train dev test; do
    python3 aishell_data_prep.py  $aishell_data/$x $datadir/$x $featdir/$x || exit 1;
done
## make test_pair for verification
for x in dev test; do
    python3 make_pairs.py $datadir/$x 15 15 || exit 1;
done