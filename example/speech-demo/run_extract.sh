#!/bin/bash -u

. ./cmd.sh
. ./path.sh

# SDM - Signle Distant Microphone
micid=1 #which mic from array should be used?
mic=sdm$micid

# Set bash to 'debug' mode, it prints the commands (option '-x') and exits on :
# -e 'error', -u 'undefined variable', -o pipefail 'error in pipeline',
set -euxo pipefail

# Path where AMI gets downloaded (or where locally available):
AMI_DIR=$PWD/wav_db # Default,
data_dir=$PWD/data/$mic

# make filter bank data
for dset in train dev eval; do
  steps/make_fbank.sh --nj 48 --cmd "$train_cmd" $data_dir/$dset \
    $data_dir/$dset/log $data_dir/$dset/data-fbank
  steps/compute_cmvn_stats.sh $data_dir/$dset \
    $data_dir/$dset/log $data_dir/$dset/data

  apply-cmvn --utt2spk=ark:$data_dir/$dset/utt2spk \
    scp:$data_dir/$dset/cmvn.scp scp:$data_dir/$dset/feats.scp \
    ark,scp:$data_dir/$dset/feats-cmvn.ark,$data_dir/$dset/feats-cmvn.scp

  mv $data_dir/$dset/feats-cmvn.scp $data_dir/$dset/feats.scp
done
