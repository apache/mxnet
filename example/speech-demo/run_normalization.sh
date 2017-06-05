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
  compute-cmvn-stats scp:$data_dir/$dset/feats.scp $data_dir/$dset/cmvn_g.ark
  if [ ! -d "${data_dir}/${dset}/fbank_gcmvn" ]; then
    mkdir $data_dir/$dset/fbank_gcmvn
  fi
  cp $data_dir/$dset/utt2spk $data_dir/$dset/fbank_gcmvn/utt2spk
  apply-cmvn --norm-vars=true $data_dir/$dset/cmvn_g.ark scp:$data_dir/$dset/feats.scp ark,scp:$data_dir/$dset/fbank_gcmvn/feats.ark,$data_dir/$dset/fbank_gcmvn/feats.scp
done
