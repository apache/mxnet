Speech Acoustic Modeling Example
================================
This folder contains examples for speech recognition.

- [lstm.py](lstm.py): Functions for building a LSTM Network.
- [io_util.py](io_util.py): Wrapper functions for `DataIter` over speech data.
- [train_lstm.py](train_lstm.py): Script for training LSTM acoustic model.
- [ami.cfg](ami.cfg): Configuration for training on the `AMI` SDM1 dataset. Can be used as a template for writing other configuration files.
- [python_wrap](python_wrap): C wrappers for Kaldi C++ code, this is built into a .so. Python code that loads the .so and calls the C wrapper functions in `io_func/feat_readers/reader_kaldi.py`.

To reproduce the results, use the following steps.

### Build Kaldi

Build Kaldi as **shared libraties** if you have not already done so.

```bash
cd kaldi/src
./configure --shared # and other options that you need
make depend
make
```

### Build Python Wrapper

1. Copy or link the attached `python_wrap` folder to `kaldi/src`.
2. Compile python_wrap/

```
cd kaldi/src/python_wrap/
make
```

### Extract Features and Prepare Frame-level Labels

The acoustic models use *Mel filter-bank* or *MFCC* as input features. It also need to use Kaldi to do force-alignment to generate frame-level labels from the text transcriptions. For example, if you want to work on the `AMI` data `SDM1`. You can run `kaldi/egs/ami/s5/run_sdm.sh`. You will need to do some configuration of paths in `kaldi/egs/ami/s5/cmd.sh` and `kaldi/egs/ami/s5/run_sdm.sh` before you can run the examples. Please refer to Kaldi's document for more details.

The default `run_sdm.sh` script generates the force-alignment labels in their stage 7, and saves the force-aligned labels in `exp/sdm1/tri3a_ali`. The default script generates MFCC features (13-dimensional). You can try training with the MFCC features, or you can create Mel filter bank features by your self. For example, a script like this can be used to compute Mel filter bank features using Kaldi.

```bash
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
```

Finally, you need to put the features and labels together in a file so that MXNet can find them. More specifically, for each data set (train, dev, eval), you will need to create a file like `train_mxnet.feats`, will the following contents:

```
TRANSFORM scp:feat.scp
scp:label.scp
```

Here the `TRANSFORM` is the transformation you want to apply to the features. By default we use `NO_FEATURE_TRANSFORM`. The `scp:` syntax is from Kaldi. The `feat.scp` is typically the file from `data/sdm1/train/feats.scp`, and the `label.scp` is converted from the force-aligned labels located in `exp/sdm1/tri3a_ali`. We use a script like below to prepare the feature files. Because the force-alignments are only generated on the training data, we simply split the training set into 90/10 parts, and use the 1/10 hold-out as the dev set (validation set).

```bash
#!/bin/bash

# SDM - Signle Distant Microphone
micid=1 #which mic from array should be used?
mic=sdm$micid

# split the data : 90% train 10% cross-validation (held-out),
dir=$PWD/data/$mic/train
[ ! -e ${dir}_tr90 ] && utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10

# prepare listing data
dir=$PWD/exp/$mic/data-for-mxnet
mkdir -p $dir

# make post.scp, post.ark
ali-to-pdf exp/$mic/tri3a_ali/final.mdl "ark:gunzip -c exp/$mic/tri3a_ali/ali.*.gz |" \
  ark:- | ali-to-post ark:- ark,scp:$dir/post.ark,$dir/post.scp

# generate dataset list
echo NO_FEATURE_TRANSFORM scp:$PWD/data/$mic/train_tr90/feats.scp > $dir/train.feats
echo scp:$dir/post.scp >> $dir/train.feats

echo NO_FEATURE_TRANSFORM scp:$PWD/data/$mic/train_cv10/feats.scp > $dir/dev.feats
echo scp:$dir/post.scp >> $dir/dev.feats
```

### Run MXNet Acoustic Model Training

1. Go back to this speech demo directory in MXNet. Make a copy of your own `ami.cfg` and edit necessary items like the path to the dataset you just prepared.
2. Run `python train_lstm.py ami.cfg`.

Here are some example outputs that we got from training on the TIMIT dataset.

```
Example output for TIMIT:
Summary of dataset ==================
bucket of len 100 : 3 samples
bucket of len 200 : 346 samples
bucket of len 300 : 1496 samples
bucket of len 400 : 974 samples
bucket of len 500 : 420 samples
bucket of len 600 : 90 samples
bucket of len 700 : 11 samples
bucket of len 800 : 2 samples
Summary of dataset ==================
bucket of len 100 : 0 samples
bucket of len 200 : 28 samples
bucket of len 300 : 169 samples
bucket of len 400 : 107 samples
bucket of len 500 : 41 samples
bucket of len 600 : 6 samples
bucket of len 700 : 3 samples
bucket of len 800 : 0 samples
2016-04-21 20:02:40,904 Epoch[0] Train-Acc_exlude_padding=0.154763
2016-04-21 20:02:40,904 Epoch[0] Time cost=91.574
2016-04-21 20:02:44,419 Epoch[0] Validation-Acc_exlude_padding=0.353552
2016-04-21 20:04:17,290 Epoch[1] Train-Acc_exlude_padding=0.447318
2016-04-21 20:04:17,290 Epoch[1] Time cost=92.870
2016-04-21 20:04:20,738 Epoch[1] Validation-Acc_exlude_padding=0.506458
2016-04-21 20:05:53,127 Epoch[2] Train-Acc_exlude_padding=0.557543
2016-04-21 20:05:53,128 Epoch[2] Time cost=92.390
2016-04-21 20:05:56,568 Epoch[2] Validation-Acc_exlude_padding=0.548100
```

The final frame accuracy was around 62%.
