# Speech LSTM
You can get the source code for these examples on [GitHub](https://github.com/dmlc/mxnet/tree/master/example/speech-demo).

## Speech Acoustic Modeling Example

The examples folder contains examples for speech recognition:

- [lstm_proj.py](https://github.com/dmlc/mxnet/tree/master/example/speech-demo/lstm_proj.py): Functions for building an LSTM network with and without a projection layer.
- [io_util.py](https://github.com/dmlc/mxnet/tree/master/example/speech-demo/io_util.py): Wrapper functions for `DataIter` over speech data.
- [train_lstm_proj.py](https://github.com/dmlc/mxnet/tree/master/example/speech-demo/train_lstm_proj.py): A script for training an LSTM acoustic model.
- [decode_mxnet.py](https://github.com/dmlc/mxnet/tree/master/example/speech-demo/decode_mxnet.py): A script for decoding an LSTMP acoustic model.
- [default.cfg](https://github.com/dmlc/mxnet/tree/master/example/speech-demo/default.cfg): Configuration for training on the `AMI` SDM1 dataset. You can use it as a template for writing other configuration files.
- [python_wrap](https://github.com/dmlc/mxnet/tree/master/example/speech-demo/python_wrap): C wrappers for Kaldi C++ code, built into an .so file. Python code that loads the .so file and calls the C wrapper functions in `io_func/feat_readers/reader_kaldi.py`.

Connect to Kaldi:

- [decode_mxnet.sh](https://github.com/dmlc/mxnet/tree/master/example/speech-demo/decode_mxnet.sh): Called by Kaldi to decode an acoustic model trained by MXNet (select the `simple` method for decoding).

A full receipt:

- [run_ami.sh](https://github.com/dmlc/mxnet/tree/master/example/speech-demo/run_ami.sh): A full receipt to train and decode an acoustic model on AMI. It takes features and alignment from Kaldi to train an acoustic model and decode it.

To create the speech acoustic modeling example, use the following steps.

### Build Kaldi

Build Kaldi as shared libraries if you have not already done so.

```bash
cd kaldi/src
./configure --shared # and other options that you need
make depend
make
```

### Build the Python Wrapper

1. Copy or link the attached `python_wrap` folder to `kaldi/src`.
2. Compile python_wrap/.

```
cd kaldi/src/python_wrap/
make
```

### Extract Features and Prepare Frame-level Labels

The acoustic models use Mel filter-bank or MFCC as input features. They also need to use Kaldi to perform force-alignment to generate frame-level labels from the text transcriptions. For example, if you want to work on the `AMI` data `SDM1`, you can run `kaldi/egs/ami/s5/run_sdm.sh`. Before you can run the examples, you need to configure some paths in `kaldi/egs/ami/s5/cmd.sh` and `kaldi/egs/ami/s5/run_sdm.sh`. Refer to Kaldi's documentation for details.

The default `run_sdm.sh` script generates the force-alignment labels in their stage 7, and saves the force-aligned labels in `exp/sdm1/tri3a_ali`. The default script generates MFCC features (13-dimensional). You can try training with the MFCC features, or you can create Mel filter-bank features by yourself. For example, you can use a script like this to compute Mel filter-bank features using Kaldi:

```bash
#!/bin/bash -u

. ./cmd.sh
. ./path.sh

# SDM - Single Distant Microphone
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
`apply-cmvn` provides mean-variance normalization. The default setup was applied per speaker. It's more common to perform mean-variance normalization for the whole corpus, and then feed the results to the neural networks:

```
 compute-cmvn-stats scp:data/sdm1/train_fbank/feats.scp data/sdm1/train_fbank/cmvn_g.ark
 apply-cmvn --norm-vars=true data/sdm1/train_fbank/cmvn_g.ark scp:data/sdm1/train_fbank/feats.scp ark,scp:data/sdm1/train_fbank_gcmvn/feats.ark,data/sdm1/train_fbank_gcmvn/feats.scp
```
Note that Kaldi always tries to find features in `feats.scp`. Ensure that the normalized features are organized as Kaldi expects them during decoding.

Finally, put the features and labels together in a file so that MXNet can find them. More specifically, for each data set (train, dev, eval), you will need to create a file similar to `train_mxnet.feats`, with the following contents:

```
TRANSFORM scp:feat.scp
scp:label.scp
```

`TRANSFORM` is the transformation you want to apply to the features. By default, we use `NO_FEATURE_TRANSFORM`. The `scp:` syntax is from Kaldi. `feat.scp` is typically the file from `data/sdm1/train/feats.scp`, and `label.scp` is converted from the force-aligned labels located in `exp/sdm1/tri3a_ali`. Because the force-alignments are generated only on the training data, we split the training set in two, using a 90/10 ratio, and then use the 1/10 holdout as the dev set (validation set). The script [run_ami.sh](https://github.com/dmlc/mxnet/blob/master/example/speech-demo/run_ami.sh) automatically splits and formats the file for MXNet. Before running it, set the path in the script correctly. The [run_ami.sh](https://github.com/dmlc/mxnet/blob/master/example/speech-demo/run_ami.sh) script actually runs the full pipeline, including training the acoustic model and decoding. If the scripts ran successfully, you can skip the following sections.

### Run MXNet Acoustic Model Training

1. Return to the speech demo directory in MXNet. Make a copy of `default.cfg`, and edit the necessary parameters, such as the path to the dataset you just prepared.
2. Run `python train_lstm.py --configfile=your-config.cfg`. For help, use `python train_lstm.py --help`. You can set all of the configuration parameters in `default.cfg`, the customized config file, and through the command line (e.g., using `--train_batch_size=50`). The latter values overwrite the former ones.

Here are some example outputs from training on the TIMIT dataset:

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

The final frame accuracy was approximately 62%.

### Run Decode on the Trained Acoustic Model

1. Estimate senone priors by running `python make_stats.py --configfile=your-config.cfg | copy-feats ark:- ark:label_mean.ark` (edit necessary items, such as the path to the training dataset). This command generates the label counts in `label_mean.ark`.
2. Link to the necessary Kaldi decode setup, e.g., `local/` and `utils/` and run `./run_ami.sh --model prefix model --num_epoch num`.

Here are the results for the TIMIT and AMI test sets (using the default setup, three-layer LSTM with projection layers):

	| Corpus | WER |
	|--------|-----|
	|TIMIT   | 18.9|
	|AMI     | 51.7 (42.2) |

For AMI 42.2 was evaluated non-overlapped speech. The Kaldi-HMM baseline was 67.2%, and DNN was 57.5%.

## Next Steps
* [MXNet tutorials index](http://mxnet.io/tutorials/index.html)
