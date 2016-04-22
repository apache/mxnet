RNN Example
===========
This folder contains examples for speech recognition.

- [lstm.py](lstm.py) Functions for building a LSTM Network
- [lstm_bucketing.py](lstm_bucketing.py) Acoustic Model by using LSTM
- [python_wrap] C wrappers for Kaldi C++ code, this is built into a .so. Python code that loads the .so and calls the C wrapper functions in io_func/feat_readers/reader_kaldi.py

DATASET:

DATASETS["TIMIT_train"] = {
        "lst_file": "train_mxnet.feats",
        "format": "kaldi",
        "in": 40
        }


train_mxnet.feats:

TRANSFORM scp:feat.scp
scp:label.scp

To run the code:
1. Build Kaldi as ``shared libraties'' if you have not already done so.
```
cd kaldi/src
make depend
make
```
2. Extract the attached python_wrap/ folder to kaldi/src.
3. Compile python_wrap/
```
cd kaldi/src/python_wrap/
make
```
4. Run the example
```
export LD_LIBRARY_PATH=../../lib:$LD_LIBRARY_PATH
```
5. To build it on the TIMIT corpus, you can first run the script for TIMIT on Kaldi: kaldi-trunk/egs/timit/s5/run.sh and then 
```
python lstm_bucketing.py TIMIT
```

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
