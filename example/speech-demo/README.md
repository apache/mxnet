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

Example output for TIMIT:
Summary of dataset ==================
bucket of len 100 : 3 samples
bucket of len 200 : 346 samples
bucket of len 300 : 1496 samples
bucket of len 400 : 974 samples
bucket of len 500 : 420 samples
Summary of dataset ==================
bucket of len 100 : 0 samples
bucket of len 200 : 28 samples
bucket of len 300 : 169 samples
bucket of len 400 : 107 samples
bucket of len 500 : 41 samples
2016-04-18 17:32:52,060 Start training with [gpu(3)]
2016-04-18 17:33:50,466 Epoch[0] Resetting Data Iterator
2016-04-18 17:33:50,466 Epoch[0] Train-Accuracy=0.262495
2016-04-18 17:33:50,466 Epoch[0] Time cost=57.966
2016-04-18 17:33:56,911 Epoch[0] Validation-Accuracy=0.389623
2016-04-18 17:34:52,226 Epoch[1] Resetting Data Iterator
2016-04-18 17:34:52,226 Epoch[1] Train-Accuracy=0.460667
2016-04-18 17:34:52,226 Epoch[1] Time cost=55.315
2016-04-18 17:34:58,703 Epoch[1] Validation-Accuracy=0.466534
