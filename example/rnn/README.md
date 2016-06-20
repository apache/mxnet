RNN Example
===========
This folder contains RNN examples using low level symbol interface.

## Python

- [lstm.py](lstm.py) Functions for building a LSTM Network
- [gru.py](gru.py) Functions for building a GRU Network
- [lstm_bucketing.py](lstm_bucketing.py) PennTreeBank language model by using LSTM
- [gru_bucketing.py](gru_bucketing.py) PennTreeBank language model by using GRU
- [char-rnn.ipynb](char-rnn.ipynb) Notebook to demo how to train a character LSTM by using ```lstm.py```


Performance Note:
More ```MXNET_GPU_WORKER_NTHREADS``` may lead to better performance. For setting ```MXNET_GPU_WORKER_NTHREADS```, please refer to [Environment Variables](https://mxnet.readthedocs.org/en/latest/how_to/env_var.html).
