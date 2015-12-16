RNN Example
===========
This folder contains RNN examples using low level symbol interface.

- [lstm.py](lstm.py) Functions for building a LSTM Network
- [lstm_ptb.py](lstm_ptb.py) PennTreeBank language model by using LSTM
- [char_lstm.ipynb](char_lstm.ipynb) Notebook to demo how to train a character LSTM by using ```lstm.py```


Performance Note:
More ```MXNET_GPU_WORKER_NTHREADS``` may lead to better performance. For setting ```MXNET_GPU_WORKER_NTHREADS```, please refer to [Environment Variables](https://mxnet.readthedocs.org/en/latest/env_var.html).
