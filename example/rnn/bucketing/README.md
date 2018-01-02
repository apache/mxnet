RNN Example
===========
This folder contains RNN examples using high level mxnet.rnn interface.

## Data
Run `get_ptb_data.sh` to download PenTreeBank data.

## Python

- [lstm_bucketing.py](lstm_bucketing.py) PennTreeBank language model by using LSTM

Performance Note:
More ```MXNET_GPU_WORKER_NTHREADS``` may lead to better performance. For setting ```MXNET_GPU_WORKER_NTHREADS```, please refer to [Environment Variables](https://mxnet.readthedocs.org/en/latest/how_to/env_var.html).
