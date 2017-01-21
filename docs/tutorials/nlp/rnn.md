# Recurrent Neural Networks
This folder contains RNN examples using a low-level symbol interface. You can get the source code for this example on [GitHub](https://github.com/dmlc/mxnet/tree/master/example/rnn).

## Python

- [lstm.py](lstm.py). Functions for building an LSTM Network
- [gru.py](gru.py). Functions for building a GRU Network
- [lstm_bucketing.py](lstm_bucketing.py). A PennTreeBank language model using LSTM
- [gru_bucketing.py](gru_bucketing.py). A PennTreeBank language model using GRU
- [char-rnn.ipynb](char-rnn.ipynb). A notebook that demonstrates how to train a character LSTM by using ```lstm.py```


Performance Note:

Using more ```MXNET_GPU_WORKER_NTHREADS``` may lead to better performance. For information on setting ```MXNET_GPU_WORKER_NTHREADS```, refer to [Environment Variables](http://mxnet.io/how_to/env_var.html).

## Next Steps
* [MXNet tutorials index](http://mxnet.io/tutorials/index.html)
