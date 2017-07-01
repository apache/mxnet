# Recurrent Neural Networks
This folder contains RNN examples using a low-level symbol interface. You can get the source code for this example on [GitHub](https://github.com/dmlc/mxnet/tree/master/example/rnn).

## Python

- [https://github.com/dmlc/mxnet/blob/master/example/rnn/lstm_bucketing.py](lstm_bucketing.py). A PennTreeBank language model using LSTM
- [https://github.com/dmlc/mxnet/blob/master/example/rnn/cudnn_lstm_bucketing.py](cudnn_lstm_bucketing.py). A PennTreeBank language model using LSTM and CUDNN

Performance Note:

Using more ```MXNET_GPU_WORKER_NTHREADS``` may lead to better performance. For information on setting ```MXNET_GPU_WORKER_NTHREADS```, refer to [Environment Variables](http://mxnet.io/how_to/env_var.html).

## Next Steps
* [MXNet tutorials index](http://mxnet.io/tutorials/index.html)
