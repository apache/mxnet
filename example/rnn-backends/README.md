# rnn-backends

This directory provides comparison between different LSTM RNN backend implementations.
To run the examples, please download the dataset using the provided script in the `dataset` directory.
Please also make sure you have 
[`mxboard`](https://github.com/awslabs/mxboard) and 
[`tensorboard`](https://github.com/tensorflow/tensorboard) installed 
(`sudo -H pip install mxboard tensorboard`), as all benchmarks use `tensorboard` to keep track of throughput.
