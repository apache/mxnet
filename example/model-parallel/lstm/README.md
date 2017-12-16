Model Parallel LSTM
===================

This is an example showing how to do model parallel LSTM in MXNet.

We use [the PenTreeBank dataset](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/ptb/)
in this example. Download the dataset with below command:

`bash get_ptb_data.sh`

This will download PenTreeBank dataset under `data` folder. Now, you can run the training as follows:

`python lstm_ptb.py`
