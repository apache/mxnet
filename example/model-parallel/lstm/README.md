Model Parallel LSTM
===================

This is an example showing how to do model parallel LSTM in MXNet.

To run this example, first make sure you download a dataset of PenTreeBank available
[here](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/ptb/) by running following command:

`bash get_ptb_data.sh`

This will download PenTreeBank dataset under `data` folder. Now, you can run the training as follows:

`python lstm_ptb.py`
