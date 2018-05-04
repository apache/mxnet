RNN Example
===========
This folder contains RNN examples using high level mxnet.rnn interface.

## Data
1) Review the license for the PenTreeBank dataset and ensure that you agree to it. Then uncomment the lines in the 'get_ptb_data.sh' script that download the dataset.

2) Run `get_ptb_data.sh` to download PenTreeBank data.

## Python

- Generate the PennTreeBank language model by using LSTM:

  For Python2 (CPU support): can take 2+ hours on AWS-EC2-p2.16xlarge

      $ python  [lstm_bucketing.py](lstm_bucketing.py) 

  For Python3 (CPU support): can take 2+ hours on AWS-EC2-p2.16xlarge

      $ python3 [lstm_bucketing.py](lstm_bucketing.py) 

  Assuming your machine has 4 GPUs and you want to use all the 4 GPUs:

  For Python2 (GPU support only): can take 50+ minutes on AWS-EC2-p2.16xlarge

      $ python  --gpus 0,1,2,3 [cudnn_lstm_bucketing.py](cudnn_lstm_bucketing.py) 

  For Python3 (GPU support only): can take 50+ minutes on AWS-EC2-p2.16xlarge

      $ python3 --gpus 0,1,2,3 [cudnn_lstm_bucketing.py](cudnn_lstm_bucketing.py) 


### Performance Note:

More ```MXNET_GPU_WORKER_NTHREADS``` may lead to better performance. For setting ```MXNET_GPU_WORKER_NTHREADS```, please refer to [Environment Variables](http://mxnet.incubator.apache.org/faq/env_var.html).

