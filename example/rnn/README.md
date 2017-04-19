RNN Example
===========
This folder contains RNN examples using high level mxnet.rnn interface.

Examples using low level symbol interface have been deprecated and moved to old/

## Data
Run `get_ptb_data.sh` to download PenTreeBank data.

## Python

- [lstm_bucketing.py](lstm_bucketing.py) Example code of how to use the bucket
- [cudnn_lstm_bucketing.py](cudnn_lstm_bucketing.py) Example code of how to use the bucket (Using CUDNN accelerated LSTM)
- [lstm_ptb.py](lstm_ptb.py) Example of replicating the language modelling experiment on the PTB dataset
Performance Note:
More ```MXNET_GPU_WORKER_NTHREADS``` may lead to better performance. For setting ```MXNET_GPU_WORKER_NTHREADS```, please refer to [Environment Variables](https://mxnet.readthedocs.org/en/latest/how_to/env_var.html).

## Language Modeling

To replicate LM result of PTB described in (Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329, run the following command

```bash
# Small Configuration
python lstm_ptb.py --num-hidden 200 \
                   --num-embed 200 \
                   --init-scale 0.1 \
                   --lr-decay 0.5 \
                   --dropout 0.5 \
                   --max-norm 5 \
                   --gpus 0
# Medium Configuration
python lstm_ptb.py --num-hidden 650 \
                   --num-embed 650 \
                   --init-scale 0.05 \
                   --lr-decay 0.8 \
                   --dropout 0.5 \
                   --max-norm 5 \
                   --gpus 0
# Large Configuration
python lstm_ptb.py --num-hidden 1500 \
                   --num-embed 1500 \
                   --init-scale 0.04 \
                   --lr-decay 0.87 \
                   --dropout 0.65 \
                   --max-norm 10 \
                   --gpus 0
```