# End to End Neural Art

Please refer to this [blog](http://dmlc.ml/mxnet/2016/06/20/end-to-end-neural-style.html) for details of how it is implemented.

## How to use


1. First use `../download.sh` to download pre-trained model and sample inputs.

2. Prepare training dataset. Put image samples to `../data/` (one file for each image sample). The pretrained model here was trained by 26k images sampled from [MIT Place dataset](http://places.csail.mit.edu).

3. Use `boost_train.py` for training.

## Pretrained Model

- Model: [https://github.com/dmlc/web-data/raw/master/mxnet/art/model.zip](https://github.com/dmlc/web-data/raw/master/mxnet/art/model.zip)
- Inference script: `boost_inference.py`
