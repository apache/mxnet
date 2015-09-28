# Training Neural Networks on Imagenet

## Prepare Dataset

TODO

## Neural Networks

- [alexnet.py](alexnet.py) : alexnet with 5 convolution layers followed by 3
  fully connnected layers

## Results

Machine: Dual Xeon E5-2680 2.8GHz, GTX 980, Ubuntu 14.0, GCC 4.8, MKL, CUDA
7, CUDNN v3

* AlexNet


| | val accuracy | 1 x GTX 980 | 2 x GTX 980 | 4 x GTX 980 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `alexnet.py` | ? | 527 img/sec | 1030 img/sec | 1413 img/sec |
|   cxxnet    | ?|256 img/sec | 492 img/sec | 914 img/sec | 

* Inception-BN

| | val accuracy | 1 x GTX 980 | 2 x GTX 980 | 4 x GTX 980 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `inception.py` | ? | 97 img/sec (batch 32) | 178 img/sec (batch 64) | 357 img/sec (batch 128) |
|   cxxnet    | ?|57 img/sec (batch 16) | 112 img/sec (batch 32) | 224 img/sec (batch 64) |

Note: MXNet is much more memory efficiency than cxxnet, so we are able to train on larger batch.
