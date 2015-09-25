# Training Neural Networks on Imagenet

## Prepare Dataset

TODO

## Neural Networks

- [alexnet.py](alexnet.py) : alexnet with 5 convolution layers followed by 3
  fully connnected layers

## Results

Machine: Dual Xeon E5-2680 2.8GHz, GTX 980, Ubuntu 14.0, GCC 4.8, MKL, CUDA
7, CUDNN v3

| | val accuracy | 1 x GTX 980 | 2 x GTX 980 | 4 x GTX 980 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `alexnet.py` | ? | ? | 1020 img/sec | |
|   cxxnet    | ?|256 img/sec | 492 img/sec | 914 img/sec | 
