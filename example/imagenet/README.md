# Training Neural Networks on Imagenet

## Prepare Dataset

We are using RecordIO to pack image together. By packing images into Record IO, we can reach 3000 images/second on a normal HDD disk. This includes cost of crop from (3 x 256 x 256) to (3 x 224 x 224), random flip and other augmentation.

Please read the document of [How to Create Dataset Using RecordIO](https://mxnet.readthedocs.org/en/latest/python/io.html#create-dataset-using-recordio)

Note: A commonly mistake is forgetting shuffle the image list. This will lead fail of training, eg. ```accuracy``` keeps 0.001 for several rounds.

## Neural Networks

- [alexnet.py](alexnet.py) : alexnet with 5 convolution layers followed by 3
  fully connnected layers
- [inception.py](inception.py) : inception + batch norm network

## Results

Machine: Dual Xeon E5-2680 2.8GHz, GTX 980, Ubuntu 14.0, GCC 4.8, MKL, CUDA
7, CUDNN v3

* AlexNet

|                  | 1 x GTX 980 | 2 x GTX 980  | 4 x GTX 980  |
| ---------------- | ----------- | ------------ | ------------ |
| ```alexnet.py``` | 527 img/sec | 1030 img/sec | 1413 img/sec |
| cxxnet           | 256 img/sec | 492 img/sec  | 914 img/sec  |

For AlexNet, single model + single center test top-5 accuracy will be around 81%.


* Inception-BN

|                    | 1 x GTX 980           | 2 x GTX 980            | 4 x GTX 980             |
| ------------------ | --------------------- | ---------------------- | ----------------------- |
| ```inception.py``` | 97 img/sec (batch 32) | 178 img/sec (batch 64) | 357 img/sec (batch 128) |
| cxxnet             | 57 img/sec (batch 16) | 112 img/sec (batch 32) | 224 img/sec (batch 64)  |

For Inception-BN network, single model + single center test top-5 accuracy will be round 90%.

Note: MXNet is much more memory efficiency than cxxnet, so we are able to train on larger batch.
