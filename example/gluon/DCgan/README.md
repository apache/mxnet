# DCgan in MXNet

train the dcgan with mxnet, and eval dcgan model with inception_score.

DCgan model is from: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

The inception score is refer to: [openai/improved-gan
](https://github.com/openai/improved-gan)


#### Generated pic(use dataset cifar10)
![Generated pic](https://github.com/pengxin99/incubator-mxnet/blob/dcgan-inception_score/example/gluon/DCgan/pic/fake_img_iter_13900.png)

#### Generated pic(use dataset mnist)
![Generated pic](https://github.com/pengxin99/incubator-mxnet/blob/dcgan-inception_score/example/gluon/DCgan/pic/fake_img_iter_21700.png)

#### inception_score in cpu and gpu (the real image`s score is around 3.3)
CPU & GPU

![inception_socre_with_cpu](https://github.com/pengxin99/incubator-mxnet/blob/dcgan-inception_score/example/gluon/DCgan/pic/inception_score_cifar10_cpu.png)
![inception_score_with_gpu](https://github.com/pengxin99/incubator-mxnet/blob/dcgan-inception_score/example/gluon/DCgan/pic/inception_score_cifar10.png)
## Quick start
use below code to see the configurations you can set:
```python
python dcgan.py -h
```

use below code to train dcgan model with default configurations and dataset(cifar10), and metric with inception_score:
```python
python dcgan.py
```
