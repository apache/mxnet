# DCGAN in MXNet

[Deep Convolutional Generative Adversarial Networks(DCGAN)](https://arxiv.org/abs/1511.06434) implementation with Apache MXNet GLUON.
This implementation uses [inception_score](https://github.com/openai/improved-gan) to evaluate the model.

You can use this reference implementation on the MNIST and CIFAR-10 datasets.


#### Generated pic(use dataset cifar10)
![Generated pic](https://github.com/pengxin99/incubator-mxnet/blob/dcgan-inception_score/example/gluon/DCGAN/pic/fake_img_iter_13900.png)

#### Generated pic(use dataset mnist)
![Generated pic](https://github.com/pengxin99/incubator-mxnet/blob/dcgan-inception_score/example/gluon/DCGAN/pic/fake_img_iter_21700.png)

#### inception_score in cpu and gpu (the real image`s score is around 3.3)
CPU & GPU

![inception_socre_with_cpu](https://github.com/pengxin99/incubator-mxnet/blob/dcgan-inception_score/example/gluon/DCGAN/pic/inception_score_cifar10_cpu.png)
![inception_score_with_gpu](https://github.com/pengxin99/incubator-mxnet/blob/dcgan-inception_score/example/gluon/DCGAN/pic/inception_score_cifar10.png)

## Quick start
use below code to see the configurations you can set:
```python
python dcgan.py -h
```
    

    optional arguments:
      -h, --help            show this help message and exit
      --dataset DATASET     dataset to use. options are cifar10 and mnist.
      --batch-size BATCH_SIZE  input batch size, default is 64
      --nz NZ               size of the latent z vector, default is 100
      --ngf NGF             the channel of each generator filter layer, default is 64.
      --ndf NDF             the channel of each descriminator filter layer, default is 64.
      --nepoch NEPOCH       number of epochs to train for, default is 25.
      --niter NITER         save generated images and inception_score per niter iters, default is 100.
      --lr LR               learning rate, default=0.0002
      --beta1 BETA1         beta1 for adam. default=0.5
      --cuda                enables cuda
      --netG NETG           path to netG (to continue training)
      --netD NETD           path to netD (to continue training)
      --outf OUTF           folder to output images and model checkpoints
      --check-point CHECK_POINT
                            save results at each epoch or not
      --inception_score INCEPTION_SCORE
                            To record the inception_score, default is True.


use below code to train DCGAN model with default configurations and dataset(cifar10), and metric with inception_score:
```python
python dcgan.py
```
