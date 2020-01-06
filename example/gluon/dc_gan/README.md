<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# DCGAN in MXNet

[Deep Convolutional Generative Adversarial Networks(DCGAN)](https://arxiv.org/abs/1511.06434) implementation with Apache MXNet GLUON.
This implementation uses [inception_score](https://github.com/openai/improved-gan) to evaluate the model.

You can use this reference implementation on the MNIST and CIFAR-10 datasets.


#### Generated image output examples from the CIFAR-10 dataset
![Generated image output examples from the CIFAR-10 dataset](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/gluon/DCGAN/fake_img_iter_13900.png)

#### Generated image output examples from the MNIST dataset
![Generated image output examples from the MNIST dataset](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/gluon/DCGAN/fake_img_iter_21700.png)

#### inception_score in cpu and gpu (the real image`s score is around 3.3)
CPU & GPU

![inception score with CPU](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/gluon/DCGAN/inception_score_cifar10_cpu.png)
![inception score with GPU](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/gluon/DCGAN/inception_score_cifar10.png)

## Quick start
Use the following code to see the configurations you can set:
```bash
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


Use the following Python script to train a DCGAN model with default configurations using the CIFAR-10 dataset and record metrics with `inception_score`:
```bash
python dcgan.py
```
