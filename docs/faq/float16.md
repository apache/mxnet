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

# Mixed precision training using float16

In this tutorial you will walk through how one can train deep learning neural networks with mixed precision on supported hardware. You will first see how to use float16 (both with Gluon and Symbolic APIs) and then some techniques on achieving good performance and accuracy.

## Background
The computational resources required for training deep neural networks have been increasing of late because of complexity of the architectures and size of models. Mixed precision training allows us to reduces the resources required by using lower precision arithmetic. In this approach you can train using 16 bit floating points (half precision) while using 32 bit floating points (single precision) for output buffers of float16 computation. This combination of single and half precision gives rise to the name mixed precision. It allows us to achieve the same accuracy as training with single precision, while decreasing the required memory and training or inference time.

The float16 data type is a 16 bit floating point representation according to the IEEE 754 standard. It has a dynamic range where the precision can go from 0.0000000596046 (highest, for values closest to 0) to 32 (lowest, for values in the range 32768-65536). Despite the inherent reduced precision when compared to single precision float (float32), using float16 has many advantages. The most obvious advantages are that you can reduce the size of the model by half allowing the training of larger models and using larger batch sizes. The reduced memory footprint also helps in reducing the pressure on memory bandwidth and lowering communication costs. On hardware with specialized support for float16 computation you can also greatly improve the speed of training and inference. The Volta range of Graphics Processing Units (GPUs) from Nvidia have [Tensor Cores](https://www.nvidia.com/en-us/data-center/tensorcore/) which perform efficient float16 computation. A tensor core allows accumulation of half precision products into single or half precision outputs. For the rest of this tutorial we assume that we are working with Nvidia's Tensor Cores on a Volta GPU.

## Prerequisites
- Volta range of Nvidia GPUs
- Cuda 9 or higher
- CUDNN v7 or higher

This tutorial also assumes that you understand how to train a network with float32. Please refer to other tutorials [here](http://mxnet.incubator.apache.org/tutorials/index.html) to get started with MXNet and/or Gluon. This tutorial focuses on the changes needed to switch from float32 to mixed precision and tips on achieving the best performance with mixed precision.

## Using the Gluon API

### Training or Inference

With Gluon, you need to take care of three things to convert a model to support float16.

1. Cast the Gluon Block, so as to cast the parameters of layers and change the type of input expected, to float16. This is as simple as calling the [cast](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html#mxnet.gluon.Block.cast) method of the Block representing the network.
```
net = net.cast('float16')
```

2. Ensure the data input to the network is of float16 type. If your DataLoader or Iterator produces output in another datatype, then you would have to cast your data. There are different ways you can do this. The easiest would be to use the [`astype`](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.astype) method of ndarrays.
```
data = data.astype('float16', copy=False)
```

If you are using images and DataLoader, you can also use a [Cast transform](https://mxnet.incubator.apache.org/api/python/gluon/data.html#mxnet.gluon.data.vision.transforms.Cast)

3. It is preferable to use **multi_precision mode of optimizer** when training in float16. This mode of optimizer maintains a master copy of weights in float32 even when the training (i.e. forward and backward pass) is in float16. This helps increase precision of the weight updates and can lead to faster convergence for some networks. (Further discussion on this towards the end.)

```python
optimizer = mx.optimizer.create('sgd', multi_precision=True, lr=0.01)
```

You can play around with mixed precision using the image classification example [here](https://github.com/apache/incubator-mxnet/blob/master/example/gluon/image_classification.py). We suggest using the Caltech101 dataset option in that example and using a Resnet50_v1 network so you can quickly see the performance improvement and how the accuracy is unaffected. Here's a starter command to run this.

```
python image_classification.py --model resnet50_v1 --dataset caltech101 --gpus 0 --num-worker 30 --dtype float16
```


### Fine-tuning

You can also fine-tune in float16, a model which was originally trained in float32. Here is how you would do it. As an example if you are trying to use a model pretrained on the Imagenet dataset from the ModelZoo, you would first fetch the pretrained network and then cast that network to float16.

```
pretrained_net = models.get_model(name='resnet50_v2', ctx=ctx, pretrained=True, classes=1000)
pretrained_net.cast('float16')
```
Then if you have another Resnet50_v2 model you want to fine-tune, you can just assign the features to that network and then cast it.

```
net = models.get_model(name='resnet50_v2', ctx=ctx, pretrained=False, classes=101)
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
net.features = pretrained_net.features
net.cast(dtype)
```

## Using the Symbolic API

Training a network in float16 with the Symbolic API involves the following steps.
1. Add a layer at the beginning of the network, to cast the data to float16. This will ensure that all the following layers compute in float16.
2. It is advisable to cast the output of the layers before softmax to float32, so that the softmax computation is done in float32. This is because softmax involves large reductions and it helps to keep that in float32 for more precise answer.
3. It is advisable to use the multi-precision mode of the optimizer for more precise weight updates. This is discussed in some detail below. Here's how you would enable this mode when creating an optimizer.

```python
optimizer = mx.optimizer.create('sgd', multi_precision=True, lr=0.01)
```

There are a few examples of building such networks which can handle float16 input in [examples/image-classification/symbols/](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification/symbols). Specifically you could look at the [resnet](https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/symbols/resnet.py) example.

An illustration of the relevant section of the code is below.
```
data = mx.sym.Variable(name="data")
if dtype == 'float16':
    data = mx.sym.Cast(data=data, dtype=np.float16)

// the rest of the network 
net_out = net(data)

if dtype == 'float16':
    net_out = mx.sym.Cast(data=net_out, dtype=np.float32)
output = mx.sym.SoftmaxOutput(data=net_out, name='softmax')
```

We have an example script which show how to train imagenet with resnet50 using float16 [here](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification/train_imagenet.py) 

Here's how you can use the above script to train Resnet50 v1 model with synthetic data using float16, so you can try it out even if you don't have the Imagenet dataset handy.
```
python train_imagenet.py --network resnet-v1 --num-layers 50 --benchmark 1 --gpus 0 --batch-size 256 --dtype float16
```

There's a similar example for fine tuning [here](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification/fine-tune.py). The following command shows how to use that script to fine tune a Resnet50 model trained on Imagenet for the Caltech 256 dataset using float16.
```
python fine-tune.py --network resnet --num-layers 50 --pretrained-model imagenet1k-resnet-50 --data-train ~/data/caltech-256/caltech256-train.rec --data-val ~/data/caltech-256/caltech256-val.rec --num-examples 15420 --num-classes 256 --gpus 0 --batch-size 64 --dtype float16
```

## Example training results
Let us consider training a Resnet50 v1 model on the Imagenet 2012 dataset. For this model, the GPU memory usage is close to the capacity of V100 GPU with a batch size of 128 when using float32. Using float16 allows the use of 256 batch size. Shared below are results using 8 V100 GPUs on a AWS p3.16x large instance. Let us compare the three scenarios that arise here: float32 with 1024 batch size, float16 with 1024 batch size and float16 with 2048 batch size. These jobs trained for 90 epochs using a learning rate of 0.4 for 1024 batch size and 0.8 for 2048 batch size. This learning rate was decayed by a factor of 0.1 at the 30th, 60th and 80th epochs. The only changes made for the float16 jobs when compared to the float32 job were that the network and data were cast to float16, and the multi-precision mode was used for optimizer. The final accuracy at 90th epoch and the time to train are tabulated below for these three scenarios. The top-1 validation errors at the end of each epoch are also plotted below.

Batch size | Data type | Top 1 Validation accuracy | Time to train | Speedup |
--- | --- | --- | --- | --- |
1024 | float32 | 76.18% | 11.8 hrs | 1 |
1024 | float16 | 76.34% | 7.3 hrs | 1.62x |
2048 | float16 | 76.29% | 6.5 hrs | 1.82x |

![Training curves of Resnet50 v1 on Imagenet 2012](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/mixed-precision/resnet50v1b_imagenet_fp16_fp32_training.png)

The differences in accuracies above are within normal random variation, and there is no reason to expect float16 to have better accuracy than float32 in general. As the plot indicates training behaves similarly for these cases, even though we didn't have to change any other hyperparameters. We can also see from the table that using float16 helps train faster through faster computation with float16 as well as allowing the use of larger batch sizes.

## Things to keep in mind

### For performance

Typical performance gains seen for float16 typically range 1.6x-2x for convolutional networks like Resnet and even about 3x for networks with LSTMs. The performance gain you see can depend on certain things which this section will introduce you to.

1. Nvidia Tensor Cores essentially perform the computation D = A * B + C, where A and B are half precision matrices, while C and D could be either half precision or full precision. The tensor cores are most efficient when dimensions of these matrices are multiples of 8. This means that Tensor Cores can not be used in all cases for fast float16 computation. When training models like Resnet50 on the Cifar10 dataset, the tensors involved are sometimes smaller, and Tensor Cores can not always be used. The computation in that case falls back to slower algorithms and using float16 turns out to be slower than float32 on a single GPU. Note that when using multiple GPUs, using float16 can still be faster than float32 because of reduction in communication costs.

2. When you scale up the batch size ensure that IO and data pre-processing is not your bottleneck. If you see a slowdown this would be the first thing to check.

3. It is advisable to use batch sizes that are multiples of 8 because of the above reason when training with float16. As always, batch sizes which are powers of 2 would be best when compared to those around it.

4. You can check whether your program is using Tensor cores for fast float16 computation by profiling with `nvprof`.
The operations with `s884cudnn` in their names represent the use of Tensor cores.

5. When not limited by GPU memory, it can help to set the environment variable MXNET_CUDNN_AUTOTUNE_DEFAULT to 2. This configures MXNet to run tuning tests and choose the fastest convolution algorithm whose memory requirements may exceed the default memory of CUDA workspace.

6. Please note that float16 on CPU might not be supported for all operators, as in most cases float16 on CPU is much slower than float32.


### For accuracy

#### Multi precision mode
When training in float16, it is advisable to still store the master copy of the weights in float32 for better accuracy. The higher precision of float32 helps overcome cases where gradient update can become 0 if represented in float16. This mode can be activated by setting the parameter `multi_precision` of optimizer params to `True` as in the above example. It has been found that this is not required for all networks to achieve the same accuracy as with float32, but nevertheless recommended. Note that for distributed training, this is currently slightly slower than without `multi_precision`, but still much faster than using float32 for training.

#### Large reductions 
Since float16 has low precision for large numbers, it is best to leave layers which perform large reductions in float32. This includes BatchNorm and Softmax. Ensuring that Batchnorm performs reduction in float32 is handled by default in both Gluon and Module APIs. While Softmax is set to use float32 even during float16 training in Gluon, in the Module API there needs to be a cast to float32 before softmax as the above symbolic example code shows.

#### Loss scaling
For some networks just switching the training to float16 mode was not found to be enough to reach the same accuracy as when training with float32. This is because the activation gradients computed are too small and could not be represented in float16 representable range. Such networks can be made to achieve the accuracy reached by float32 with a couple of changes. 

Most of the float16 representable range is not used by activation gradients generally. So you can shift the gradients into float16 range by scaling up the loss by a factor `S`. By the chain rule, this scales up the loss before backward pass, and then you can scale back the gradients before updating the weights. This ensures that training in float16 can use the same hyperparameters as used during float32 training.

Here's how you can configure the loss to be scaled up by 128 and rescale the gradient down before updating the weights.

*Gluon*
```
loss = gluon.loss.SoftmaxCrossEntropyLoss(weight=128)
optimizer = mx.optimizer.create('sgd', multi_precision=True, rescale_grad=1.0/128)
```
*Module*
```
mxnet.sym.SoftmaxOutput(other_args, grad_scale=128.0)
optimizer = mx.optimizer.create('sgd', multi_precision=True, rescale_grad=1.0/128)
```

Networks like Multibox SSD, R-CNN, bigLSTM and Seq2seq were found to exhibit such behavior.
You can choose a constant scaling factor while ensuring that the absolute value of gradient when multiplied by this factor remains in the range of float16. Generally powers of 2 like 64,128,256,512 are chosen. Refer the linked articles below for more details on this.

## Video Tutorial

We also have a video tutorial for using Mixed Precision with MXNet. You can check that out [here](https://www.youtube.com/watch?v=pR4KMh1lGC0)

## References
1. [Training with Mixed Precision User Guide](http://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)
2. [Mixed Precision Training at ICLR 2018](https://arxiv.org/pdf/1710.03740.pdf)
3. [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/)

