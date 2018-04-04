
# Mixed precision training using float16

The computational resources required for training deep neural networks has been increasing of late because of complexity of the architectures and size of models. Mixed precision training allows us to reduces the resources required by using lower precision arithmetic. In this approach we train using 16 bit floating points (half precision) while using 32 bit floating points (single precision) for output buffers of float16 computation. This combination of single and half precision gives rise to the name Mixed precision. It allows us to achieve the same accuracy as training with single precision, while decreasing the required memory and training or inference time.

The float16 data type, is a 16 bit floating point representation according to the IEEE 754 standard. It has a dynamic range where the precision can go from 0.0000000596046 (highest, for values closest to 0) to 32 (lowest, for values in the range 32768-65536). Despite the decreased precision when compared to single precision (float32), float16 computation can be much faster on supported hardware. The motivation for using float16 for deep learning comes from the idea that deep neural network architectures have natural resilience to errors due to backpropagation. Half precision is typically sufficient for training neural networks. This means that on hardware with specialized support for float16 computation we can greatly improve the speed of training and inference. This speedup results from faster matrix multiplication, saving on memory bandwidth and reduced communication costs. It also reduces the size of the model, allowing us to train larger models and use larger batch sizes. 

The Volta range of Graphics Processing Units (GPUs) from Nvidia have Tensor Cores which perform efficient float16 computation. A tensor core allows accumulation of half precision products into single or half precision outputs. For the rest of this tutorial we assume that we are working with Nvidia's Tensor Cores on a Volta GPU.

In this tutorial we will walk through how one can train deep learning neural networks with mixed precision on supported hardware. We will first see how to use float16 and then some techniques on achieving good performance and accuracy.

## Prerequisites

- Volta range of Nvidia GPUs
- Cuda 9 or higher
- CUDNN v7 or higher

## Using the Gluon API

With Gluon, we need to take care of two things to convert a model to support float16.
1. Cast the Gluon Block, so as to cast the parameters of layers and change the type of input expected, to float16.
2. Cast the data to float16 to match the input type expected by the blocks if necessary.

### Training
Let us look at an example of training a Resnet50 model with the Caltech101 dataset with float16. 
First, let us get some import stuff out of the way.


```python
import os
import tarfile
import multiprocessing
import time
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon.model_zoo import vision as models
from mxnet.metric import Accuracy
from mxnet.gluon.data.vision.datasets import ImageFolderDataset
```

Let us start by fetching the Caltech101 dataset and extracting it. 


```python
url = "https://s3.us-east-2.amazonaws.com/mxnet-public/101_ObjectCategories.tar.gz"
dataset_name = "101_ObjectCategories"
data_folder = "data"
if not os.path.isdir(data_folder):
    os.makedirs(data_folder)
tar_path = mx.gluon.utils.download(url, path='data')
if (not os.path.isdir(os.path.join(data_folder, "101_ObjectCategories")) or 
    not os.path.isdir(os.path.join(data_folder, "101_ObjectCategories_test"))):
    tar = tarfile.open(tar_path, "r:gz")
    tar.extractall(data_folder)
    tar.close()
    print('Data extracted')
training_path = os.path.join(data_folder, dataset_name)
testing_path = os.path.join(data_folder, "{}_test".format(dataset_name))
```

Now we have the images in two folders, one for training and the other for test. Let us next create Gluon Dataset from these folders, and then create Gluon DataLoader from those datasets. Let us also define a transform function so that each image loaded is resized, cropped and transposed. 


```python
EDGE = 224
SIZE = (EDGE, EDGE)
NUM_WORKERS = multiprocessing.cpu_count()
# Lower batch size if you run out of memory on your GPU
BATCH_SIZE = 64

def transform(image, label):
    resized = mx.image.resize_short(image, EDGE)
    cropped, crop_info = mx.image.center_crop(resized, SIZE)
    transposed = nd.transpose(cropped, (2,0,1))
    return transposed, label

dataset_train = ImageFolderDataset(root=training_path, transform=transform)
dataset_test = ImageFolderDataset(root=testing_path, transform=transform)

train_data = gluon.data.DataLoader(dataset_train, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_data = gluon.data.DataLoader(dataset_test, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
```

Next, we'll define softmax cross entropy as our loss, accuracy as our metric and the context on which to run our training jobs. It is set by default to gpu. Please note that float16 on CPU might not be supported for all operators, as float16 on CPU is slower than float32.


```python
ctx = mx.gpu(0)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
metric = Accuracy()
```

Now, let us fetch our model from Gluon ModelZoo and initialize the parameters. Let us also hybridize the net for efficiency. Here comes the first change we need to make to use float16 for the neural network. We **cast the network** to our required data type. Let us keep the data type as an argument so that we can compare float32 and float16 easily later.


```python
# Creating the network
def get_network(dtype):
    net = models.get_model(name='resnet50_v2', ctx=ctx, pretrained=False, classes=101)
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    net.hybridize()
    net.cast(dtype)
    return net
```

It's preferable to use **multi_precision mode of optimizer** when training in float16. This mode of optimizer maintains the weights in float32 even when the training is in float16. This helps increase precision of the weights and leads to faster convergence for some networks. (Further discussion on this towards the end.)


```python
optimizer = mx.optimizer.create('sgd', multi_precision=True, lr=0.01)
```

Let us next define helper functions `test` and `train`. Here comes the next change we need to make. We need to **cast the data** to float16. Note the use of `astype` in the below functions to ensure this.


```python
def test(net, val_data, dtype):
    metric.reset()
    for (data, label) in val_data:
        data = data.as_in_context(ctx).astype(dtype)
        label = label.as_in_context(ctx)
        output = net(data)
        metric.update(label, output)
    return metric.get()
```


```python
def train(net, dtype, num_epochs):
    print('Starting training with %s' % dtype)
    trainer = gluon.Trainer(net.collect_params(), optimizer)
    for epoch in range(num_epochs):
        tic = time.time()
        metric.reset()
        btic = time.time()
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx).astype(dtype)
            label = label.as_in_context(ctx)
            outputs = []
            Ls = []
            with autograd.record():
                z = net(data)
                L = loss(z, label)
            L.backward()            
            trainer.step(data.shape[0])
            metric.update(label, z)
            if i and not i%50:
                name, acc = metric.get()
                print('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f'%(
                               epoch, i, BATCH_SIZE/(time.time()-btic), name, acc))
            btic = time.time()

        name, acc = metric.get()
        print('[Epoch %d] training: %s=%f'%(epoch, name, acc))
        print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        name, val_acc = test(net, test_data, dtype)
        print('[Epoch %d] validation: %s=%f'%(epoch, name, val_acc))
```

Now let's start use the above functions together to create a network and start training with float16. 


```python
DTYPE = 'float16'
net = get_network(DTYPE)
train(net, dtype=DTYPE, num_epochs=25)
```

Note the accuracy you observe above. You can change DTYPE above to float32 if you want to observe the speedup gained by using float16.


### Finetuning

You can also finetune in float16, a model which was originally trained in float32. The section of the code which builds the network would now look as follows. We first fetch the pretrained resnet50_v2 model from model zoo. This was trained using Imagenet data, so we need to pass classes as 1000 for fetching the pretrained model. Then we create our new network for Caltech 101 by passing number of classes as 101. We will then cast it to `float16` so that we cast all parameters to `float16`.


```python
def get_pretrained_net(dtype):
    pretrained_net = models.get_model(name='resnet50_v2', ctx=ctx, pretrained=True, classes=1000)
    pretrained_net.hybridize()
    pretrained_net.cast(dtype)

    net = models.get_model(name='resnet50_v2', ctx=ctx, pretrained=False, classes=101)
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    net.features = pretrained_net.features
    net.hybridize()
    net.cast(dtype)
    return net
```

Now let us use the above function to get a pretrained network and train in float16.


```python
DTYPE = 'float16'
net = get_pretrained_net(DTYPE)
train(net, dtype=DTYPE, num_epochs=25)
```

We can confirm above that the pretrained model helps achieve much higher accuracy of about 0.97 in the same number of epochs.

## Using the Symbolic API

Training a network in float16 with the Symbolic API involves the following steps.
1. Add a layer at the beginning of the network, to cast the data to float16. This will ensure that all the following layers compute in float16.
2. It is advisable to cast the output of the layers before softmax to float32, so that the softmax computation is done in float32. This is because softmax involves large reductions and it helps to keep that in float32 for more precise answer.

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

Here's how you can use the above script to train Resnet50 model with Imagenet data using float16:
```
python train_imagenet.py --network resnet --num-layers 50 --data-train ~/efs/data/imagenet/imagenet-train.rec --data-val ~/efs/data/imagenet/imagenet-val.rec --gpus 0 --batch-size 64 --dtype float16
```

There's a similar example for fine tuning [here](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification/fine-tune.py). The following command shows how to use that script to fine tune a Resnet50 model trained on Imagenet for the Caltech 256 dataset using float16.
```
python fine-tune.py --network resnet --num-layers 50 --pretrained-model imagenet1k-resnet-50 --data-train ~/efs/data/caltech-256/caltech256-train.rec --data-val ~/efs/data/caltech-256/caltech256-val.rec --num-examples 15420 --num-classes 256 --gpus 0 --batch-size 64 --dtype float16
```


## Things to keep in mind

### For performance
1. Nvidia Tensor core essentially perform the computation D = A * B + C, where A and B are half precision matrices, while C and D could be either half precision or full precision. The tensor cores are most efficient when dimensions of these matrices are multiples of 8. This means that Tensor Cores can not be used in all cases for fast float16 computation. When training models like Resnet50 on the Cifar10 dataset, the tensors involved are sometimes smaller, and tensor cores can not always be used. The computation in that case falls back to slower algorithms and using float16 turns out to be slower than float32 on a single GPU. Note that when using multiple GPUs, using float16 can still be faster than float32 because of reduction in communication costs.

2. It is advisable to use batch sizes that are multiples of 8 because of the above reason when training with float16. As always, batch sizes which are powers of 2 would be best when compared to those around it.

3. You can check whether your program is using Tensor cores for fast float16 computation by profiling with `nvprof`.
The operations with `s884cudnn` in their names represent the use of Tensor cores.

4. When not limited by GPU memory, it can help to set the environment variable MXNET_CUDNN_AUTOTUNE_DEFAULT to 2. This configures MXNet to run tuning tests and choose the fastest convolution algorithm whose memory requirements may exceed the default memory of CUDA workspace.

### For accuracy

#### Multi precision mode
When training in float16, it is advisable to still store the master copy of the weights in float32 for better accuracy. The higher precision of float32 helps overcome cases where gradient update can become 0 if represented in float16. This mode can be activated by setting the parameter `multi_precision` of optimizer params to `True` as in the above example. It has been found that this is not required for all networks to achieve the same accuracy as with float32, but nevertheless recommended. Note that for distributed training, this is currently slightly slower than without `multi_precision`, but still faster than using float32 for training.


#### Large reductions 
Since float16 has low precision for large numbers, it is best to leave layers which perform large reductions in float32. This includes BatchNorm and Softmax. Ensuring that batchnorm performs reduction in float32 is handled by default in both Gluon and Module APIs. While Softmax is set to use float32 even during float16 training in Gluon, in the Module API there needs to be a cast to float32 before softmax as the above symbolic example code shows.

#### Loss scaling
For some networks just switching the training to float16 mode was not found to be enough to reach the same accuracy as when training with float32. This is because the activation gradients computed are too small and could not be represented in float16 representable range. Such networks can be made to achieve the accuracy reached by float32 with a couple of changes. 

Most of the float16 representable range is not used by activation gradients generally. So we can shift the gradients into float16 range by scaling up the loss by a factor `S`. Essentially we scale up the loss before backward pass, and then scale back the gradients before updating the weights.

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

Networks like Multibox SSD, R-CNN, bigLSTM and Seq2seq were found to exhibit such behavior. Refer the linked articles below for more details on this.
    
## References
1. [Training with Mixed Precision User Guide](http://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)
2. [Mixed Precision Training at ICLR 2018](https://arxiv.org/pdf/1710.03740.pdf)
3. [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/)

