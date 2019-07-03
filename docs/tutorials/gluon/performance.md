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

# Gluon Performance Tips & Tricks

Compared to traditional machine learning methods, the field of deep-learning has increased model accuracy across a wide range of tasks, but it has also increased the amount of computation required for model training and inference. Specialised hardware chips, such as GPUs and FPGAs, can speed up the execution of networks, but it can sometimes be hard to write code that uses the hardware to its full potential. We will be looking at a few simple tips and trick in this tutorial that you can use to speed up training and ultimately save on training costs. You'll find most of these tips and tricks useful for inference too.

We'll start by writing some code to train an image classification network for the CIFAR-10 dataset, and then benchmark the throughput of the network in terms of samples processed per second. After some performance analysis, we'll identify the bottlenecks (i.e. the components limiting throughput) and improve the training speed step-by-step. We'll bring together all the tips and tricks at the end and evaluate our performance gains.


```python
from __future__ import print_function
import multiprocessing
import time
import mxnet as mx
import numpy as np
```

An Amazon EC2 p3.2xlarge instance was used to benchmark the code in this tutorial. You are likely to get different results and find different bottlenecks on other hardware, but these tips and tricks should still help improve training speed for bottleneck components. A GPU is recommended for this tutorial.


```python
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
print("Using {} context.".format(ctx))
```

    Using gpu(0) context.


We'll use the `CIFAR10` dataset provided out-of-the-box with Gluon.


```python
dataset = mx.gluon.data.vision.CIFAR10(train=True)
print('{} samples'.format(len(dataset)))
```

    50000 samples


So we can learn how to identify training bottlenecks, let's intentionally introduce a bottleneck by adding a short `sleep` into the data loading pipeline. We transform each 32x32 CIFAR-10 image to 224x224 so we can use it with the ResNet-50 network designed for ImageNet. [CIFAR-10 specific ResNet networks](https://gluon-cv.mxnet.io/api/model_zoo.html#gluoncv.model_zoo.get_cifar_resnet) exist but we use the more standard ImageNet variants in this example.


```python
def transform_fn(x):
    time.sleep(0.01)  # artificial slow-down
    image = mx.image.imresize(x, w=224, h=224)
    return image.astype('float32').transpose((2, 0, 1))

dataset = dataset.transform_first(transform_fn)
```

Setting our batch size to 16, we can create the `DataLoader`.


```python
batch_size = 16
dataloader = mx.gluon.data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      last_batch="discard")
print('{} batches'.format(len(dataloader)))
```

    3125 batches


Up next, we create all of the other components required for training, such as the network, the loss function, the evaluation metric and parameter trainer.


```python
net = mx.gluon.model_zoo.vision.resnet50_v2(pretrained=False, ctx=ctx)
net.initialize(mx.init.Xavier(magnitude=2.3), ctx=ctx)
loss_fn = mx.gluon.loss.SoftmaxCrossEntropyLoss()
metric = mx.metric.Accuracy()
learning_rate = 0.001
trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})
```

## Initial Benchmark

As a starting point, let's benchmark the throughput of our training loop: calculating the average samples per second across 25 iterations, where each iteration is a batch of 16 samples. We'll run a single forward pass through the network before starting our benchmark timer to avoid including shape inference and lazy initialization in the throughput calculations.


```python
def single_forward(net, dataloader, dtype='float32'):
    data, label = next(iter(dataloader))
    data = data.astype(dtype)
    data = data.as_in_context(ctx)
    pred = net(data)
    pred.wait_to_read()
```


```python
single_forward(net, dataloader)
iters = 25
num_samples = 0
num_iters = 0
start_time = time.time()
for iter_idx, (data, label) in enumerate(dataloader):
    num_samples += data.shape[0]
    num_iters += 1
    data = data.as_in_context(ctx)
    label = label.as_in_context(ctx)
    with mx.autograd.record():
        pred = net(data)
        loss = loss_fn(pred, label)
    loss.backward()
    trainer.step(data.shape[0])
    metric.update(label, pred)
    print('.', end='')
    if num_iters >= iters:
        break
mx.nd.waitall()
end_time = time.time()
total_time = end_time - start_time
print('\n')
print('average iterations/sec: {:.4f}'.format(num_iters/total_time))
print('average samples/sec: {:.4f}'.format(num_samples/total_time))
```

    .........................
    
    average iterations/sec: 4.4936
    average samples/sec: 71.8975


Although ~70 samples per second might sound respectable, let's see if we can do any better by identifying the bottleneck in the training loop and optimizing that component. A significant amount of time can be wasted by optimizing components that aren't bottlenecks.

## Identifying the bottleneck

Monitoring the CPU (with `top`) and GPU utilization (with `nvidia-smi`) provide clues as to where potential bottlenecks lie. With the example above, when simultaneously running these monitoring tool, you might spot a single process on the CPU fixed at ~100% utilization while the GPU utilization behaves erratically and often falls to ~0%. Seeing behaviour like can indicate the CPU is struggling to process data and the GPU is being starved of data.

MXNet's Profiler is another highly recommended tool for identifying bottlenecks, since it gives timing data for individual MXNet operations. Check out [this comprehensive tutorial](https://mxnet.incubator.apache.org/versions/master/tutorials/python/profiler.html) for more details. As a simpler form of analysis, we will split our training loop into two common components:

1. Data Loading
2. Network Execution (forward and backward passes)

We define two function to independently benchmark these components: `benchmark_dataloader` and `benchmark_network`.


```python
def benchmark_dataloader(dataloader, iters=25):
    num_samples = 0
    num_iters = 0
    start_time = time.time()
    startup_time = None
    for iter_idx, sample in enumerate(dataloader):
        if startup_time is None:
            startup_time = time.time()
        num_samples += sample[0].shape[0]
        num_iters += 1
        if num_iters >= iters:
            break
        print('.', end='')
    end_time = time.time()
    total_time = end_time - start_time
    total_startup_time = startup_time - start_time
    total_iter_time = end_time - startup_time
    print('\n')
    print('total startup time: {:.4f}'.format(total_startup_time))
    print('average iterations/sec: {:.4f}'.format(num_iters/total_iter_time))
    print('average samples/sec: {:.4f}'.format(num_samples/total_iter_time))
    
    
def benchmark_network(data, label, net, loss_fn, trainer, iters=25):
    num_samples = 0
    num_iters = 0
    mx.nd.waitall()
    start_time = time.time()
    for iter_idx in range(iters):
        num_samples += data.shape[0]
        num_iters += 1
        with mx.autograd.record():
            pred = net(data)
            loss = loss_fn(pred, label)
        loss.backward()
        trainer.step(data.shape[0])
        mx.nd.waitall()
        if num_iters >= iters:
            break
        print('.', end='')
    end_time = time.time()
    total_time = end_time - start_time
    print('\n')
    print('average iterations/sec: {:.4f}'.format(num_iters/total_time))
    print('average samples/sec: {:.4f}'.format(num_samples/total_time))
```

Our `benchmark_dataloader` function just loops through the `DataLoader` for a given number of iterations: it doesn't transfer the data to the correct context or pass it to the network. Our `benchmark_network` function just performs a forward and backward pass on an identical (and pre-transferred) batch of data: it doesn't require new data to be loaded. We'll run both of these functions now.


```python
print('\n', '### benchmark_dataloader', '\n')
benchmark_dataloader(dataloader)
print('\n', '### benchmark_network', '\n')
data, label = next(iter(dataloader))
data = data.as_in_context(ctx)
label = label.as_in_context(ctx)
benchmark_network(data, label, net, loss_fn, trainer)
```

    
     ### benchmark_dataloader 
    
    ........................
    
    total startup time: 0.1697
    average iterations/sec: 6.2201
    average samples/sec: 99.5217
    
     ### benchmark_network 
    
    ........................
    
    average iterations/sec: 15.1908
    average samples/sec: 243.0525


Our data loading pipeline appears to be the bottleneck for training: ~100 samples/second compared with ~250 samples/second for network execution. One limiting factor could be disk throughput when reading samples (using a SSD instead of HDD can help with this), but in this case we intentionally added a delay in data transformation. Augmentation can often be a bottleneck in training if the following trick isn't applied.

## Tips & Tricks #1: Use multiple workers on `DataLoader`

In the previous section, we established that the data loading component of the training loop was the bottleneck. Instead of simply removing the artificial delay, let's assume it was some pre-processing or augmentation step that couldn't be removed. We found that the CPU utilization was fixed at 100%, but this was just for a single core. Usually machines have multiple cores and with one easy trick we can leverage more CPU cores to pre-process the data. Setting `num_workers` on the `DataLoader` will result in multiple workers being used to preprocess the data. We can use `multiprocessing.cpu_count()` to find the number of CPU cores available on the machine, and we save 1 core for the main thread.


```python
num_workers = multiprocessing.cpu_count() - 1
dataloader = mx.gluon.data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      last_batch="discard",
                                      num_workers=num_workers)
print('Using {} workers for DataLoader.'.format(num_workers))
```

    Using 7 workers for DataLoader.


We benchmark the two main components once again:


```python
print('\n', '### benchmark_dataloader', '\n')
benchmark_dataloader(dataloader)
print('\n', '### benchmark_network', '\n')
data, label = next(iter(dataloader))
data = data.as_in_context(ctx)
label = label.as_in_context(ctx)
benchmark_network(data, label, net, loss_fn, trainer, iters=10)
```

    
     ### benchmark_dataloader 
    
    ........................
    
    total startup time: 0.1935
    average iterations/sec: 46.2375
    average samples/sec: 739.7994
    
     ### benchmark_network 
    
    .........
    
    average iterations/sec: 14.8614
    average samples/sec: 237.7819


Our data loading pipeline is no longer the bottleneck for training throughput: ~700 samples per second versus ~250 samples per second for network execution as before. Check out [NVIDIA DALI](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/mxnet/gluon.html) if you need to futher optimize your data pipeline. We can now focus our attention on improving the network throughput.

## Tips & Tricks #2: Hybridize the network

Gluon networks run in imperative mode by default, executing `NDArray` operations as the lines of code are stepped through one by one. While in imperative mode, debugging is often simplified and more flexible networks can be defined (using Python control flow). But this comes at a slight cost in terms of throughput. Since the network doesn't know what line of code will be run next, the network operations cannot be optimized and additional memory allocations are required (which all takes time). Most networks though can be written as `HybridBlocks` and, with the `hybridize` method, can be converted to symbolic mode execution. We can expect throughput to increase slightly in this mode. Watch out though: debugging can get more complicated. Setting `static_alloc=True` and `static_shape=True` reduce the number of memory allocations required while training. Once again, we run `single_forward` to force the hybridization process to occur before benchmarking.


```python
net.hybridize(static_alloc=True, static_shape=True)
single_forward(net, dataloader)
```


```python
print('\n', '### benchmark_dataloader', '\n')
benchmark_dataloader(dataloader)
print('\n', '### benchmark_network', '\n')
data, label = next(iter(dataloader))
data = data.as_in_context(ctx)
label = label.as_in_context(ctx)
benchmark_network(data, label, net, loss_fn, trainer)
```

    
     ### benchmark_dataloader 
    
    ........................
    
    total startup time: 0.1847
    average iterations/sec: 46.3004
    average samples/sec: 740.8072
    
     ### benchmark_network 
    
    ........................
    
    average iterations/sec: 16.5738
    average samples/sec: 265.1812


We can see quite a modest ~10% increase in throughput after hybridization. Gains can depend on a number of factors including the network architecture and the batch size used (a larger increase expected for smaller batch size). Our network execution is still the bottleneck in training so let's focus on that again.

## Tips & Tricks #3: Increase the batch size

GPUs are optimized for high throughput and they do this by performing many operations in parallel. Our NVIDIA Tesla V100 GPU utilization peaks at ~85% while running the last example. Although this is already quite high, there's still room improvement given this metric shows the percentage of time *at least one* kernel is running (over the last 1 second by default). Given we have enough memory available, the throughput of the network can be improved by increasing the batch size since more samples are processed in parallel. At this stage we're using approximately 1/4 of the available GPU memory, so let's increase out batch size by a factor of 4, from 16 to 64. Changing the batch size does have some side effects though. Using the same optimizer with same hyperparameters often leads to slower convergence. More gradients, from more samples, are averaged which leads to a smaller variance in the batch gradient overall. One simple trick to mitigate this is to increase the learning rate by the same factor: so in this case, from 0.001 to 0.004.


```python
batch_size = batch_size * 4
print('batch_size: {}'.format(batch_size))
learning_rate = learning_rate * 4
print('learning_rate: {}'.format(learning_rate))
dataloader = mx.gluon.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            last_batch="discard",
                                            num_workers=num_workers)
trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})
single_forward(net, dataloader)
```

    batch_size: 64
    learning_rate: 0.004



```python
print('\n', '### benchmark_dataloader', '\n')
benchmark_dataloader(dataloader)
print('\n', '### benchmark_network', '\n')
data, label = next(iter(dataloader))
data = data.as_in_context(ctx)
label = label.as_in_context(ctx)
benchmark_network(data, label, net, loss_fn, trainer)
```

    
     ### benchmark_dataloader 
    
    ........................
    
    total startup time: 0.7195
    average iterations/sec: 11.7149
    average samples/sec: 749.7521
    
     ### benchmark_network 
    
    ........................
    
    average iterations/sec: 5.5370
    average samples/sec: 354.3710


Once again, we see improvements in throughput. ~30% higher this time. Checking GPU memory usage, we still have room to increase the batch size higher than 64 (on NVIDIA Tesla V100). When the batch size starts to reach very large numbers (>512), simple tricks such as linear scaling of the learning rate might be insufficient for maintaining good convergence. Consider using a [warm-up learning rate schedule](https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/learning_rate_schedules_advanced.html) and changing to specialized optimizers such as [LBSGD](https://mxnet.incubator.apache.org/api/python/optimization/optimization.html#mxnet.optimizer.LBSGD). 

## Tips & Tricks #4: Using Mixed-Precision (`float32` and `float16`)

Model execution is still our bottleneck, so let's try out a new trick called [mixed precision](https://mxnet.incubator.apache.org/versions/master/faq/float16.html) training. Some recent GPUs have cores that are optimized for 'half-precision' (i.e. `float16`) operations and they can be much faster than their 'full-precision' (i.e. `float32`) counterparts. Given all of the randomness already in neural network training, this reduction in precision doesn't significantly impact the model accuracy in many cases. Convergence is slightly better when you keep the network parameters at full-precision but forward and backward passes can be performed at half-precision: hence the term 'mixed-precision'. Also check out [Automatic Mixed Precision](https://mxnet.incubator.apache.org/versions/master/tutorials/amp/amp_tutorial.html) (AMP) for a more automated way of optimizing your network.

We need to `cast` the network to `'float16'`, configure our optimizer to use `multi_precision` and convert our input data types to `'float16'` too.


```python
net.cast('float16')
trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate,
                                                         'multi_precision': True})
single_forward(net, dataloader, dtype='float16')
```


```python
print('\n', '### benchmark_dataloader', '\n')
benchmark_dataloader(dataloader)
print('\n', '### benchmark_network', '\n')
data, label = next(iter(dataloader))
data = data.astype('float16').as_in_context(ctx)
label = label.astype('float16').as_in_context(ctx)
benchmark_network(data, label, net, loss_fn, trainer)
```

    
     ### benchmark_dataloader 
    
    ........................
    
    total startup time: 0.7006
    average iterations/sec: 11.6537
    average samples/sec: 745.8338
    
     ### benchmark_network 
    
    ........................
    
    average iterations/sec: 11.3614
    average samples/sec: 727.1298


Overall we see a substantial increase in training throughput: double compared to full-precision training.

## Tips & Tricks #5: Others

Many other tips and tricks exist for optimizing the throughput of training.

One area we didn't explicitly benchmark in this tutorial is data transfer from CPU to GPU memory. Usually this isn't an issue, but for very large arrays this can become a bottleneck too. You might be able to compress your data significantly before transferring if your data is sparse (i.e. mostly zero values). Check out the [sparse array tutorial](https://mxnet.incubator.apache.org/versions/master/tutorials/index.html) for more details and an example of how this can impact training speed.

Another useful trick if data pre-processing or data transfer is the bottleneck is pre-fetching batches. You can write your training loop to transfer the next batch of data to GPU before processing the current batch. Once again, this trick is memory permitting.

And finally, if you are an advanced user, check out the various [environment variables](https://mxnet.incubator.apache.org/faq/env_var.html) that can be configured to change the behaviour of the MXNet backend.

## Final Benchmark

We will now combine all of the above tricks and tips in the complete training loop and compare to the initial benchmark.


```python
iters = 25
num_samples = 0
num_iters = 0
start_time = time.time()
for iter_idx, (data, label) in enumerate(dataloader):
    num_samples += data.shape[0]
    num_iters += 1
    data = data.as_in_context(ctx).astype('float16')
    label = label.as_in_context(ctx).astype('float16')
    with mx.autograd.record():
        pred = net(data)
        loss = loss_fn(pred, label)
    loss.backward()
    trainer.step(data.shape[0])
    metric.update(label, pred)
    print('.', end='')
    if num_iters >= iters:
        break
end_time = time.time()
total_time = end_time - start_time
print('\n')
print('average iterations/sec: {:.4f}'.format(num_iters/total_time))
print('average samples/sec: {:.4f}'.format(num_samples/total_time))
```

    .........................
    
    average iterations/sec: 7.9435
    average samples/sec: 508.3821


Using the above tips and tricks we managed to increase the throughput of training by ~600% from the initial benchmark! Our training throughput is less than the throughput of the individual components we tested, but there are additional overheads that we didn't previously measure (such as data transfer to GPU).

## Conclusion

We learned a number of tips and tricks to optimize the throughput of training, and they lead to a considerable increase compared to our initial baseline. As general rules, set `num_workers` on the `DataLoader` to >0, and hybridize your network if you're not debugging. You should increase `batch_size` where possible, but do this with care because of its potential effects on convergence. And finally, consider mixed precision training for substantial speed-ups.

## Recommended Next Steps

* Use the [MXNet Profiler](https://mxnet.incubator.apache.org/versions/master/tutorials/python/profiler.html) to identify additional bottlenecks and other areas for optimization.
* Check out the [hybridization tutorial](https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/hybrid.html) for more details on how to write custom `HybridBlock`s.
* Consider using [multi-GPU](https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/multi_gpu.html) and [multi-host](https://github.com/apache/incubator-mxnet/tree/master/example/distributed_training) training if you reach the limits of single GPU training.

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
