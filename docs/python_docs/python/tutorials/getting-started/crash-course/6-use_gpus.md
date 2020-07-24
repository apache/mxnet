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

# Step 6: Use GPUs to increase efficiency

In this step, you learn how to use graphics processing units (GPUs) with MXNet. If you use GPUs to train and deploy neural networks, you get significantly more computational power when compared to central processing units (CPUs).

## Prerequisites

Before you start the other steps here, make sure you have at least one Nvidia GPU in your machine and CUDA properly installed. GPUs from AMD and Intel are not supported. Install the GPU-enabled version of MXNet.

Use the following commands to check the number GPUs that are available.

```{.python .input  n=2}
from mxnet import np, npx, gluon, autograd
from mxnet.gluon import nn
import time
npx.set_np()

npx.num_gpus()
```

## Allocate data to a GPU

MXNet's ndarray is very similar to NumPy. One major difference is MXNet's ndarray has a `context` attribute that specifies which device an array is on. By default, it is on `npx.cpu()`. Change it to the first GPU with the following code. Use `npx.gpu()` or `npx.gpu(0)` to indicate the first GPU.

```{.python .input  n=10}
gpu = npx.gpu() if npx.num_gpus() > 0 else npx.cpu()
x = np.ones((3,4), ctx=gpu)
x
```

If you're using a CPU, MXNet allocates data on main memory and tries to use as many CPU cores as possible.  This is true even if there is more than one CPU socket. If there are multiple GPUs, MXNet specifies which GPUs the ndarray is allocated.

Assume there is a least one more GPU. Create another ndarray and assign it there. If you only have one GPU, then you get an error. In the example code here, you copy `x` to the second GPU, `npx.gpu(1)`:

```{.python .input  n=11}
gpu_1 = npx.gpu(1) if npx.num_gpus() > 1 else npx.cpu()
x.copyto(gpu_1)
```

MXNet requries that users explicitly move data between devices. But several operators such as `print`, and `asnumpy`, will implicitly move data to main memory.

## Run an operation on a GPU

To perform an operation on a particular GPU, you only need to guarantee that the input of an operation is already on that GPU. The output is allocated on the same GPU as well. Almost all operators in the `np` and `npx` module support running on a GPU.

```{.python .input  n=21}
y = np.random.uniform(size=(3,4), ctx=gpu)
x + y
```

Remember that if the inputs are not on the same GPU, you get an error.

## Run a neural network on a GPU

To run a neural network on a GPU, you only need to copy and move the input data and parameters to the GPU. Reuse the previously defined LeNet. The following code example shows this.

```{.python .input  n=16}
net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation="relu"),
        nn.Dense(84, activation="relu"),
        nn.Dense(10))
```

Load the saved parameters into GPU 0 directly as shown here, or use `net.collect_params().reset_ctx` to change the device.

```{.python .input  n=20}
net.load_parameters('net.params', ctx=gpu)
```

Use the following command to create input data on GPU 0. The forward function will then run on GPU 0.

```{.python .input  n=22}
# x = np.random.uniform(size=(1,1,28,28), ctx=gpu)
# net(x) FIXME
```

## Training with multiple GPUs

Finally, you can see how to use multiple GPUs to jointly train a neural network through data parallelism. Assume there are *n* GPUs. Split each data batch into *n* parts, and then each GPU will run the forward and backward passes using one part of the data.

First copy the data definitions with the following commands, and the transform function from the [Predict tutorial](5-predict.md).

```{.python .input}
batch_size = 256
transformer = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize(0.13, 0.31)])
train_data = gluon.data.DataLoader(
    gluon.data.vision.datasets.FashionMNIST(train=True).transform_first(
        transformer), batch_size, shuffle=True, num_workers=4)
valid_data = gluon.data.DataLoader(
    gluon.data.vision.datasets.FashionMNIST(train=False).transform_first(
        transformer), batch_size, shuffle=False, num_workers=4)
```

The training loop is quite similar to that shown earlier. The major differences are highlighted in the following code.

```{.python .input}
# Diff 1: Use two GPUs for training.
devices = [gpu, gpu_1]
# Diff 2: reinitialize the parameters and place them on multiple GPUs
net.collect_params().initialize(force_reinit=True, ctx=devices)
# Loss and trainer are the same as before
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
for epoch in range(10):
    train_loss = 0.
    tic = time.time()
    for data, label in train_data:
        # Diff 3: split batch and load into corresponding devices
        data_list = gluon.utils.split_and_load(data, devices)
        label_list = gluon.utils.split_and_load(label, devices)
        # Diff 4: run forward and backward on each devices.
        # MXNet will automatically run them in parallel
        with autograd.record():
            losses = [softmax_cross_entropy(net(X), y)
                      for X, y in zip(data_list, label_list)]
        for l in losses:
            l.backward()
        trainer.step(batch_size)
        # Diff 5: sum losses over all devices. Here float will copy data
        # into CPU.
        train_loss += sum([float(l.sum()) for l in losses])
    print("Epoch %d: loss %.3f, in %.1f sec" % (
        epoch, train_loss/len(train_data)/batch_size, time.time()-tic))
```

## Next steps

Now you have completed training and predicting with a neural network by using NP on MXNet and
Gluon. You can check the guides to these two front ends: [What is NP on MXNet](../deepnumpy/index.html) and [gluon](../gluon_from_experiment_to_deployment.md).
