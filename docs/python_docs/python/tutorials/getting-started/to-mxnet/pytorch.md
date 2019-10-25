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

# PyTorch vs Apache MXNet

[PyTorch](https://pytorch.org/) is a popular deep learning framework due to its easy-to-understand API and its completely imperative approach. Apache MXNet includes the Gluon API which gives you the simplicity and flexibility of PyTorch and allows you to hybridize your network to leverage performance optimizations of the symbolic graph. As of April 2019, [NVidia performance benchmarks](https://developer.nvidia.com/deep-learning-performance-training-inference) show that Apache MXNet outperforms PyTorch by ~77% on training ResNet-50: 10,925 images per second vs. 6,175.

In the next 10 minutes, we'll do a quick comparison between the two frameworks and show how small the learning curve can be when switching from PyTorch to Apache MXNet. 

## Installation

PyTorch uses conda for installation by default, for example:

```{.python .input}
# !conda install pytorch-cpu -c pytorch
```

For MXNet we use pip:

```{.python .input}
# !pip install mxnet
```

To install Apache MXNet with GPU support, you need to specify CUDA version. For example, the snippet below will install Apache MXNet with CUDA 9.2 support:

```{.python .input}
# !pip install mxnet-cuda92
```

## Data manipulation

Both PyTorch and Apache MXNet relies on multidimensional matrices as a data sources. While PyTorch follows Torch's naming convention and refers to multidimensional matrices as "tensors", Apache MXNet follows NumPy's conventions and refers to them as "NDArrays".

In the code snippets below, we create a two-dimensional matrix where each element is initialized to 1. We show how to add 1 to each element of matrices and print the results.

**PyTorch:**

```{.python .input}
import torch

x = torch.ones(5,3)
y = x + 1
y
```

**MXNet:**

```{.python .input}
from mxnet import nd

x = nd.ones((5,3))
y = x + 1
y
```

The main difference apart from the package name is that the MXNet's shape input parameter needs to be passed as a tuple enclosed in parentheses as in NumPy.

Both frameworks support multiple functions to create and manipulate tensors / NDArrays. You can find more of them in the documentation.

## Model training

After covering the basics of data creation and manipulation, let's dive deep and compare how model training is done in both frameworks. In order to do so, we are going to solve image classification task on MNIST data set using Multilayer Perceptron (MLP) in both frameworks. We divide the task in 4 steps.

### 1. Read data

The first step is to obtain the data. We download the MNIST data set from the web and load it into memory so that we can read batches one by one.

**PyTorch:**

```{.python .input}
from torchvision import datasets, transforms

trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.13,), (0.31,))])
pt_train_data = torch.utils.data.DataLoader(datasets.MNIST(
    root='.', train=True, download=True, transform=trans),
    batch_size=128, shuffle=True, num_workers=4)
```

**MXNet:**

```{.python .input}
from mxnet import gluon
from mxnet.gluon.data.vision import datasets, transforms

trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(0.13, 0.31)])
mx_train_data = gluon.data.DataLoader(
    datasets.MNIST(train=True).transform_first(trans),
    batch_size=128, shuffle=True, num_workers=4)
```

Both frameworks allows you to download MNIST data set from their sources and specify that only training part of the data set is required.

The main difference between the code snippets is that MXNet uses [transform_first](https://mxnet.apache.org/api/python/docs/api/gluon/_autogen/mxnet.gluon.data.Dataset.html) method to indicate that the data transformation is done on the first element of the data batch, the MNIST picture, rather than the second element, the label.

### 2. Creating the model

Below we define a Multilayer Perceptron (MLP) with a single hidden layer
and 10 units in the output layer.

**PyTorch:**

```{.python .input}
import torch.nn as pt_nn

pt_net = pt_nn.Sequential(
    pt_nn.Linear(28*28, 256),
    pt_nn.ReLU(),
    pt_nn.Linear(256, 10))
```

**MXNet:**

```{.python .input}
import mxnet.gluon.nn as mx_nn

mx_net = mx_nn.Sequential()
mx_net.add(mx_nn.Dense(256, activation='relu'),
           mx_nn.Dense(10))
mx_net.initialize()
```

We used the Sequential container to stack layers one after the other in order to construct the neural network. Apache MXNet differs from PyTorch in the following ways:

* In PyTorch you have to specify the input size as the first argument of the `Linear` object. Apache MXNet provides an extra flexibility to network structure by automatically inferring the input size after the first forward pass.

* In Apache MXNet you can specify activation functions directly in fully connected and convolutional layers.

* After the model structure is defined, Apache MXNet requires you to explicitly call the model initialization function.

With a Sequential block, layers are executed one after the other. To have a different execution model, with PyTorch you can inherit from `nn.Module` and then customize how the `.forward()` function is executed. Similarly, in Apache MXNet you can inherit from [nn.Block](https://mxnet.apache.org/api/python/docs/api/gluon/mxnet.gluon.nn.Block.html) to achieve similar results.

### 3. Loss function and optimization algorithm

The next step is to define the loss function and pick an optimization algorithm. Both PyTorch and Apache MXNet provide multiple options to chose from, and for our particular case we are going to use the cross-entropy loss function and the Stochastic Gradient Descent (SGD) optimization algorithm.

**PyTorch:**

```{.python .input}
pt_loss_fn = pt_nn.CrossEntropyLoss()
pt_trainer = torch.optim.SGD(pt_net.parameters(), lr=0.1)
```

**MXNet:**

```{.python .input}
mx_loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
mx_trainer = gluon.Trainer(mx_net.collect_params(),
                           'sgd', {'learning_rate': 0.1})
```

The code difference between frameworks is small. The main difference is that in Apache MXNet we use [Trainer](/api/python/docs/api/gluon/trainer.html) class, which accepts optimization algorithm as an argument. We also use [.collect_params()](/api/python/docs/api/gluon/block.html#mxnet.gluon.Block.collect_params) method to get parameters of the network.

### 4. Training

Finally, we implement the training algorithm. Note that the results for each run
may vary because the weights will get different initialization values and the
data will be read in a different order due to shuffling.

**PyTorch:**

```{.python .input}
import time

for epoch in range(5):
    total_loss = .0
    tic = time.time()
    for X, y in pt_train_data:
        pt_trainer.zero_grad()
        loss = pt_loss_fn(pt_net(X.view(-1, 28*28)), y)
        loss.backward()
        pt_trainer.step()
        total_loss += loss.mean()
    print('epoch %d, avg loss %.4f, time %.2f' % (
        epoch, total_loss/len(pt_train_data), time.time()-tic))
```

**MXNet:**

```{.python .input}
from mxnet import autograd

for epoch in range(5):
    total_loss = .0
    tic = time.time()
    for X, y in mx_train_data:
        with autograd.record():
            loss = mx_loss_fn(mx_net(X), y)
        loss.backward()
        mx_trainer.step(batch_size=128)
        total_loss += loss.mean().asscalar()
    print('epoch %d, avg loss %.4f, time %.2f' % (
        epoch, total_loss/len(mx_train_data), time.time()-tic))
```

Some of the differences in Apache MXNet when compared to PyTorch are as follows:

* In Apache MXNet, you don't need to flatten the 4-D input into 2-D when feeding the data into forward pass.

* In Apache MXNet, you need to perform the calculation within the [autograd.record()](/api/python/docs/api/autograd/index.html?autograd%20record#mxnet.autograd.record) scope so that it can be automatically differentiated in the backward pass.

* It is not necessary to clear the gradient every time as with PyTorch's `trainer.zero_grad()` because by default the new gradient is written in, not accumulated.

* You need to specify the update step size (usually batch size) when performing [step()](/api/python/docs/api/gluon/trainer.html?#mxnet.gluon.Trainer.step) on the trainer.

* You need to call [.asscalar()](/api/python/docs/api/ndarray/ndarray.html?#mxnet.ndarray.NDArray.asscalar) to turn a multidimensional array into a scalar.

* In this sample, Apache MXNet is twice as fast as PyTorch. Though you need to be cautious with such toy comparisons.

## Conclusion

As we saw above, Apache MXNet Gluon API and PyTorch have many similarities. The main difference lies in terminology (Tensor vs. NDArray) and behavior of accumulating gradients: gradients are accumulated in PyTorch and overwritten in Apache MXNet. The rest of the code is very similar, and it is quite straightforward to move code from one framework to the other.

## Recommended Next Steps

While Apache MXNet Gluon API is very similar to PyTorch, there are some extra functionality that can make your code even faster.

* Check out [Hybridize tutorial](/api/python/docs/tutorials/packages/gluon/blocks/hybridize.html) to learn how to write imperative code which can be converted to symbolic one.

* Also, check out how to extend Apache MXNet with your own [custom layers](/api/python/docs/tutorials/packages/gluon/blocks/custom-layer.html?custom_layers).

## Appendix

Below you can find a detailed comparison of various PyTorch functions and their equivalent in Gluon API of Apache MXNet.

### Tensor operation

Here is the list of function names in PyTorch Tensor that are different from Apache MXNet NDArray.

| Function                      | PyTorch                                   | MXNet Gluon                                               |
|-------------------------------|-------------------------------------------|-----------------------------------------------------------|
| Element-wise inverse cosine   | `x.acos()` or `torch.acos(x)`             | `nd.arccos(x)`                                            |
| Batch Matrix product and accumulation| `torch.addbmm(M, batch1, batch2)`  | `nd.linalg_gemm(M, batch1, batch2)` Leading n-2 dim are reduced |
| Element-wise division of t1, t2, multiply v, and add t | `torch.addcdiv(t, v, t1, t2)` | `t + v*(t1/t2)`                              |
| Matrix product and accumulation| `torch.addmm(M, mat1, mat2)`             | `nd.linalg_gemm(M, mat1, mat2)`                           |
| Outer-product of two vector add a matrix | `m.addr(vec1, vec2)`           | Not available                                             |
| Element-wise applies function | `x.apply_(calllable)`                     | Not available, but there is `nd.custom(x, 'op')`          |
| Element-wise inverse sine     | `x.asin()` or `torch.asin(x)`             | `nd.arcsin(x)`                                            |
| Element-wise inverse tangent  | `x.atan()` or `torch.atan(x)`             | `nd.arctan(x)`                                            |
| Tangent of two tensor         | `x.atan2(y)` or `torch.atan2(x, y)`       | Not available                                             |
| batch matrix product          | `x.bmm(y)` or `torch.bmm(x, x)`           | `nd.linalg_gemm2(x, y)`                                   |
| Draws a sample from bernoulli distribution | `x.bernoulli()`              | Not available                                             |
| Fills a tensor with number drawn from Cauchy distribution | `x.cauchy_()` | Not available                                             |
| Splits a tensor in a given dim| `x.chunk(num_of_chunk)`                   | `nd.split(x, num_outputs=num_of_chunk)`                   |
| Limits the values of a tensor to between min and max | `x.clamp(min, max)`| `nd.clip(x, min, max)`                                    |
| Returns a copy of the tensor  | `x.clone()`                               | `x.copy()`                                                |
| Cross product                 | `x.cross(y)`                              | Not available                                             |
| Cumulative product along an axis| `x.cumprod(1)`                          | Not available                                             |
| Cumulative sum along an axis  | `x.cumsum(1)`                             | Not available                                             |
| Address of the first element  | `x.data_ptr()`                            | Not available                                             |
| Creates a diagonal tensor     | `x.diag()`                                | Not available                                             |
| Computes norm of a tensor     | `x.dist()`                                | `nd.norm(x)` Only calculate L2 norm                       |
| Computes Gauss error function | `x.erf()`                                 | Not available                                             |
| Broadcasts/Expands tensor to new shape | `x.expand(3,4)`                  | `x.broadcast_to([3, 4])`                                  |
| Fills a tensor with samples drawn from exponential distribution | `x.exponential_()` | `nd.random_exponential()`                      |
| Element-wise mod              | `x.fmod(3)`                               | `nd.module(x, 3)`                                         |
| Fractional portion of a tensor| `x.frac()`                                | `x - nd.trunc(x)`                                         |
| Gathers values along an axis specified by dim | `torch.gather(x, 1,  torch.LongTensor([[0,0],[1,0]]))` | `nd.gather_nd(x, nd.array([[[0,0],[1,1]],[[0,0],[1,0]]]))`  |
| Solves least square & least norm | `B.gels(A)`                            | Not available                                             |
| Draws from geometirc distribution | `x.geometric_(p)`                     | Not available                                             |
| Device context of a tensor    | `print(x)` will print which device x is on| `x.context`                                               |
| Repeats tensor                | `x.repeat(4,2)`                           | `x.tile(4,2)`                                             |
| Data type of a tensor         | `x.type()`                                | `x.dtype`                                                 |
| Scatter                       | `torch.zeros(2, 4).scatter_(1, torch.LongTensor([[2], [3]]), 1.23)` | `nd.scatter_nd(nd.array([1.23,1.23]), nd.array([[0,1],[2,3]]), (2,4))` |
| Returns the shape of a tensor | `x.size()`                                | `x.shape`                                                 |
| Number of elements in a tensor| `x.numel()`                               | `x.size`                                                  |
| Returns this tensor as a NumPy ndarray | `x.numpy()`                      | `x.asnumpy()`                                             |
| Eigendecomposition for symmetric matrix | `e, v = a.symeig()`             | `v, e = nd.linalg.syevd(a)`                               |
| Transpose                     | `x.t()`                                   | `x.T`                                                     |
| Sample uniformly              | `torch.uniform_()`                        | `nd.sample_uniform()`                                     |
| Inserts a new dimesion        | `x.unsqueeze()`                           | `nd.expand_dims(x)`                                       |
| Reshape                       | `x.view(16)`                              | `x.reshape((16,))`                                          |
| Veiw as a specified tensor    | `x.view_as(y)`                            | `x.reshape_like(y)`                                       |
| Returns a copy of the tensor after casting to a specified type | `x.type(type)` | `x.astype(dtype)`                                   |
| Copies the value of one tensor to another | `dst.copy_(src)`              | `src.copyto(dst)`                                         |
| Returns a zero tensor with specified shape | `x = torch.zeros(2,3)`       | `x = nd.zeros((2,3))`                                     |
| Returns a one tensor with specified shape | `x = torch.ones(2,3)`         | `x = nd.ones((2,3)`                                       |
| Returns a Tensor filled with the scalar value 1, with the same size as input | `y = torch.ones_like(x)` | `y = nd.ones_like(x)`       |

### Functional

### GPU

Just like Tensor, MXNet NDArray can be copied to and operated on GPU. This is done by specifying context.

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
| Copy to GPU            | `y = torch.FloatTensor(1).cuda()` | `y = mx.nd.ones((1,), ctx=mx.gpu(0))`                                      |
| Convert to numpy array | `x = y.cpu().numpy()`             | `x = y.asnumpy()`                                                          |
| Context scope          | `with torch.cuda.device(1):`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`y= torch.cuda.FloatTensor(1)`                    | `with mx.gpu(1):`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`y = mx.nd.ones((3,5))`      |

### Cross-device

Just like Tensor, MXNet NDArray can be copied across multiple GPUs.

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
| Copy from GPU 0 to GPU 1           | `x = torch.cuda.FloatTensor(1)`<br/>`y=x.cuda(1)`| `x = mx.nd.ones((1,), ctx=mx.gpu(0))`<br/>`y=x.as_in_context(mx.gpu(1))`                                      |
| Copy Tensor/NDArray on different GPUs | `y.copy_(x)`             | `x.copyto(y)`                                                          |

## Autograd

### Variable wrapper vs autograd scope

Autograd package of PyTorch/MXNet enables automatic differentiation of Tensor/NDArray.

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
| Recording computation       | `x = Variable(torch.FloatTensor(1), requires_grad=True)`<br/>`y = x * 2`<br/>`y.backward()`  | `x = mx.nd.ones((1,))`<br/>`x.attach_grad()`<br/>`with mx.autograd.record():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`y = x * 2`<br/>`y.backward()`                                   |

### Scope override (pause, train_mode, predict_mode)

Some operators (Dropout, BatchNorm, etc) behave differently in training and making predictions. This can be controlled with `train_mode` and `predict_mode` scope in MXNet.
Pause scope is for code that does not need gradients to be calculated.

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
| Scope override   | Not available | `x = mx.nd.ones((1,))`<br/>`with autograd.train_mode():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`y = mx.nd.Dropout(x)`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`with autograd.predict_mode():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`z = mx.nd.Dropout(y)`<br/><br/>`w = mx.nd.ones((1,))`<br/>`w.attach_grad()`<br/>`with autograd.record():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`y = x * w`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`y.backward()`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`with autograd.pause():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`w += w.grad`   |

### Batch-end synchronization is needed

Apache MXNet uses lazy evaluation to achieve superior performance. The Python thread just pushes the operations into the backend engine and then returns. In training phase batch-end synchronization is needed, e.g, `asnumpy()`, `wait_to_read()`, `metric.update(...)`.

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
| Batch-end synchronization    |  Not available  | `for (data, label) in train_data:`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`with autograd.record():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`output = net(data)`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`L = loss(output, label)`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`L.backward()`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`trainer.step(data.shape[0])`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`metric.update([label], [output])` |

## PyTorch module and Gluon blocks

### For new block definition, gluon needs name_scope

`name_scope` coerces Gluon to give each parameter an appropriate name, indicating which model it belongs to.

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
| New block definition   | `class Net(torch.nn.Module):`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`def __init__(self, D_in, D_out):`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`super(Net, self).__init__()`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`self.linear = torch.nn.Linear(D_in, D_out)`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`def forward(self, x):`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`return self.linear(x)`       |    `class Net(mx.gluon.Block):`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`def __init__(self, D_in, D_out):`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`super(Net, self).__init__()`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`with self.name_scope():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`self.dense=mx.gluon.nn.Dense(D_out, in_units=D_in)`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`def forward(self, x):`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`return self.dense(x)`      |

### Parameter and Initializer

When creating new layers in PyTorch, you do not need to specify its parameter initializer, and different layers have different default initializer. When you create new layers in Gluon API, you can specify its initializer or just leave it none. The parameters will finish initializing after calling `net.initialize(<init method>)` and all parameters will be initialized in `init method` except those layers whose initializer specified.

| Function       | PyTorch           | MXNet Gluon        |
|----------------|-------------------|--------------------|
| Get all parameters |  `net.parameters()` | `net.collect_params()` |
| Initialize network |  Not Available | `net.initialize(mx.init.Xavier())` |
| Specify layer initializer | `layer = torch.nn.Linear(20, 10)`<br/> `torch.nn.init.normal(layer.weight, 0, 0.01)` | `layer = mx.gluon.nn.Dense(10, weight_initializer=mx.init.Normal(0.01))` |

### Usage of existing blocks look alike

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
| Usage of existing blocks    |  `y=net(x)`  |  `y=net(x)`   |

### HybridBlock can be hybridized, and allows partial-shape info

HybridBlock supports forwarding with both Symbol and NDArray. After hybridized, HybridBlock will create a symbolic graph representing the forward computation and cache it. Most of the built-in blocks (Dense, Conv2D, MaxPool2D, BatchNorm, etc.) are HybridBlocks.

Instead of explicitly declaring the number of inputs to a layer, we can simply state the number of outputs. The shape will be inferred on the fly once the network is provided with some input.

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
| partial-shape  <br/> hybridized    |  Not Available   |  `net = mx.gluon.nn.HybridSequential()`<br/>`with net.name_scope():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`net.add(mx.gluon.nn.Dense(10))`<br/>`net.hybridize()`   |

### SymbolBlock

SymbolBlock can construct block from symbol. This is useful for using pre-trained models as feature extractors.

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
|  SymbolBlock    |  Not Available   |  `alexnet = mx.gluon.model_zoo.vision.alexnet(pretrained=True, prefix='model_')`<br/>`out = alexnet(inputs)`<br/>`internals = out.get_internals()`<br/>`outputs = [internals['model_dense0_relu_fwd_output']]`<br/>`feat_model = gluon.SymbolBlock(outputs, inputs, params=alexnet.collect_params())`   |

## PyTorch optimizer vs Gluon Trainer
### For Gluon API calling zero_grad is not necessary most of the time
`zero_grad` in optimizer (PyTorch) or Trainer (Gluon API) clears the gradients of all parameters. In Gluon API, there is no need to clear the gradients every batch if `grad_req = 'write'`(default).

| Function               | PyTorch                           | MXNet Gluon                              |
|------------------------|-----------------------------------|------------------------------------------|
| clear the gradients |   `optm = torch.optim.SGD(model.parameters(), lr=0.1)`<br/>`optm.zero_grad()`<br/>`loss_fn(model(input), target).backward()`<br/>`optm.step()`    | `trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})`<br/>`with autograd.record():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`loss = loss_fn(net(data), label)`<br/>`loss.backward()`<br/>`trainer.step(batch_size)`      |

### Multi-GPU training

| Function               | PyTorch                           | MXNet Gluon                              |
|------------------------|-----------------------------------|------------------------------------------|
| data parallelism |   `net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])`<br/>`output = net(data)`    | `ctx = [mx.gpu(i) for i in range(3)]`<br/>`data = gluon.utils.split_and_load(data, ctx)`<br/>`label = gluon.utils.split_and_load(label, ctx)`<br/>`with autograd.record():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`losses = [loss(net(X), Y) for X, Y in zip(data, label)]`<br/>`for l in losses:`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`l.backward()`      |

### Distributed training

| Function               | Pytorch                           | MXNet Gluon                              |
|------------------------|-----------------------------------|------------------------------------------|
| distributed data parallelism |   `torch.distributed.init_process_group(...)`<br/>`model = torch.nn.parallel.distributedDataParallel(model, ...)`    | `store = kv.create('dist')`<br/>`trainer = gluon.Trainer(net.collect_params(), ..., kvstore=store)`  |

## Monitoring

### Apache MXNet has pre-defined metrics

Gluon provide several predefined metrics which can online evaluate the performance of a learned model.

| Function               | PyTorch                           | MXNet Gluon                              |
|------------------------|-----------------------------------|------------------------------------------|
| metric |  Not available   | `metric = mx.metric.Accuracy()`<br/>`with autograd.record():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`output = net(data)`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`L = loss(ouput, label)`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`loss(ouput, label).backward()`<br/>`trainer.step(batch_size)`<br/>`metric.update(label, output)`  |

### Data visualization

TensorboardX (PyTorch) and [MXBoard](https://github.com/awslabs/mxboard) (MXNet) can be used to visualize your network and plot quantitative metrics about the execution of your graph.

| PyTorch                                        | MXNet                                          |
| ---------------------------------------------- | ---------------------------------------------- |
| `sw = tensorboardX.SummaryWriter()`            | `sw = mxboard.SummaryWriter()`                 |
| `...`                                          | `...`                                          |
| `for name, param in model.named_parameters():` | `for name, param in net.collect_params():`     |
| `    grad = param.clone().cpu().data.numpy()`  | `    grad = param.grad.asnumpy().flatten()`    |
| `    sw.add_histogram(name, grad, n_iter)`     | `    sw.add_histogram(tag=str(param),`         |
| `...`                                          | `       values=grad,`                          |
| `sw.close()`                                   | `       bins=200,`                             |
|                                                | `       global_step=i)`                        |
|                                                | `...`                                          |
|                                                | `sw.close()`                                   |

## I/O and deploy

### Data loading

`Dataset` and `DataLoader` are the basic components for loading data.

| Class               | PyTorch                           | MXNet Gluon                              |
|------------------------|-----------------------------------|------------------------------------------|
| Dataset holding arrays | `torch.utils.data.TensorDataset(data_tensor, label_tensor)`| `gluon.data.ArrayDataset(data_array, label_array)`                        |
| Data loader | `torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>, drop_last=False)` | `gluon.data.DataLoader(dataset, batch_size=None, shuffle=False, sampler=None, last_batch='keep', batch_sampler=None, batchify_fn=None, num_workers=0)`|
| Sequentially applied sampler | `torch.utils.data.sampler.SequentialSampler(data_source)` | `gluon.data.SequentialSampler(length)` |
| Random order sampler | `torch.utils.data.sampler.RandomSampler(data_source)` | `gluon.data.RandomSampler(length)`|

Some commonly used datasets for computer vision are provided in `mx.gluon.data.vision` package.

| Class               | PyTorch                           | MXNet Gluon                              |
|------------------------|-----------------------------------|------------------------------------------|
| MNIST handwritten digits dataset. | `torchvision.datasets.MNIST`| `mx.gluon.data.vision.MNIST` |
| CIFAR10 Dataset. | `torchvision.datasets.CIFAR10` | `mx.gluon.data.vision.CIFAR10`|
| CIFAR100 Dataset. | `torchvision.datasets.CIFAR100` | `mx.gluon.data.vision.CIFAR100` |
| A generic data loader where the images are arranged in folders. | `torchvision.datasets.ImageFolder(root, transform=None, target_transform=None, loader=<function default_loader>)` | `mx.gluon.data.vision.ImageFolderDataset(root, flag, transform=None)`|

### Serialization

Serialization and deserialization are achieved by calling `save_parameters` and `load_parameters`.

| Class               | PyTorch                           | MXNet Gluon                              |
|------------------------|-----------------------------------|------------------------------------------|
| Save model parameters | `torch.save(the_model.state_dict(), filename)`| `model.save_parameters(filename)`|
| Load parameters | `the_model.load_state_dict(torch.load(PATH))` | `model.load_parameters(filename, ctx, allow_missing=False, ignore_extra=False)` |
