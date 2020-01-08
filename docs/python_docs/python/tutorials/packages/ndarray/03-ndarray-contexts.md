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

# NDArray Contexts

## Overview
This guide will introduce you to managing CPU versus GPU contexts for handling data.

This content was extracted and simplified from the gluon tutorials in
[Dive Into Deep Learning](https://d2l.ai/).

## Prerequisites
* [MXNet installed (with GPU support) in a Python environment](/get_started).
* Python 2.7.x or Python 3.x
* **One or more GPUs**


## Managing Context

In MXNet, every array has a context.
One context could be the CPU. Other contexts might be various GPUs.
Things can get even hairier when we deploy jobs across multiple servers.
By assigning arrays to contexts intelligently, we can minimize
the time spent transferring data between devices.
For example, when training neural networks on a server with a GPU,
we typically prefer for the model's parameters to live on the GPU.
If you have a GPU, let's try initializing an array on the first GPU.
Otherwise, use `ctx=mx.cpu()` in place of `ctx=gpu(0)`.

```{.python .input}
from mxnet import gpu
from mxnet import nd
z = nd.ones(shape=(3,3), ctx=gpu(0))
print(z)
```

Given an NDArray on a given context, we can copy it to another context by using
the copyto() method. Skip this if you don't have a GPU at the moment.

```{.python .input}
x_gpu = x.copyto(gpu(0))
print(x_gpu)
```

The result of an operator will have the same context as the inputs.

```{.python .input}
x_gpu + z
```

## Watch out!

Imagine that your variable z already lives on your second GPU
(`gpu(0)`). What happens if we call `z.copyto(gpu(0))`? It will make a copy and
allocate new memory, even though that variable already lives on the desired
device!
<!-- wouldn't the second GPU be gpu(1)? -->

Often, we only want to make
a copy if the variable currently lives in the wrong context. In these cases, we
can call `as_in_context()`. If the variable is already on `gpu(0)` then this is
a no-op.

```{.python .input}
print('id(z):', id(z))
z = z.copyto(gpu(0))
print('id(z):', id(z))
z = z.as_in_context(gpu(0))
print('id(z):', id(z))
print(z)
```

## Next Up

[Back to NDArray API Guides](.)
