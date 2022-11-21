---
layout: page_category
title:  Exception Handling in Apache MXNet
category: architecture
permalink: /api/architecture/exception_handling
---
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

# Exception Handling in Apache MXNet

This tutorial explains the exception handling support in Apache MXNet,
and provides examples on how to throw and handle exceptions when in a multithreaded context.
Although, the examples are in Python, they can be easily extended to MXNet
language bindings.

MXNet exceptions can be thrown from two areas:
- MXNet main thread. For eg. Infershape and InferType.
- Spawned threads:
    * By dependency engine for operator execution in parallel
    * By the iterators, during the data loading, text parsing phase etc.

In the first case, the exception is thrown and can be handled in the main thread.
In the second case, the exception is thrown in a spawned thread, caught and transported to the
main thread, where it is rethrown. This tutorial will give more explanation and examples on how
to handle exceptions for the second case.

## Prerequisites

To complete this tutorial, we need:
- MXNet [7b24137](https://github.com/apache/mxnet/commit/7b24137ed45df605defa4ce72ec91554f6e445f0). See Instructions in [Setup and Installation](https://mxnet.io/get_started).

## Exception Handling for Iterators

The below example shows how to handle exceptions for iterators. In this example,
we populate files for data and labels with fewer number of labels compared to the
number of samples. This should throw an exception.

CSVIter uses PrefetcherIter for loading and parsing data.
The PrefetcherIter spawns a producer thread in the background which prefetches
the data while the main thread consumes the data. The exception is thrown in the spawned
producer thread during the prefetching, when the label is not found corresponding to a specific sample.

The exception is transported to the main thread, where it is rethrown when Next is
called as part of the following line: `for batch in iter(data_train)`.

In general, Exception may be rethrown as part of `Next` and `BeforeFirst` calls which correspond to `reset()` and `next()` methods in `MXDataIter` for Python language bindings.

```python
import os
import mxnet as mx

cwd = os.getcwd()
data_path = os.path.join(cwd, "data.csv")
label_path = os.path.join(cwd, "label.csv")

with open(data_path, "w") as fout:
    for i in range(8):
        fout.write("1,2,3,4,5,6,7,8,9,10\n")

with open(label_path, "w") as fout:
    for i in range(7):
        fout.write("label"+str(i))

try:
    data_train = mx.io.CSVIter(data_csv=data_path, label_csv=label_path, data_shape=(1, 10),
                               batch_size=4)

    for batch in iter(data_train):
        print(data_train.getdata().asnumpy())
except mx.base.MXNetError as ex:
    print("Exception handled")
    print(ex)
```

### Limitation

There is a race condition when your last `next()` call doesnt reach the batch in your dataset where exception occurs. Exception may or may not be thrown in this case depending on which thread wins the race. To avoid this situation, you should try and iterate through your full dataset if you think it can throw exceptions which need to be handled.


## Exception Handling for Operators

The below example shows how to handle exceptions for operators in the imperative mode.

For the operator case, the dependency engine spawns a number of threads if it is running in the `ThreadedEnginePool` or `ThreadedEnginePerDevice` mode. The final operator is executed in one of the spawned threads.

If an operator throws an exception during execution, this exception is propagated
down the dependency chain. Once there is a synchronizing call i.e. WaitToRead for a variable in the dependency chain, the propagated exception is rethrown.

In the below example, I illustrate how an exception that occured in the first line is propagated down the dependency chain, and finally is rethrown when we make a synchronizing call to WaitToRead.

```python
import mxnet as mx
a = mx.nd.random.normal(0, 1, (2, 2))
b = mx.nd.random.normal(0, 2, (2, 2))
c = mx.nd.dot(a, b)
d = mx.nd.random.normal(0, -1, (2, 2))
e = mx.nd.dot(c, d)
e.wait_to_read()
```

Although the above exception occurs when executing the operation which writes to the variable d in one of the child threads, it is thrown only when the synchronization happens as part of the line: `e.wait_to_read()`.

Let us take another example. In the following case, we write to two variables and then `wait_to_read` for both. This example shows that any particular exception will not be thrown more than once.

```python
import mxnet as mx
a = mx.nd.random.normal(0, 1, (2, 2))
b = mx.nd.random.normal(0, -1, (2, 2))
c, d  = mx.nd.dot(a, b)
try:
    c.asnumpy()
except mx.base.MXNetError as ex:
    print("Exception handled")
d.asnumpy()
```
