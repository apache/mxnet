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

# Profiling MXNet Models

It is often helpful to check the execution time of each operation in a neural network. You can then determine where to focus your effort to speed up model training or inference. In this tutorial, we will learn how to profile MXNet models to measure their running time and memory consumption using the MXNet profiler.

## The incorrect way to profile

If you have just started to use MXNet, you might be tempted to measure the execution time of your model using Python's `time` module like shown below:

```python
from time import time
from mxnet import autograd, nd
import mxnet as mx

start = time()
x = nd.random_uniform(shape=(2000,2000))
y = nd.dot(x, x)
print('Time for matrix multiplication: %f sec\n' % (time() - start))

start = time()                                
y_np = y.asnumpy()                             
print('Time for converting to numpy: %f sec' % (time() - start))
```

**Time for matrix multiplication: 0.005051 sec**<!--notebook-skip-line-->

**Time for converting to numpy: 0.167693 sec**<!--notebook-skip-line-->

From the timings above, it seems as if converting to numpy takes lot more time than multiplying two large matrices. That doesn't seem right.

This is because, in MXNet, all operations are executed asynchronously. So, when `nd.dot(x, x)` returns, the matrix multiplication is not complete, it has only been queued for execution. However, [`asnumpy`](http://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html?highlight=asnumpy#mxnet.ndarray.NDArray.asnumpy) has to wait for the result to be calculated in order to convert it to numpy array on CPU, hence taking a longer time. Other examples of 'blocking' operations include [`asscalar`](http://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html?highlight=asscalar#mxnet.ndarray.NDArray.asscalar) and [`wait_to_read`](http://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html?highlight=wait_to_read#mxnet.ndarray.NDArray.wait_to_read).

While it is possible to use [`NDArray.waitall()`](http://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html?highlight=waitall#mxnet.ndarray.waitall) before and after operations to get running time of operations, it is not a scalable method to measure running time of multiple sets of operations, especially in a [`Sequential`](http://mxnet.incubator.apache.org/api/python/gluon/gluon.html?highlight=sequential#mxnet.gluon.nn.Sequential) or hybridized network.

## The correct way to profile

The correct way to measure running time of MXNet models is to use MXNet profiler. In the rest of this tutorial, we will learn how to use the MXNet profiler to measure the running time and memory consumption of MXNet models. You can import the profiler and configure it from Python code.

```python
from mxnet import profiler

profiler.set_config(profile_all=True,
                    aggregate_stats=True,
                    filename='profile_output.json')
```

`profile_all` enables all types of profiling. You can also individually enable the following types of profiling:

- `profile_symbolic` (boolean): whether to profile symbolic operators
- `profile_imperative` (boolean): whether to profile imperative operators
- `profile_memory` (boolean): whether to profile memory usage
- `profile_api` (boolean): whether to profile the C API

`aggregate_stats` aggregates statistics in memory which can then be printed to console by calling `profiler.dumps()`.

### Setup: Build a model

Let's build a small convolutional neural network that we can use to demonstrate profiling.

```python
from mxnet import gluon

net = gluon.nn.HybridSequential()
with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(512, activation="relu"))
    net.add(gluon.nn.Dense(10))
```

We need data that we can run through the network for profiling. We'll use the MNIST dataset.

```python
from mxnet.gluon.data.vision import transforms

dataset = gluon.data.vision.MNIST(train=True)
dataset = dataset.transform_first(transforms.ToTensor())
dataloader = gluon.data.DataLoader(dataset, batch_size=64, shuffle=True)
```

Let's define a function that will run a single training iteration given `data` and `label`.

```python
# Use GPU if available
if mx.context.num_gpus():
    ctx=mx.gpu()
else:
    ctx=mx.cpu()

# Initialize the parameters with random weights
net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

# Use SGD optimizer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

# Softmax Cross Entropy is a frequently used loss function for multi-class classification
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# A helper function to run one training iteration
def run_training_iteration(data, label):
    # Load data and label is the right context
    data = data.as_in_context(ctx)
    label = label.as_in_context(ctx)
    # Run the forward pass
    with autograd.record():
        output = net(data)
        loss = softmax_cross_entropy(output, label)
    # Run the backward pass
    loss.backward()
    # Apply changes to parameters
    trainer.step(data.shape[0])
```

### Starting and stopping the profiler from Python

When the first forward pass is run on a network, MXNet does a number of housekeeping tasks including inferring the shapes of various parameters, allocating memory for intermediate and final outputs, etc. For these reasons, profiling the first iteration doesn't provide representative results for the rest of training. We will, therefore, skip the first iteration.

```python
# Run the first iteration without profiling
itr = iter(dataloader)
run_training_iteration(*next(itr))
```

We'll run the next iteration with the profiler turned on.

```python
data, label = next(itr)

# Ask the profiler to start recording
profiler.set_state('run')

run_training_iteration(*next(itr))

# Make sure all operations have completed
mx.nd.waitall()
# Ask the profiler to stop recording
profiler.set_state('stop')
```

Between running and stopping the profiler, you can also pause and resume the profiler using `profiler.pause()` and `profiler.resume()` respectively to profile only parts of the code you want to profile.

### Starting the profiler automatically using an environment variable

The method described above requires code changes to start and stop the profiler. You can also start the profiler automatically and profile the entire code without any code changes using the `MXNET_PROFILER_AUTOSTART` environment variable.

`$ MXNET_PROFILER_AUTOSTART=1 python my_script.py`

MXNet will start the profiler automatically if you run your code with the environment variable `MXNET_PROFILER_AUTOSTART` set to `1`. The profiler output is stored in `profile.json` inside the current directory.

Note that the profiler output could be large depending on your code. It might be helpful to profile only sections of your code using the `set_state` API described in the previous section.

### Increasing granularity of the profiler output

MXNet executes computation graphs in 'bulk mode' which reduces kernel launch gaps in between symbolic operators for faster execution. This could reduce the granularity of the profiler output. If you need profiling result of every operator, please set the environment variables `MXNET_EXEC_BULK_EXEC_INFERENCE` and `MXNET_EXEC_BULK_EXEC_TRAIN` to `0` to disable the bulk execution mode.

When working with networks created using the Gluon API, you will get a more granular profiling outputs if you profile networks that haven't been hybridized. Operations can appear fused together in the profiling outputs after hybridization, which can make debugging tricky.

### Viewing profiler output

There are a few ways to view the information collected by the profiler. You can view it in the console, you can view a more graphical version in a browser, or you can use a vendor tool such as Intel VTune or Nvidia NVProf to view output. For most scenarios the information you need can be obtained with MXNet's built in profiler support, but if you want to investigate the performance of operators alongside extra context about your hardware (e.g. cache hit rates, or CUDA kernel timings) then profiling jointly with vendor tools is recommended.

#### 1. View in console

You can use the `profiler.dumps()` method to view the information collected by the profiler in the console. The collected information contains time taken by each operator, time taken by each C API and memory consumed in both CPU and GPU.

```python
print(profiler.dumps())
```

![Profile Statistics](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/python/profiler/profile_stats.png)<!--notebook-skip-line-->

#### 2. View in browser

You can also dump the information collected by the profiler into a `json` file using the `profiler.dump()` function and view it in a browser.

```python
profiler.dump()
```

`dump()` creates a `json` file which can be viewed using a trace consumer like `chrome://tracing` in the Chrome browser. Here is a snapshot that shows the output of the profiling we did above.

![Tracing Screenshot](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/python/profiler/profiler_output_chrome.png)

Let's zoom in to check the time taken by operators

![Operator profiling](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/python/profiler/profile_operators.png)

The above picture visualizes the sequence in which the operators were executed and the time taken by each operator.

## Advanced: Using NVIDIA Profiling Tools

MXNet's Profiler is the recommended starting point for profiling MXNet code, but NVIDIA also provides a couple of tools for low-level profiling of CUDA code: [NVProf](https://devblogs.nvidia.com/cuda-pro-tip-nvprof-your-handy-universal-gpu-profiler/), [Visual Profiler](https://developer.nvidia.com/nvidia-visual-profiler) and [Nsight Compute](https://developer.nvidia.com/nsight-compute). You can use these tools to profile all kinds of executables, so they can be used for profiling Python scripts running MXNet. And you can use these in conjunction with the MXNet Profiler to see high-level information from MXNet alongside the low-level CUDA kernel information.

#### NVProf and Visual Profiler

NVProf and Visual Profiler are available in CUDA 9 and CUDA 10 toolkits. You can get a timeline view of CUDA kernel executions, and also analyse the profiling results to get automated recommendations. It is useful for profiling end-to-end training but the interface can sometimes become slow and unresponsive.

You can initiate the profiling directly from inside Visual Profiler or from the command line with `nvprof` which wraps the execution of your Python script. If it's not on your path already, you can find `nvprof` inside your CUDA directory. See [this discussion post](https://discuss.mxnet.io/t/using-nvidia-profiling-tools-visual-profiler-and-nsight-compute/) for more details on setup.

`$ nvprof -o my_profile.nvvp python my_profiler_script.py`

`==11588== NVPROF is profiling process 11588, command: python my_profiler_script.py`

`==11588== Generated result file: /home/user/Development/incubator-mxnet/ci/my_profile.nvvp`

We specified an output file called `my_profile.nvvp` and this will be annotated with NVTX ranges (for MXNet operations) that will be displayed alongside the standard NVProf timeline. This can be very useful when you're trying to find patterns between operators run by MXNet, and their associated CUDA kernel calls.

You can open this file in Visual Profiler to visualize the results.

![Operator profiling](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/python/profiler/profiler_nvprof.png)

At the top of the plot we have CPU tasks such as driver operations, memory copy calls, MXNet engine operator invocations, and imperative MXNet API calls.  Below we see the kernels active on the GPU during the same time period.

![Operator profiling](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/python/profiler/profiler_nvprof_zoomed.png)

Zooming in on a backwards convolution operator we can see that it is in fact made up of a number of different GPU kernel calls, including a cuDNN winograd convolution call, and a fast-fourier transform call.

![Operator profiling](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/python/profiler/profiler_winograd.png)

Selecting any of these kernel calls (the winograd convolution call shown here) will get you some interesting GPU performance information such as occupancy rates (vs theoretical), shared memory usage and execution duration.

#### Nsight Compute

Nsight Compute is available in CUDA 10 toolkit, but can be used to profile code running CUDA 9. You don't get a timeline view, but you get many low level statistics about each individual kernel executed and can compare multiple runs (i.e. create a baseline).

![Nsight Compute](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/python/profiler/profile_nsight_compute.png)

### Further reading

- [Examples using MXNet profiler.](https://github.com/apache/incubator-mxnet/tree/master/example/profiler)
- [Some tips for improving MXNet performance.](https://mxnet.incubator.apache.org/faq/perf.html)

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->

