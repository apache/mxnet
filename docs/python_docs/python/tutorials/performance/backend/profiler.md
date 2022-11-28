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

```{.python .input}
from time import time
from mxnet import autograd, np
import mxnet as mx

start = time()
x = np.random.uniform(size=(2000,2000))
y = np.dot(x, x)
print('Time for matrix multiplication: %f sec\n' % (time() - start))

start = time()                                
y_np = y.asnumpy()                             
print('Time for converting to numpy: %f sec' % (time() - start))
```

**Time for matrix multiplication: 0.005051 sec**<!--notebook-skip-line-->

**Time for converting to numpy: 0.167693 sec**<!--notebook-skip-line-->

From the timings above, it seems as if converting to numpy takes lot more time than multiplying two large matrices. That doesn't seem right.

This is because, in MXNet, all operations are executed asynchronously. So, when `nd.dot(x, x)` returns, the matrix multiplication is not complete, it has only been queued for execution. However, [asnumpy](../../../api/legacy/ndarray/ndarray.rst#mxnet.ndarray.NDArray.asnumpy) has to wait for the result to be calculated in order to convert it to numpy array on CPU, hence taking a longer time. Other examples of 'blocking' operations include [asscalar](../../../api/legacy/ndarray/ndarray.rst#mxnet.ndarray.NDArray.asscalar) and [wait_to_read](../../../api/legacy/ndarray/ndarray.rst#mxnet.ndarray.NDArray.wait_to_read).

While it is possible to use [NDArray.waitall()](../../../api/legacy/ndarray/ndarray.rst#mxnet.ndarray.waitall) before and after operations to get running time of operations, it is not a scalable method to measure running time of multiple sets of operations, especially in a [Sequential](../../../api/gluon/nn/index.rst#mxnet.gluon.nn.Sequential) or hybridized network.

## The correct way to profile

The correct way to measure running time of MXNet models is to use MXNet profiler. In the rest of this tutorial, we will learn how to use the MXNet profiler to measure the running time and memory consumption of MXNet models. You can import the profiler and configure it from Python code.

```{.python .input}
from mxnet import profiler

profiler.set_config(profile_all=True,
                    aggregate_stats=True,
                    continuous_dump=True,
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

```{.python .input}
from mxnet import gluon

net = gluon.nn.HybridSequential()
net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
net.add(gluon.nn.Flatten())
net.add(gluon.nn.Dense(512, activation="relu"))
net.add(gluon.nn.Dense(10))
```

We need data that we can run through the network for profiling. We'll use the MNIST dataset.

```{.python .input}
from mxnet.gluon.data.vision import transforms

dataset = gluon.data.vision.MNIST(train=True)
dataset = dataset.transform_first(transforms.ToTensor())
dataloader = gluon.data.DataLoader(dataset, batch_size=64, shuffle=True)
```

Let's define a function that will run a single training iteration given `data` and `label`.

```{.python .input}
# Use GPU if available
if mx.device.num_gpus():
    device=mx.gpu()
else:
    device=mx.cpu()

# Initialize the parameters with random weights
net.initialize(mx.init.Xavier(), device=device)

# Use SGD optimizer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

# Softmax Cross Entropy is a frequently used loss function for multi-class classification
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# A helper function to run one training iteration
def run_training_iteration(data, label):
    # Load data and label is the right device
    data = data.to_device(device)
    label = label.to_device(device)
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

```{.python .input}
# Run the first iteration without profiling
itr = iter(dataloader)
run_training_iteration(*next(itr))
```

We'll run the next iteration with the profiler turned on.

```{.python .input}
data, label = next(itr)

# Ask the profiler to start recording
profiler.set_state('run')

run_training_iteration(*next(itr))

# Make sure all operations have completed
mx.npx.waitall()
# Ask the profiler to stop recording
profiler.set_state('stop')
# Dump all results to log file before download
profiler.dump()
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

There are a few ways to view the information collected by the profiler. You can view it in the console, you can view a more graphical version in a browser, or you can use a vendor tool such as Intel VTune or Nvidia NVProf to view output. For most scenarios the information you need can be obtained with MXNet's built in profiler support, but if you want to investigate the performance of operators alongside extra device about your hardware (e.g. cache hit rates, or CUDA kernel timings) then profiling jointly with vendor tools is recommended.

#### 1. View in console

You can use the `profiler.dumps()` method to view the information collected by the profiler in the console. The collected information contains time taken by each operator, time taken by each C API and memory consumed in both CPU and GPU.

```{.python .input}
profiler.set_state('run')
profiler.set_state('stop')
print(profiler.dumps())
```

![Profile Statistics](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/python/profiler/profile_stats.png)<!--notebook-skip-line-->

#### 2. View in browser

You can also dump the information collected by the profiler into a `json` file using the `profiler.dump()` function and view it in a browser.

```{.python .input}
profiler.dump(finished=False)
```

`dump()` creates a `json` file which can be viewed using a trace consumer like `chrome://tracing` in the Chrome browser. Here is a snapshot that shows the output of the profiling we did above. Note that setting the `finished` parameter to `False` will prevent the profiler from finishing dumping to file. If you just use `profiler.dump()`, you will no longer be able to profile the remaining sections of your model. 

![Tracing Screenshot](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/python/profiler/profiler_output_chrome.png)

Let's zoom in to check the time taken by operators

![Operator profiling](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/python/profiler/profile_operators.png)

The above picture visualizes the sequence in which the operators were executed and the time taken by each operator.

### Profiling oneDNN Operators
Reagrding oneDNN operators, the library has already provided the internal profiling tool. Firstly, you need set `DNNL_VERBOSE=1` to enable internal profiler.

`$ DNNL_VERBOSE=1 python my_script.py > dnnl_verbose.log`

Now, the detailed profiling insights of each oneDNN prmitive are saved into `dnnl_verbose.log` (like below).

```
dnnl_verbose,info,DNNL v1.1.2 (commit cb2cc7ac17ff4e2ef50805c7048d33256d82be4d)
dnnl_verbose,info,Detected ISA is Intel AVX-512 with Intel DL Boost
dnnl_verbose,exec,cpu,convolution,jit:avx512_common,forward_inference,src_f32::blocked:aBcd16b:f0 wei_f32::blocked:ABcd16b16a:f0 bia_undef::undef::f0 dst_f32::blocked:aBcd16b:f0,,alg:convolution_direct,mb32_ic32oc32_ih256oh256kh3sh1dh0ph1_iw256ow256kw3sw1dw0pw1,20.7539
```

For example, if you want to calculate the total executing time of `convolution` primitive, you can just run:

`$ cat dnnl_verbose.log | grep "exec,cpu,convolution" | awk 'BEGIN{FS=","} {SUM+=$11} END {print SUM}'`

Moreover, you can set `DNNL_VERBOSE=2` to collect both creating and executing time of each primitive.

`$ cat dnnl_verbose.log | grep "create,cpu,convolution" | awk 'BEGIN{FS=","} {SUM+=$11} END {print SUM}'`

`$ cat dnnl_verbose.log | grep "exec,cpu,convolution" | awk 'BEGIN{FS=","} {SUM+=$11} END {print SUM}'`


### Profiling Custom Operators
Should the existing NDArray operators fail to meet all your model's needs, MXNet supports [Custom Operators](../../extend/customop.ipynb) that you can define in Python. In `forward()` and `backward()` of a custom operator, there are two kinds of code: "pure Python" code (NumPy operators included) and "sub-operators" (NDArray operators called within `forward()` and `backward()`). With that said, MXNet can profile the execution time of both kinds without additional setup. Specifically, the MXNet profiler will break a single custom operator call into a pure Python event and several sub-operator events if there are any. Furthermore, all of those events will have a prefix in their names, which is, conveniently, the name of the custom operator you called.

Let's try profiling custom operators with the following code example:

```{.python .input}
class MyAddOne(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):  
        self.assign(out_data[0], req[0], in_data[0]+1)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])

@mx.operator.register('MyAddOne')
class CustomAddOneProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(CustomAddOneProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return [in_shape[0]], [in_shape[0]], []

    def create_operator(self, device, shapes, dtypes):
        return MyAddOne()


inp = mx.np.zeros(shape=(500, 500))

profiler.set_config(profile_all=True, continuous_dump=True, \
                    aggregate_stats=True)
profiler.set_state('run')

w = nd.Custom(inp, op_type="MyAddOne")

mx.npx.waitall()

profiler.set_state('stop')
print(profiler.dumps())
profiler.dump(finished=False)
```

Here, we have created a custom operator called `MyAddOne`, and within its `forward()` function, we simply add one to the input. We can visualize the dump file in `chrome://tracing/`:

![Custom Operator Profiling Screenshot](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/python/profiler/profiler_output_custom_operator_chrome.png)

As shown by the screenshot, in the **Custom Operator** domain where all the custom operator-related events fall into, we can easily visualize the execution time of each segment of `MyAddOne`. We can tell that `MyAddOne::pure_python` is executed first. We also know that `CopyCPU2CPU` and `_plus_scalr` are two "sub-operators" of `MyAddOne` and the sequence in which they are executed.

Please note that: to be able to see the previously described information, you need to set `profile_imperative` to `True` even when you are using custom operators in [symbolic mode](https://mxnet.apache.org/versions/master/api/python/docs/api/legacy/symbol/index.html) (refer to the code snippet below, which is the symbolic-mode equivelent of the code example above). The reason is that within custom operators, pure python code and sub-operators are still called imperatively. 

```{.python .input} 
# Set profile_all to True
profiler.set_config(profile_all=True, aggregate_stats=True, continuous_dump=True)
# OR, Explicitly Set profile_symbolic and profile_imperative to True
profiler.set_config(profile_symbolic=True, profile_imperative=True, \
                    aggregate_stats=True, continuous_dump=True)

profiler.set_state('run')
# Use Symbolic Mode
a = mx.symbol.Variable('a')
b = mx.symbol.Custom(data=a, op_type='MyAddOne')
c = b.bind(mx.cpu(), {'a': inp})
y = c.forward()
mx.npx.waitall()
profiler.set_state('stop')
print(profiler.dumps())
profiler.dump()
```

### Some Rules to Pay Attention to
1. Always use `profiler.dump(finished=False)` if you do not intend to finish dumping to file. Otherwise, calling `profiler.dump()` in the middle of your model may lead to unexpected behaviors; and if you subsequently call `profiler.set_config()`, the program will error out.

2. You can only dump to one file. Do not change the target file by calling `profiler.set_config(filename='new_name.json')` in the middle of your model. This will lead to incomplete dump outputs.

## Advanced: Using NVIDIA Profiling Tools

MXNet's Profiler is the recommended starting point for profiling MXNet code, but NVIDIA also provides a couple of tools for low-level profiling of CUDA code: [NVProf](https://devblogs.nvidia.com/cuda-pro-tip-nvprof-your-handy-universal-gpu-profiler/), [Visual Profiler](https://developer.nvidia.com/nvidia-visual-profiler) and [Nsight Compute](https://developer.nvidia.com/nsight-compute). You can use these tools to profile all kinds of executables, so they can be used for profiling Python scripts running MXNet. And you can use these in conjunction with the MXNet Profiler to see high-level information from MXNet alongside the low-level CUDA kernel information.

### NVProf and Visual Profiler

NVProf and Visual Profiler are available in CUDA 9 and CUDA 10 toolkits. You can get a timeline view of CUDA kernel executions, and also analyse the profiling results to get automated recommendations. It is useful for profiling end-to-end training but the interface can sometimes become slow and unresponsive.

You can initiate the profiling directly from inside Visual Profiler or from the command line with `nvprof` which wraps the execution of your Python script. If it's not on your path already, you can find `nvprof` inside your CUDA directory. See [this discussion post](https://discuss.mxnet.io/t/using-nvidia-profiling-tools-visual-profiler-and-nsight-compute/) for more details on setup.

`$ nvprof -o my_profile.nvvp python my_profiler_script.py`

`==11588== NVPROF is profiling process 11588, command: python my_profiler_script.py`

`==11588== Generated result file: /home/user/Development/mxnet/ci/my_profile.nvvp`

We specified an output file called `my_profile.nvvp` and this will be annotated with NVTX ranges (for MXNet operations) that will be displayed alongside the standard NVProf timeline. This can be very useful when you're trying to find patterns between operators run by MXNet, and their associated CUDA kernel calls.

You can open this file in Visual Profiler to visualize the results.

![Operator profiling nvprof](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/python/profiler/profiler_nvprof.png)

At the top of the plot we have CPU tasks such as driver operations, memory copy calls, MXNet engine operator invocations, and imperative MXNet API calls.  Below we see the kernels active on the GPU during the same time period.

![Operator profiling nvprof zoomed](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/python/profiler/profiler_nvprof_zoomed.png)

Zooming in on a backwards convolution operator we can see that it is in fact made up of a number of different GPU kernel calls, including a cuDNN winograd convolution call, and a fast-fourier transform call.

![Operator profiling winograd](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/python/profiler/profiler_winograd.png)

Selecting any of these kernel calls (the winograd convolution call shown here) will get you some interesting GPU performance information such as occupancy rates (vs theoretical), shared memory usage and execution duration.

### Nsight Compute

Nsight Compute is available in CUDA 10 toolkit, but can be used to profile code running CUDA 9. You don't get a timeline view, but you get many low level statistics about each individual kernel executed and can compare multiple runs (i.e. create a baseline).

![Nsight Compute](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/python/profiler/profile_nsight_compute.png)

## Further reading

- [Examples using MXNet profiler.](https://github.com/apache/mxnet/tree/master/example/profiler)
- [Some tips for improving MXNet performance.](https://mxnet.apache.org/api/faq/perf)

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->

