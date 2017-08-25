Basics
======

This tutorial provides basic usages of the C++ package through the classical handwritten digits
identification database--[MNIST](http://yann.lecun.com/exdb/mnist/).

The following contents assume that the working directory is `/path/to/mxnet/cpp-package/example`.

Load Data
--------
Before going into codes, we need to fetch MNIST data. You can either use the script `get_mnist.sh`,
or download mnist data by yourself from Lecun's [website](http://yann.lecun.com/exdb/mnist/)
and decompress them into `mnist_data` folder.

Except linking the MXNet shared library, the C++ package itself is a header-only package,
which means all you need to do is to include the header files. Among the header files,
`op.h` is special since it is generated dynamically. The generation should be done when
[building the C++ package](http://mxnet.io/get_started/build_from_source.html#build-the-c++-package).
After that, you also need to copy the shared library (`libmxnet.so` in linux,
`libmxnet.dll` in windows) from `/path/to/mxnet/lib` to the working directory.
We do not recommend you to use pre-built binaries because MXNet is under heavy development,
the operator definitions in `op.h` may be incompatible with the pre-built version.

In order to use functionalities provides by the C++ package, first we include the general 
header file `MxNetCpp.h` and specify the namespaces.

```cpp
#include "mxnet-cpp/MxNetCpp.h"

using namespace std;
using namespace mxnet::cpp;
```

Next we can use the data iter to load MNIST data (separated to training sets and validation sets).
The digits in MNIST are 2-dimension arrays, so we should set `flat` to true to flatten the data.

```cpp
auto train_iter = MXDataIter("MNISTIter")
    .SetParam("image", "./mnist_data/train-images-idx3-ubyte")
    .SetParam("label", "./mnist_data/train-labels-idx1-ubyte")
    .SetParam("batch_size", batch_size)
    .SetParam("flat", 1)
    .CreateDataIter();
auto val_iter = MXDataIter("MNISTIter")
    .SetParam("image", "./mnist_data/t10k-images-idx3-ubyte")
    .SetParam("label", "./mnist_data/t10k-labels-idx1-ubyte")
    .SetParam("batch_size", batch_size)
    .SetParam("flat", 1)
    .CreateDataIter();
```

The data have been successfully loaded, we can now easily construct various models to identify
the digits with the help of C++ package.


Multilayer Perceptron
---------------------
If you are not familiar with multilayer perceptron, you can get some basic information
[here](http://mxnet.io/tutorials/python/mnist.html#multilayer-perceptron). We only focus on
the implementation in this tutorial.

Constructing multilayer perceptron model is straightforward, assume we store the hidden size
for each layer in `layers`, and each layer uses
[ReLu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) function as activation.

```cpp
Symbol mlp(const vector<int> &layers) {
  auto x = Symbol::Variable("X");
  auto label = Symbol::Variable("label");

  vector<Symbol> weights(layers.size());
  vector<Symbol> biases(layers.size());
  vector<Symbol> outputs(layers.size());

  for (int i=0; i<layers.size(); ++i) {
    weights[i] = Symbol::Variable("w" + to_string(i));
    biases[i] = Symbol::Variable("b" + to_string(i));
    Symbol fc = FullyConnected(
      i == 0? x : outputs[i-1]
      weights[i],
      biases[i],
      layers[i]
    );
    outputs[i] = i == layers.size()-1 ? fc : Activation(fc, ActivationActType::relu);
  }

  return SoftmaxOutput(outputs.back(), label);
}
```

The above function defines a multilayer perceptron model where hidden sizes are specified
by `layers`.

We now create and initialize the parameters after the model is constructed. MXNet can help
 you to infer shapes of most of the parameters. Basically only the shape of data and label
 is needed.

```cpp
std::map<string, NDArray> args;
args["X"] = NDArray(Shape(batch_size, image_size*image_size), ctx);
args["label"] = NDArray(Shape(batch_size), ctx);
// Let MXNet infer shapes other parameters such as weights
net.InferArgsMap(ctx, &args, args);

// Initialize all parameters with uniform distribution U(-0.01, 0.01)
auto initializer = Uniform(0.01);
for (auto& arg : args) {
  // arg.first is parameter name, and arg.second is the value
  initializer(arg.first, &arg.second);
}
```

The rest is to train the model with an optimizer.
```cpp
// Create sgd optimizer
Optimizer* opt = OptimizerRegistry::Find("sgd");
opt->SetParam("rescale_grad", 1.0/batch_size);

// Start training
for (int iter = 0; iter < max_epoch; ++iter) {
  train_iter.Reset();

  while (train_iter.Next()) {
    auto data_batch = train_iter.GetDataBatch();
    // Set data and label
    args["X"] = data_batch.data;
    args["label"] = data_batch.label;

    // Create executor by binding parameters to the model
    auto *exec = net.SimpleBind(ctx, args);
    // Compute gradients
    exec->Forward(true);
    exec->Backward();
    // Update parameters
    exec->UpdateAll(opt, learning_rate, weight_decay);
    // Remember to free the memory
    delete exec;
  }
}
```

We also want to see how our model performs. The C++ package provides convenient APIs for
evaluating. Here we use accuracy as metric. The inference is almost the same as training,
 except that we don't need gradients.

```cpp
Accuracy acc;
val_iter.Reset();
while (val_iter.Next()) {
  auto data_batch = val_iter.GetDataBatch();
  args["X"] = data_batch.data;
  args["label"] = data_batch.label;
  auto *exec = net.SimpleBind(ctx, args);
  // Forward pass is enough as no gradient is needed when evaluating
  exec->Forward(false);
  acc.Update(data_batch.label, exec->outputs[0]);
  delete exec;
}
```

You can find the complete code in `mlp_cpu.cpp`. Use `make mlp_cpu` to compile it,
 and `./mlp_cpu` to run it.

GPU Support
-----------
It's worth noting that changing context from `Context::cpu()` to `Context::gpu()` is not enough,
because the data read by data iter are stored in memory, we cannot assign it directly to the
parameters. To bridge this gap, NDArray provides data synchronization functionalities between
GPU and CPU. We will illustrate it by making the mlp code run on GPU.

In the previous code, data are used like

```cpp
args["X"] = data_batch.data;
args["label"] = data_batch.label;
```

It will be problematic if other parameters are created in the context of GPU. We can use
`NDArray::CopyTo` to solve this problem.

```cpp
// Data provided by DataIter are stored in memory, should be copied to GPU first.
data_batch.data.CopyTo(&args["X"]);
data_batch.label.CopyTo(&args["label"]);
// CopyTo is imperative, need to wait for it to complete.
NDArray::WaitAll();
```

By replacing the former code to the latter one, we successfully port the code to GPU. You can find the complete code in `mlp_gpu.cpp`. Compilation is similar to the cpu version. (Note: The shared library should be built with GPU support on)
