# How to Use MXNet As an (Almost) Full-function Torch Front End

This topic demonstrates how to use MXNet as a front end to two of Torch's major functionalities:

* Call Torch's tensor mathematical functions with MXNet.NDArray 

* Embed Torch's neural network modules (layers) into MXNet's symbolic graph 
## Compile with Torch


* Install Torch using the [official guide](http://torch.ch/docs/getting-started.html).
	* If you haven't already done so, copy `make/config.mk` (Linux) or `make/osx.mk` (Mac) into the MXNet root folder as `config.mk`. In `config.mk` uncomment the lines `TORCH_PATH = $(HOME)/torch` and `MXNET_PLUGINS += plugin/torch/torch.mk`.
    * By default, Torch should be installed in your home folder (so `TORCH_PATH = $(HOME)/torch`). Modify TORCH_PATH to point to your torch installation, if necessary. 
* Run `make clean && make` to build MXNet with Torch support.

## Tensor Mathematics
The mxnet.th module supports calling Torch's tensor mathematical functions with mxnet.nd.NDArray. See [complete code](https://github.com/dmlc/mxnet/blob/master/example/torch/torch_function.py):

 ```Python
    import mxnet as mx
    x = mx.th.randn(2, 2, ctx=mx.cpu(0))
    print x.asnumpy()
    y = mx.th.abs(x)
    print y.asnumpy()

    x = mx.th.randn(2, 2, ctx=mx.cpu(0))
    print x.asnumpy()
    mx.th.abs(x, x) # in-place
    print x.asnumpy()
 ```
For help, use the `help(mx.th)` command. 

We've added support for most common functions listed on [Torch's documentation page](https://github.com/torch/torch7/blob/master/doc/maths.md). 
If you find that the function you need is not supported, you can easily register it in `mxnet_root/plugin/torch/torch_function.cc` by using the existing registrations as examples.

## Torch Modules (Layers)
MXNet supports Torch's neural network modules through  the`mxnet.symbol.TorchModule` symbol.
For example, the following code defines a three-layer DNN for classifying MNIST digits ([full code](https://github.com/dmlc/mxnet/blob/master/example/torch/torch_module.py)):

 ```Python
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.TorchModule(data_0=data, lua_string='nn.Linear(784, 128)', num_data=1, num_params=2, num_outputs=1, name='fc1')
    act1 = mx.symbol.TorchModule(data_0=fc1, lua_string='nn.ReLU(false)', num_data=1, num_params=0, num_outputs=1, name='relu1')
    fc2 = mx.symbol.TorchModule(data_0=act1, lua_string='nn.Linear(128, 64)', num_data=1, num_params=2, num_outputs=1, name='fc2')
    act2 = mx.symbol.TorchModule(data_0=fc2, lua_string='nn.ReLU(false)', num_data=1, num_params=0, num_outputs=1, name='relu2')
    fc3 = mx.symbol.TorchModule(data_0=act2, lua_string='nn.Linear(64, 10)', num_data=1, num_params=2, num_outputs=1, name='fc3')
    mlp = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
 ```
Let's break it down. First `data = mx.symbol.Variable('data')` defines a Variable as a placeholder for input.
Then, it's fed through Torch's nn modules with:
     `fc1 = mx.symbol.TorchModule(data_0=data, lua_string='nn.Linear(784, 128)', num_data=1, num_params=2, num_outputs=1, name='fc1')`.
To use Torch's criterion as loss functions, you can replace the last line with:
 ```Python
    logsoftmax = mx.symbol.TorchModule(data_0=fc3, lua_string='nn.LogSoftMax()', num_data=1, num_params=0, num_outputs=1, name='logsoftmax')
    # Torch's label starts from 1
    label = mx.symbol.Variable('softmax_label') + 1
    mlp = mx.symbol.TorchCriterion(data=logsoftmax, label=label, lua_string='nn.ClassNLLCriterion()', name='softmax')
 ```
The input to the nn module is named data_i for i = 0 ... num_data-1. `lua_string` is a single Lua statement that creates the module object.
For Torch's built-in module, this is simply `nn.module_name(arguments)`.
If you are using a custom module, place it in a .lua script file and load it with `require 'module_file.lua'` if your script returns a torch.nn object, or `(require 'module_file.lua')()` if your script returns a torch.nn class.

