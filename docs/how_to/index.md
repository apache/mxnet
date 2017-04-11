# MXNet How To

The how-tos provide a range of information from installation, basic concepts and general guidance to demos complete with pre-trained models, instructions, and commands.


&nbsp;

The following topics explain basic concepts and provide procedures for specific tasks. Some include demos complete with pre-trained models.

## Using Pre-trained Models
The MXNet [Model zoo](http://mxnet.io/model_zoo/index.html) is a growing collection of pre-trained models for a variety of tasks.
In particular, the popular task of using a ConvNet to figure out what is in an image is described in detail in the tutorial on
[use pre-trained image classification models](http://mxnet.io/tutorials/python/predict_imagenet.html).  This provides step-by-step instructions on loading, customizing, and predicting image classes with the provided pre-trained image classification model.

## Use MXNet to Perform Specific Tasks

* [How to use Fine-tune with Pre-trained Models](http://mxnet.io/how_to/finetune.html)
*Provides instructions for tuning a pre-trained neural network for use with a new, smaller data set. It describes preparing the data, training a new final layer, and evaluating the results. For comparison, it uses two pre-trained models.*

* [How to visualize Neural Networks as computation graph](http://mxnet.io/how_to/visualize_graph.html)
*Provides commands and instructions to visualize neural networks on a Jupyter notebook.*

* [How to train with multiple CPU/GPUs with data parallelism](http://mxnet.io/how_to/multi_devices.html)
*Provides the MXNet defaults for using multiple GPUs. It also provides instructions and commands for customizing GPU data parallelism setting (such as the number of GPUs and individual GPU workload), training a model with multiple GPUs, setting GPU communication options, synchronizing directories between GPUs, choosing a network interface, and debugging connections.*

* [How to train with multiple GPUs in model parallelism - train LSTM](http://mxnet.io/how_to/model_parallel_lstm.html)
*Discusses the basic practices of model parallelism, such as using each GPU for a layer of a multi-layer model, and how to balance and organize model layers among multiple GPUs to reduce data transmission and bottlenecks.*


* [How to run MXNet on smart or mobile devices](http://mxnet.io/how_to/smart_device.html)
*Provides general guidance on porting software to other systems and languages. It also describes the trade-offs that you need to consider when running a model on a mobile device. Provides basic pre-trained image recognition models for use with a mobile device.*

* [How to set up MXNet on the AWS Cloud using Amazon EC2 and Amazon S3](http://mxnet.io/how_to/cloud.html)
*Provides step-by-step instructions on using MXNet on AWS. It describes the prerequisites for using Amazon Simple Storage Service (Amazon S3) and Amazon Elastic Compute Cloud (Amazon EC2) and the libraries that MXNet depends on in this environment. Instructions explain how to set up, build, install, test, and run MXNet, and how to run MXNet on multiple GPUs in the Cloud.*

* [How to use MXNet on variable input length/size (bucketing)](http://mxnet.io/how_to/bucketing.html)
*Explains the basic concepts and reasoning behind using bucketing for models that have different architectures, gives an example of how to modify models to use buckets, and provides instructions on how to create and iterate over buckets.*

* [How to improve MXNet performance](http://mxnet.io/how_to/perf.html)
*Explains how to improve MXNet performance by using the recommended data format, storage locations, batch sizes, libraries, and parameters, and more.*

* [How to use nnpack improve cpu performance of MXNet](http://mxnet.io/how_to/nnpack.html)
*Explains how to improve cpu performance of MXNet by using [nnpack](https://github.com/Maratyszcza/NNPACK). currently, nnpack support convolution, max-pooling, fully-connected operator.*

* [How to use MXNet within a Matlab environment](https://github.com/dmlc/mxnet/tree/master/matlab)
*Provides the commands to load a model and data, get predictions, and do feature extraction in Matlab using the MXNet library. It includes an implementation difference between the two that can cause issues, and some basic troubleshooting.*

* [How to use MXNet in a browser using Java Script](https://github.com/dmlc/mxnet.js/)
*Provides a JavaScript port of MXNet, along with some basic commands to set it up and run it on a server for use in a browser. It includes browser comparisons, instructions on using your own model, and the library code.*


## Develop and Hack MXNet

* [Create new operators](new_op.md)
*Provides an example of a custom layer along with the steps required to define each part of the layer. It includes a breakdown of the parameters involved, calls out possible optimizations, and provides a link to the complete code for the example.*

* [Use Torch from MXNet](torch.md)
*Describes how to install and build MXNet for use with a Torch frontend. It also provides a list of supported Torch mathematical functions and neural network modules, and instructions on how to use them.*


* [Set Environment Variables for MXNet](env_var.md)

Provides a list of default MXNet environment variables, along with a short description of what each control.

## Frequently Asked Questions

* [FAQ](faq.md)
*The FAQ provides debugging and optimization information for MXNet, along with links to additional resources.*
