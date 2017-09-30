# Frequently Asked Questions

This topic provides answers to the frequently asked questions on [mxnet/issues](https://github.com/dmlc/mxnet/issues). Before posting an issue, please check this page. If you would like to contribute to this page, please make the questions and answers simple. If your answer is  extremely detailed, please post it elsewhere and link to it.

## Building and Installation

You can find answers to most questions the [build page](http://mxnet.io/get_started/setup.html).

## Speed

#### It took a long time to start running on a GPU

Try to disable opencv to use a GPU: [build opencv from source with GPU module disabled](http://mxnet.io/get_started/setup.html#build-opencv-from-source-code).

#### It's slow on a single GPU

Check the following:

1. Ensure that your CUDA/driver version is not too old.
2. Build with `USE_CUDNN=1`. This often increases speed 50+%. Try to use the newest version.
3. Set `export MXNET_CUDNN_AUTOTUNE_DEFAULT=1` before running. This often increases speed 10%-15%.
4. If you are using Tesla GPUs by `nvidia-smi -e 0`, disable ECC. You might need root permission and have to reboot.
5. For Tesla cards by `nvidia-smi -ac ??`, set to the maximal clock. For details, see [this blog](https://devblogs.nvidia.com/parallelforall/increase-performance-gpu-boost-k80-autoboost/).
6. No throttle reason `nvidia-smi -q -d PERFORMANCE` is often caused by temperature.

#### No increase in speed when using more than one GPU or computer

Check the following:

1. Does your neural network already run fast, such as >1000 example/sec or >10 batches/sec? If yes, it's unlikely to speed up any further by adding more resources because of the communication overhead.
2. Are you using a small batch size? Try to increase it.
3. Are you using more than 4 GPUs? Try using `--kv-store=device`.

## Memory Usage

#### Abnormal CPU memory usage

This might be due to the data pre-fetch. See [issue 2111](https://github.com/dmlc/mxnet/issues/2111).

## Pending Review
The following topics need to be reviewed.

#### How to Copy Part of Parameters to Another Model
Most MXNet model consists two parts, the argument arrays and symbol. You can simply copy the argument array to the argument array of another model. For example, in the Python model API, you can do this:

```python
copied_model =  mx.model.FeedForward(ctx=mx.gpu(), symbol=new_symbol,
                                     arg_params=old_arg_params, aux_params=old_aux_params,
                                     allow_extra_params=True);
```
For information about copying model parameters from an existing ```old_arg_params```, see this [notebook](https://github.com/dmlc/mxnet-notebooks/blob/master/python/how_to/predict.ipynb). More notebooks please refer to [dmlc/mxnet-notebooks](https://github.com/dmlc/mxnet-notebooks).

#### How to Extract the Feature Map of a Certain Layer
See this [notebook](https://github.com/dmlc/mxnet-notebooks/blob/master/python/how_to/predict.ipynb). More notebooks please refer to [dmlc/mxnet-notebooks](https://github.com/dmlc/mxnet-notebooks).


#### What Is the Relationship Between MXNet and CXXNet, Minerva, and Purine2?
MXNet is created in collaboration by authors from the three projects.
MXNet reflects what we have learned from these projects.
It combines the important aspects of the existing projects: general efficiency, flexibility, and memory efficiency.

MXNet also contains new approaches that allow you to combine different
ways of programming and write CPU/GPU applications that are more
memory efficient than CXXNet and Purine, and more flexible than Minerva.


#### What Is the Relationship Between MXNet and TensorFlow?
Both MXNet and [TensorFlow](https://www.tensorflow.org/) use computation graph abstraction, which was initially used by Theano, then adopted by other packages, such as CGT, Caffe2, and Purine. Currently, TensorFlow adopts an optimized symbolic API. MXNet supports a [mixed approach](https://mxnet.io/architecture/program_model.html), with a dynamic dependency scheduler to combine symbolic and imperative programming.
In short, MXNet is lightweight and *mixed* with flexibility from imperative programming, while using a computation graph to make it very fast and memory efficient.
