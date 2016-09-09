# Frequently Asked Questions

We collect the frequently asked questions on [mxnet/issues](https://github.com/dmlc/mxnet/issues). For whom is going to post issues, please consider to check this page first. For contributors, please make the questions and answers simple; otherwise put the detailed answer in more proper place and then refer the link.

## Build and install

The answers for most questions can be found on the [build page](build.md)

## Speed

#### Took a long time to start running on GPU

Try to disable opencv to use GPU, such as [build opencv from source with GPU module disabled](build.md#build-opencv-from-source). 

#### Slow on a single GPU

Check the following items:

1. Check your CUDA/driver version is not too old. 
2. Build with `USE_CUDNN=1`, often brings 50+% speedup. Try to use the newest version. 
3. Set `export MXNET_CUDNN_AUTOTUNE_DEFAULT=1` before running, often 10%-15% speedup
4. Dsiable ECC if using Tesla GPUs by `nvidia-smi -e 0`. Root permission and reboot may be needed.
5. Set to maximal clock for Tesla cards by `nvidia-smi -ac ??`. See [this blog](https://devblogs.nvidia.com/parallelforall/increase-performance-gpu-boost-k80-autoboost/)
6. Check no throttle reason `nvidia-smi -q -d PERFORMANCE` often caused by temperature. 

#### No speedup for using more than one GPUs or machines. 

Check the following items:
1. does your neural network already run fast, such as >1000 example/sec or >10 batches/sec? If yes, it's unlikely to get further speedup for adding more resources due to the communication overhead. 
2. Are you using a small batch size? Try to increase it.
3. Are you using more than 4 GPUs? Try to use `--kv-store=device`

## Memory Usage

#### Abnormal CPU memory usage

May be due to the data prefetch. Refer to [issue 2111](https://github.com/dmlc/mxnet/issues/2111) Should be fixed later.

## the following part needs refactor

#### How to Copy Part of Parameters to Another Model
Most MXNet's model consists two parts, the argument arrays and symbol. You can simply copy the argument arrary to the argument array of another model. For example, in python model API, you can do
```python
copied_model =  mx.model.FeedForward(ctx=mx.gpu(), symbol=new_symbol,
                                     arg_params=old_arg_params, aux_params=old_aux_params,
                                     allow_extra_params=True);
```
To copy model parameter from existing ```old_arg_params```, see also this [notebook](https://github.com/dmlc/mxnet/blob/master/example/notebooks/predict-with-pretrained-model.ipynb)

#### How to Extract Feature Map of Certain Layer
See this [notebook](https://github.com/dmlc/mxnet/blob/master/example/notebooks/predict-with-pretrained-model.ipynb)


#### What is the relation between MXNet and CXXNet, Minerva, Purine2
MXNet is created in collaboration by authors from the three projects.
The project reflects what we have learnt from the past projects.
It combines important flavour of the existing projects, being
efficient, flexible and memory efficient.

It also contains new ideas, that allows user to combine different
ways of programming, and write CPU/GPU applications that are more
memory efficient than cxxnet, purine and more flexible than minerva.


#### What is the Relation to Tensorflow
Both MXNet and Tensorflow use a computation graph abstraction, which is initially used by Theano, then also adopted by other packages such as CGT, caffe2, purine. Currently TensorFlow adopts an optimized symbolic API. While mxnet supports a more [mixed flavor](https://mxnet.readthedocs.org/en/latest/program_model.html), with a dynamic dependency scheduler to combine symbolic and imperative programming together. 
In short, mxnet is lightweight and “mixed”, with flexiblity from imperative programing, while getting similar advantages by using a computation graph to make it very fast and memory efficient. That being said, most systems will involve and we expect both systems can learn and benefit from each other.
