Frequently Asked Questions
==========================
This document contains the frequently asked questions to mxnet.

How to Copy Part of Parameters to Another Model
-----------------------------------------------
Most MXNet's model consists two parts, the argument arrays and symbol. You can simply copy the argument arrary to the argument array of another model. For example, in python model API, you can do
```python
copied_model =  mx.model.FeedForward(ctx=mx.gpu(), symbol=new_symbol,
                                     arg_params=old_arg_params, aux_params=old_aux_params,
                                     allow_extra_params=True);
```
To copy model parameter from existing ```old_arg_params```, see also this [notebook](https://github.com/dmlc/mxnet/blob/master/example/notebooks/predict-with-pretrained-model.ipynb)

How to Extract Feature Map of Certain Layer
------------------------------------------
See this [notebook](https://github.com/dmlc/mxnet/blob/master/example/notebooks/predict-with-pretrained-model.ipynb)


What is the relation between MXNet and CXXNet, Minerva, Purine2
---------------------------------------------------------------
MXNet is created in collaboration by authors from the three projects.
The project reflects what we have learnt from the past projects.
It combines important flavour of the existing projects, being
efficient, flexible and memory efficient.

It also contains new ideas, that allows user to combine different
ways of programming, and write CPU/GPU applications that are more
memory efficient than cxxnet, purine and more flexible than minerva.


What is the Relation to Tensorflow
----------------------------------
Both MXNet and Tensorflow use a computation graph abstraction, which is initially used by Theano, then also adopted by other packages such as CGT, caffe2, purine. Currently TensorFlow adopts an optimized symbolic API. While mxnet supports a more [mixed flavor](https://mxnet.readthedocs.org/en/latest/program_model.html), with a dynamic dependency scheduler to combine symbolic and imperative programming together. 
In short, mxnet is lightweight and “mixed”, with flexiblity from imperative programing, while getting similar advantages by using a computation graph to make it very fast and memory efficient. That being said, most systems will involve and we expect both systems can learn and benefit from each other.


How to Build the Project
------------------------
See [build instruction](build.md)
