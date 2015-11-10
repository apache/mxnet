Frequent Asked Questions
========================
This document contains the frequent asked question to mxnet.


What is the relation between MXNet and CXXNet, Minerva, Purine2
---------------------------------------------------------------
MXNet is created in collaboration by authors from the three projects.
The project reflects what we have learnt from the past projects.
It combines important flavour of the existing projects, being
efficient, flexible and memory efficient.

It also contains new ideas, that allows user to combines different
ways of programming, and write CPU/GPU applications that are more
memory efficient than cxxnet, purine and more flexible than minerva.


What is the Relation to Tensorflow
----------------------------------
Both MXNet and Tensorflow uses a computation graph abstraction, which is initially used by Theano, then also adopted by other packages such as CGT, caffe2, purine. Currently TensorFlow adopts an optimized symbolic API. While mxnet supports a more [mixed flavor]( mxnet.readthedocs.org/en/latest/program_model.html), with a dynamic dependency scheduler to combine symbolic and imperative programming together. T
 In short, mxnet is lightweight and “mixed”, while getting similar advantages by using a computation graph to make it fast and memory efficient. This being said, most system will involve and we expect both systems can learn and benefit from each other.


How to Build the Project
------------------------
See [build instruction](build.md)
