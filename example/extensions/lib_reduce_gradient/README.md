<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
  ~
-->

Add Reduce operation to computation Graph
=======================================

## Introduction
This is the part of work of transferring [DeepSpeed's work](https://arxiv.org/abs/1910.02054) into MXNet.
Since the difference between symbolic and imperative, we divide the whole proecss into two phases:  

phase 1: Add reduce operation into graph. The reduce operation will do nothing
in forward but reduce the gradient to the right GPU(according to POS-trainer).  

phase2: In backward graph, delete the outputs in arrays so the memory planner can reuse such memory.  

 ## Getting start 
 ### Prepare NCCL and horovod
 Since we use horovod to communicate, please firstly install horovod. And we use NCCL reduce, please also install it.  
 
 ### Complie the Graph Pass and load
 Please firstly compile it like [lib pass](../lib_pass/). Run `make` and it will generate dynamic library
 **add_reduce_op_lib.so**  which is compiled from the `add_reduce_op.cc` file. Then load such file in your python code like
```python
import mxnet as mx
mx.library.load('add_reduce_op_lib.so')
```
 
 ### Prepare options
 Then we need know the correct partition of parameters and gradients about their GPUs.
 So please use **POS_Trainer** from `pos_trainer.py` like normal trainer in MXNet.
 ```python
from pos_trainer import POS_Trainer
trainer = POS_Trainer(params_dict, "adam", optimizer_params)
```
Then trainer can generate corresponding options like:
 ```python
options = trainer.generate_graph_pass_options()
backward_options = trainer.generate_backward_options()]
```
### modify graph
Before forward, we use 
 ```python
model.optimize_for(x, backend = "add_reduce_op", **options)
```
to insert reduce operation into graphs.   
Then we call backward option as 
 ```python
loss.backward(backward_option = backward_options)
```