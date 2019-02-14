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

Linear Classification Using Sparse Matrix Multiplication
===========
This examples trains a linear model using the sparse feature in MXNet. This is for demonstration purpose only.

The example utilizes the sparse data loader ([mx.io.LibSVMIter](https://mxnet.incubator.apache.org/versions/master/api/python/io/io.html#mxnet.io.LibSVMIter)),
the sparse dot operator and [sparse gradient updaters](https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/sparse.html#updater)
to train a linear model on the
[Avazu](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#avazu) click-through-prediction dataset.

The example also shows how to perform distributed training with the sparse feature.

- `python train.py`

Notes on Distributed Training:

- For distributed training, please use the `../../tools/launch.py` script to launch a cluster.
- For example, to run two workers and two servers with one machine, run `../../../tools/launch.py -n 2 --launcher=local python train.py --kvstore=dist_async`
