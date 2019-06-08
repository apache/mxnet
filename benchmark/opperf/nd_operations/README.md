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

# TODO - Operators to add to this Benchmark Utility

Below are list of operators to be added to the benchmarking utility:

**NN Operations**

MXNet NDArray NN Operators

1. FullyConnected (Basic)
3. Embedding

MXNet NDArray Normalization Operators.

1. Dropout
2. BatchNorm

MXNet NDArray Recurrent Operators

1. RNN
2. LSTM
3. GRU

(Under the hood uses mx.nd.rnn)


**Tensor Operations**

MXNet NDArray Conversion Operations

1. Copy
2. CopyTo
3. as_in_context
4. asnumpy
5. asscalar
6. astype

MXNet NDArray Creation Operations

1. Zeros
2. Ones
5. full
6. arange

MXNet NDArray Indexing Operations

1. get_item (x[i])
2. set_item (x[i])
3. slice
4. slice_axis
5. take
6. batch_take
7. pick

MXNet NDArray Join and Split Operations

1. concat
2. split
3. stack

MXNet NDArray Reduction Operations

1. sum
2. nansum
3. prod
4. nanprod
5. mean
6. max
7. min
8. norm

MXNet NDArray Shape change Operations

1. Transpose
2. shape_array
3. size_array
4. reshape
5. reshape_like
6. flatten
7. expand_dims
8. split
9. diag
10. tile
11. pad

MXNet NDArray Sorting and Searching Operations

1. sort
2. argsort
3. topk
4. argmax
5. argmin
6. Sort and Argsort
    6.1 Descending Order
    6.2 Flatten and sort
7. TopK
    7.1 K being a very small number (ex: 1) on a axis with 1000 values.

MXNet NDArray Miscellaneous Operations

1. where
2. clip
