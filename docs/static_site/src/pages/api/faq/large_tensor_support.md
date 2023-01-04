---
layout: page_category
title: Using MXNet with Large Tensor Support
category: faq
faq_c: Extend and Contribute to MXNet
question: How do I use MXNet built with Large Tensor Support
permalink: /api/faq/large_tensor_support
---
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

# Using MXNet with Large Tensor Support

## What is large tensor support?
When creating a network that uses large amounts of data, as in a deep graph problem, you may need large tensor support. This means tensors are indexed using INT64, instead of INT32 indices.

This feature is enabled when MXNet is built with a flag *USE_INT64_TENSOR_SIZE=1*, which is now a default setting. You can also make MXNet use INT32 indices by changing this flag.

## When do you need it?
1. When you are creating NDArrays of size larger than 2^31 elements.
2. When the input to your model requires tensors that have inputs larger than 2^31 (when you load them all at once in your code) or attributes greater than 2^31.

## How to identify that you need to use large tensors ?
When you see one of the following errors:


1. OverflowError: unsigned int is greater than maximum
2. Check failed: inp->shape().Size() < 1 >> 31 (4300000000 vs. 0) : Size of tensor you are trying to allocate is larger than 2^32 elements. Please build with flag USE_INT64_TENSOR_SIZE=1
3. Invalid Parameter format for end expect int or None but value='2150000000', in operator slice_axis(name="", end="2150000000", begin="0", axis="0"). *_Basically input attribute was expected to be int32, which is less than 2^31 and the received value is larger than that so, operator's parmeter inference treats that as a string which becomes unexpected input.`_*

## How to use it ?
You can create a large NDArray that requires large tensor enabled build to run as follows:

```python
LARGE_X=4300000000
a = mx.nd.arange(0, LARGE_X, dtype=“int64”)
or
a = nd.ones(shape=LARGE_X)
or
a = nd.empty(LARGE_X)
or
a = nd.random.exponential(shape=LARGE_X)
or
a = nd.random.gamma(shape=LARGE_X)
or
a = nd.random.normal(shape=LARGE_X)
```

## Caveats
1. Use `int64` as `dtype` whenever attempting to slice an NDArray when range is over maximum `int32` value
2. Use `int64` as `dtype` when passing indices as parameters or expecting output as parameters to and from operators

The following are the cases for large tensor usage where you must specify `dtype` as `int64`:


* _randint():_

```python
low_large_value = 2**32
high_large_value = 2**34
# dtype is explicitly specified since default type is int32 for randint
a = nd.random.randint(low_large_value, high_large_value, dtype=np.int64)
```

* _ravel_multi_index()_ and _unravel_index()_:

```python
x1, y1 = rand_coord_2d((LARGE_X - 100), LARGE_X, 10, SMALL_Y)
x2, y2 = rand_coord_2d((LARGE_X - 200), LARGE_X, 9, SMALL_Y)
x3, y3 = rand_coord_2d((LARGE_X - 300), LARGE_X, 8, SMALL_Y)
indices_2d = [[x1, x2, x3], [y1, y2, y3]]
# dtype is explicitly specified for indices else they will default to float32
idx = mx.nd.ravel_multi_index(mx.nd.array(indices_2d, dtype=np.int64),
                                  shape=(LARGE_X, SMALL_Y))
indices_2d = mx.nd.unravel_index(mx.nd.array(idx_numpy, dtype=np.int64),
                                  shape=(LARGE_X, SMALL_Y))
```

* _argsort()_ and _topk()_

They both return indices which are specified by `dtype=np.int64`.

```python
b = create_2d_tensor(rows=LARGE_X, columns=SMALL_Y)
# argsort
s = nd.argsort(b, axis=0, is_ascend=False, dtype=np.int64)
# topk
k = nd.topk(b, k=10, axis=0, dtype=np.int64)
```

* _index_copy()_

Again whenever we are passing indices as arguments and using large tensor, the `dtype` of indices must be `int64`.

```python
x = mx.nd.zeros((LARGE_X, SMALL_Y))
t = mx.nd.arange(1, SMALL_Y + 1).reshape((1, SMALL_Y))
# explicitly specifying dtype of indices to np.int64
index = mx.nd.array([LARGE_X - 1], dtype="int64")
x = mx.nd.contrib.index_copy(x, index, t)
```

* _one_hot()_

Here again array is used as indices that act as location of bits inside the large vector that need to be activated.

```python
# a is the index array here whose dtype should be int64.
a = nd.array([1, (VLARGE_X - 1)], dtype=np.int64)
b = nd.one_hot(a, VLARGE_X)
```

## What platforms and version of MXNet are supported ?
You can use MXNet with large tensor support in the following configuration:

*MXNet built for CPU on Linux (Ubuntu or Amazon Linux), and only for python bindings.*
*Custom wheels are provided with this configuration.*

These flavors of MXNet are currently built with large tensor support:

1. MXNet for linux-cpu
2. MXNet for linux_cu100

Large tensor support only works for *forward pass*. 
Backward pass is partially supported and not completely tested, so it is considered experimental at best.

Not supported:

* GPU. 
* Windows, ARM or any operating system other than Ubuntu
* Other language bindings like Scala, Java, R,  and Julia.


## Other known Issues:
* Randint operator is flaky: https://github.com/apache/mxnet/issues/16172.
* dgemm operations using BLAS libraries currently don’t support int64.
* linspace() is not supported.

```python
a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
c = 2 * a + b
texec = c.bind(mx.cpu(), {'a': nd.arange(0, LARGE_X * 2, dtype='int64').reshape(2, LARGE_X), 'b' : nd.arange(0, LARGE_X * 2, dtype='int64').reshape(2, LARGE_X)})
new_shape = {'a': (1, 2*LARGE_X), 'b': (1, 2*LARGE_X)}
texec.reshape(allow_up_sizing=True, **new_shape)

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/ubuntu/mxnet/python/mxnet/executor.py", line 449, in reshape
    py_array('i', provided_arg_shape_data)),
OverflowError: signed integer is greater than maximum}
```

Symbolic reshape is not supported. Please see the following example.

```python
a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
c = 2 * a + b
texec = c.bind(mx.cpu(), {'a': nd.arange(0, LARGE_X * 2, dtype='int64').reshape(2, LARGE_X), 'b' : nd.arange(0, LARGE_X * 2, dtype='int64').reshape(2, LARGE_X)})
new_shape = {'a': (1, 2 * LARGE_X), 'b': (1, 2 * LARGE_X)}
texec.reshape(allow_up_sizing=True, **new_shape)

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/ubuntu/mxnet/python/mxnet/executor.py", line 449, in reshape
    py_array('i', provided_arg_shape_data)),
OverflowError: signed integer is greater than maximum
```

## Working DGL Example(dgl.ai)
The following is a sample running code for DGL which works with int64 but not with int32.

```python
import mxnet as mx
from mxnet import gluon
import dgl
import dgl.function as fn
import numpy as np
from scipy import sparse as spsp

num_nodes = 10000000
num_edges = 100000000

col1 = np.random.randint(0, num_nodes, size=(num_edges,))
print('create col1')
col2 = np.random.randint(0, num_nodes, size=(num_edges,))
print('create col2')
data = np.ones((num_edges,))
print('create data')
spm = spsp.coo_matrix((data, (col1, col2)), shape=(num_nodes, num_nodes))
print('create coo')
labels = mx.nd.random.randint(0, 10, shape=(num_nodes,))

g = dgl.DGLGraph(spm, readonly=True)
print('create DGLGraph')
g.ndata['h'] = mx.nd.random.uniform(shape=(num_nodes, 200))
print('create node data')

class node_update(gluon.Block):
    def __init__(self, out_feats):
        super(node_update, self).__init__()
        self.dense = gluon.nn.Dense(out_feats, 'relu')
        self.dropout = 0.5

    def forward(self, nodes):
        h = mx.nd.concat(nodes.data['h'], nodes.data['accum'], dim=1)
        h = self.dense(h)
        return {'h': mx.nd.Dropout(h, p=self.dropout)}
update_fn = node_update(200)
update_fn.initialize(ctx=mx.cpu())

g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='accum'), update_fn)
print('update all')

loss_fcn = gluon.loss.SoftmaxCELoss()
loss = loss_fcn(g.ndata['h'], labels)
print('loss')
loss = loss.sum()
print(loss)
```

## Performance Regression: 
Roughly 40 operators have shown performance regression in our preliminary analysis: Large Tensor Performance as shown in table below.

|Operator                                |int32(msec)|int64(msec)  |int64/int32  |int32+mkl(msec)|int64+mkl(msec)|int64+mkl/int32+mkl|
|----------------------------------------|-----------|-------------|-------------|---------------|---------------|-------------------|
|topk                                    |12.81245198|42.2472195   |329.74%      |12.728027      |43.462353      |341.47%            |
|argsort                                 |16.43896801|46.2231455   |281.18%      |17.200311      |46.7779985     |271.96%            |
|sort                                    |16.57822751|46.5644815   |280.88%      |16.401236      |46.263803      |282.08%            |
|flip                                    |0.221817521|0.535838     |241.57%      |0.2123705      |0.7950055      |374.35%            |
|depth_to_space                          |0.250976998|0.534083     |212.80%      |0.2338155      |0.631252       |269.98%            |
|space_to_depth                          |0.254336512|0.5368935    |211.10%      |0.2334405      |0.6343175      |271.73%            |
|min_axis                                |0.685826526|1.4393255    |209.87%      |0.6266175      |1.3538925      |216.06%            |
|sum_axis                                |0.720809505|1.5110635    |209.63%      |0.6566265      |0.8290575      |126.26%            |
|nansum                                  |1.279337012|2.635434     |206.00%      |1.227156       |2.4305255      |198.06%            |
|argmax                                  |4.765146994|9.682672     |203.20%      |4.6576605      |9.394067       |201.69%            |
|swapaxes                                |0.667943008|1.3544455    |202.78%      |0.649036       |1.8293235      |281.85%            |
|argmin                                  |4.774890491|9.545651     |199.91%      |4.666858       |9.5194385      |203.98%            |
|sum_axis                                |0.540210982|1.0550705    |195.31%      |0.500895       |0.616179       |123.02%            |
|max_axis                                |0.117824005|0.226481     |192.22%      |0.149085       |0.224334       |150.47%            |
|argmax_channel                          |0.261897018|0.49573      |189.28%      |0.251171       |0.4814885      |191.70%            |
|min_axis                                |0.147698505|0.2675355    |181.14%      |0.148424       |0.2874105      |193.64%            |
|nansum                                  |1.142132009|2.058077     |180.20%      |1.042387       |1.263102       |121.17%            |
|min_axis                                |0.56951947 |1.020972     |179.27%      |0.4722595      |0.998179       |211.36%            |
|min                                     |1.154684491|2.0446045    |177.07%      |1.0534145      |1.9723065      |187.23%            |
|sum                                     |1.121753477|1.959272     |174.66%      |0.9984095      |1.213339       |121.53%            |
|sum_axis                                |0.158632494|0.2744115    |172.99%      |0.1573735      |0.2266315      |144.01%            |
|nansum                                  |0.21418152 |0.3661335    |170.95%      |0.2162935      |0.269517       |124.61%            |
|random_normal                           |1.229072484|2.093057     |170.30%      |1.222785       |2.095916       |171.41%            |
|LeakyReLU                               |0.344101485|0.582337     |169.23%      |0.389167       |0.7003465      |179.96%            |
|nanprod                                 |1.273265516|2.095068     |164.54%      |1.0906815      |2.054369       |188.36%            |
|nanprod                                 |0.203272473|0.32792      |161.32%      |0.202548       |0.3288335      |162.35%            |
|sample_gamma                            |8.079962019|12.7266385   |157.51%      |12.4216245     |12.7957475     |103.01%            |
|sum                                     |0.21571602 |0.3396875    |157.47%      |0.1939995      |0.262942       |135.54%            |
|argmin                                  |0.086381478|0.1354795    |156.84%      |0.0826235      |0.134886       |163.25%            |
|argmax                                  |0.08664903 |0.135826     |156.75%      |0.082693       |0.1269225      |153.49%            |
|sample_gamma                            |7.712843508|12.0266355   |155.93%      |11.8900915     |12.143009      |102.13%            |
|sample_exponential                      |2.312778   |3.5953945    |155.46%      |3.0935085      |3.5656265      |115.26%            |
|prod                                    |0.203170988|0.3113865    |153.26%      |0.180757       |0.264523       |146.34%            |
|random_uniform                          |0.40893798 |0.6240795    |152.61%      |0.244613       |0.6319695      |258.35%            |
|min                                     |0.205482502|0.3122025    |151.94%      |0.2023835      |0.33234        |164.21%            |
|random_negative_binomial                |3.919228504|5.919488     |151.04%      |5.685851       |6.0220735      |105.91%            |
|max                                     |0.212521001|0.3130105    |147.28%      |0.2039755      |0.2956105      |144.92%            |
|LeakyReLU                               |2.813424013|4.1121625    |146.16%      |2.719118       |5.613753       |206.45%            |
|mean                                    |0.242281501|0.344385     |142.14%      |0.209396       |0.313411       |149.67%            |
|Deconvolution                           |7.43279251 |10.4240845   |140.24%      |2.9548925      |5.812926       |196.72%            |
|abs                                     |0.273286481|0.38319      |140.22%      |0.3711615      |0.338064       |91.08%             |
|arcsinh                                 |0.155792513|0.2090985    |134.22%      |0.113365       |0.1702855      |150.21%            |
|sample_gamma                            |0.137634983|0.1842455    |133.87%      |0.1792825      |0.172175       |96.04%             |
|sort                                    |0.864107016|1.1560165    |133.78%      |0.8239285      |1.1454645      |139.02%            |
|argsort                                 |0.847259507|1.1320885    |133.62%      |0.842302       |1.1179105      |132.72%            |
|cosh                                    |0.129947497|0.1727415    |132.93%      |0.1192565      |0.1217325      |102.08%            |
|random_randint                          |0.822044531|1.085645     |132.07%      |0.6036805      |1.0953995      |181.45%            |
|arctanh                                 |0.119817996|0.1576315    |131.56%      |0.115616       |0.111907       |96.79%             |
|arccos                                  |0.185662502|0.2423095    |130.51%      |0.238534       |0.2351415      |98.58%             |
|mean                                    |1.758513477|2.2908485    |130.27%      |1.5868465      |2.530801       |159.49%            |
|erfinv                                  |0.142498524|0.184796     |129.68%      |0.1529025      |0.1538225      |100.60%            |
|degrees                                 |0.12517249 |0.1576175    |125.92%      |0.1166425      |0.1199775      |102.86%            |
|sample_exponential                      |0.07651851 |0.0960485    |125.52%      |0.0885775      |0.095597       |107.92%            |
|arctan                                  |0.120863522|0.1496115    |123.79%      |0.1161245      |0.17206        |148.17%            |
|prod                                    |1.147695002|1.408007     |122.68%      |1.0491025      |1.4065515      |134.07%            |
|fix                                     |0.073436997|0.089991     |122.54%      |0.0390455      |0.099307       |254.34%            |
|exp                                     |0.047701993|0.058272     |122.16%      |0.0397295      |0.0506725      |127.54%            |
