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

# An Intro: Manipulate Data the MXNet Way with NDArray

## Overview
This guide
will introduce you to how data is handled with MXNet. You will learn the basics
about MXNet's multi-dimensional array format, `ndarray`.

This content was extracted and simplified from the gluon tutorials in
[Dive Into Deep Learning](https://d2l.ai/).

## Prerequisites
* [MXNet installed in a Python environment](/get_started?version=master&platform=linux&language=python&environ=pip&processor=cpu).
* Python 2.7.x or Python 3.x


## Getting started

In this chapter, we'll get
you going with the basic functionality. Don't worry if you don't understand any
of the basic math, like element-wise operations or normal distributions. In the
next two chapters we'll take another pass at `NDArray`, teaching you both the math
you'll need and how to realize it in code.

To get started, let's import
`mxnet`. We'll also import `ndarray` from `mxnet` for convenience. We’ll make a
habit of setting a random seed so that you always get the same results that we
do.

```python
import mxnet as mx
from mxnet import nd
```

Let's start with a very simple 1-dimensional array with a python list.

```python
x = nd.array([1,2,3])
print(x)
```

Now a 2-dimensional array.

```python
y = nd.array([[1,2,3,4], [1,2,3,4], [1,2,3,4]])
print(y)
```

Next, let's see how to create an `NDArray`, without any values initialized.
Specifically, we'll create a 2D array (also called a *matrix*) with 3 rows and 4
columns using the `.empty` function. We'll also try out `.full` which takes an
additional parameter for what value you want to fill in the array.

```python
x = nd.empty((3, 3))
print(x)
x = nd.full((3,3), 7)
print(x)
```

`empty` just grabs some memory and hands us back a matrix without setting the
values of any of its entries. This means that the entries can have any form of
values, including very big ones! Typically, we'll want our matrices initialized
and very often we want a matrix of all zeros, so we can use the `.zeros`
function.

<!-- showing something
different here (3,10) since the zeros may not produce anything different from
empty... or use the two demonstrations to show something interesting or
unique... when would I use one over the other?-->

```python
x = nd.zeros((3, 10))
print(x)
```

Similarly, `ndarray` has a function to create a matrix of all ones aptly named
[ones](/api/python/docs/api/ndarray/ndarray.html#mxnet.ndarray.ones).

```python
x = nd.ones((3, 4))
print(x)
```

Often, we'll want to create arrays whose values are sampled randomly. This is
especially common when we intend to use the array as a parameter in a neural
network. In this snippet, we initialize with values drawn from a standard normal
distribution with zero mean and unit variance using
[random_normal](/api/python/docs/api/ndarray/ndarray.html#mxnet.ndarray.random_normal).

<!--
Is it that important to introduce zero mean and unit variance right now?
Describe more? Or how about explain which is which for the 0 and the 1 and what
they're going to do... if it actually matters at this point. -->

```python
y = nd.random_normal(0, 1, shape=(3, 4))
print(y)
```

Sometimes you will want to copy an array by its shape but not its contents. You
can do this with `.zeros_like`.

```python
z = nd.zeros_like(y)
print(z)
```

As in NumPy, the dimensions of each `NDArray` are accessible via the `.shape`
attribute.

```python
y.shape
```

We can also query its `.size`, which is equal to the product of the components
of the shape. Together with the precision of the stored values, this tells us
how much memory the array occupies.
<!-- is there a function for that or do you
just do it manually? Should we show that? -->

```python
y.size
```

We can query the data type using `.dtype`.

```python
y.dtype
```

`float32` is the default data type. Performance can be improved with less
precision, or you might want to use a different data type. You can force the
data type when you create the array using a numpy type. This requires you to
import numpy first.

```python
import numpy as np
a = nd.array([1,2,3])
b = nd.array([1,2,3], dtype=np.int32)
c = nd.array([1.2, 2.3], dtype=np.float16)
(a.dtype, b.dtype, c.dtype)
```

As you will come to learn in detail later, operations and memory storage will
happen on specific devices that you can set. You can compute on CPU(s), GPU(s), a
specific GPU, or all of the above depending on your situation and preference.
Using `.context` reveals the location of the variable.

```python
y.context
```

## Next Up

[NDArray Operations](02-ndarray-operations.md)
