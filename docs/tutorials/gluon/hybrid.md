# Hybrid - Faster training and easy deployment

*Related Content:*
* [Fast, portable neural networks with Gluon HybridBlocks](https://gluon.mxnet.io/chapter07_distributed-learning/hybridize.html)
* [A Hybrid of Imperative and Symbolic Programming
](http://en.diveintodeeplearning.org/chapter_computational-performance/hybridize.html)

Deep learning frameworks can be roughly divided into two categories: declarative
and imperative. With declarative frameworks (including Tensorflow, Theano, etc)
users first declare a fixed computation graph and then execute it end-to-end.
The benefit of fixed computation graph is it's portable and runs more
efficiently. However, it's less flexible because any logic must be encoded
into the graph as special operators like `scan`, `while_loop` and `cond`.
It's also hard to debug.

Imperative frameworks (including PyTorch, Chainer, etc) are just the opposite:
they execute commands one-by-one just like old fashioned Matlab and Numpy.
This style is more flexible, easier to debug, but less efficient.

`HybridBlock` seamlessly combines declarative programming and imperative programming
to offer the benefit of both. Users can quickly develop and debug models with
imperative programming and switch to efficient declarative execution by simply
calling: `HybridBlock.hybridize()`.

## HybridBlock

`HybridBlock` is very similar to `Block` but has a few restrictions:

- All children layers of `HybridBlock` must also be `HybridBlock`.
- Only methods that are implemented for both `NDArray` and `Symbol` can be used.
  For example you cannot use `.asnumpy()`, `.shape`, etc.
- Operations cannot change from run to run. For example, you cannot do `if x:`
  if `x` is different for each iteration.

To use hybrid support, we subclass the `HybridBlock`:

```python
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

mx.random.seed(42)

class Net(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            # layers created in name_scope will inherit name space
            # from parent layer.
            self.conv1 = nn.Conv2D(6, kernel_size=5)
            self.pool1 = nn.MaxPool2D(pool_size=2)
            self.conv2 = nn.Conv2D(16, kernel_size=5)
            self.pool2 = nn.MaxPool2D(pool_size=2)
            self.fc1 = nn.Dense(120)
            self.fc2 = nn.Dense(84)
            # You can use a Dense layer for fc3 but we do dot product manually
            # here for illustration purposes.
            self.fc3_weight = self.params.get('fc3_weight', shape=(10, 84))

    def hybrid_forward(self, F, x, fc3_weight):
        # Here `F` can be either mx.nd or mx.sym, x is the input data,
        # and fc3_weight is either self.fc3_weight.data() or
        # self.fc3_weight.var() depending on whether x is Symbol or NDArray
        print(x)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # 0 means copy over size from corresponding dimension.
        # -1 means infer size from the rest of dimensions.
        x = x.reshape((0, -1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dot(x, fc3_weight, transpose_b=True)
        return x
```

## Hybridize

By default, `HybridBlock` runs just like a standard `Block`. Each time a layer
is called, its `hybrid_forward` will be run:

```python
net = Net()
net.initialize()
x = mx.nd.random_normal(shape=(16, 1, 28, 28))
net(x)
x = mx.nd.random_normal(shape=(16, 1, 28, 28))
net(x)
```

Hybrid execution can be activated by simply calling `.hybridize()` on the top
level layer. The first forward call after activation will try to build a
computation graph from `hybrid_forward` and cache it. On subsequent forward
calls the cached graph, instead of `hybrid_forward`, will be invoked:

```python
net.hybridize()
x = mx.nd.random_normal(shape=(16, 1, 28, 28))
net(x)
x = mx.nd.random_normal(shape=(16, 1, 28, 28))
net(x)
```

Note that before hybridize, `print(x)` printed out one NDArray for forward,
but after hybridize, only the first forward printed out a Symbol. On subsequent
forward `hybrid_forward` is not called so nothing was printed.

Hybridize will speed up execution and save memory. If the top level layer is
not a `HybridBlock`, you can still call `.hybridize()` on it and Gluon will try
to hybridize its children layers instead.

`hybridize` also accepts several options for performance tuning. For example, you
can do

```python
net.hybridize(static_alloc=True)
# or
net.hybridize(static_alloc=True, static_shape=True)
```

Please refer to the [API manual](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html?highlight=hybridize#mxnet.gluon.Block.hybridize)
for details.

## Serializing trained model for deployment

Models implemented as `HybridBlock` can be easily serialized. The serialized
model can be loaded back later or used for deployment
with other language front-ends like C, C++ and Scala. To this end, we simply
use `export` and `SymbolBlock.imports`:

```python
net(x)
net.export('model', epoch=1)
```

Two files `model-symbol.json` and `model-0001.params` are saved on disk.
You can use other language bindings to load them. You can also load them back
to gluon with `SymbolBlock`:

```python
net2 = gluon.SymbolBlock.imports('model-symbol.json', ['data'], 'model-0001.params')
```

## Operators that do not work with hybridize

If you want to hybridize your model, you must use `F.some_operator` in your 'hybrid_forward' function.
`F` will be `mxnet.nd` before you hybridize and `mxnet.sym` after hybridize. While most APIs are the same in NDArray and Symbol, there are some differences. Writing `F.some_operator` and call `hybridize` may not work all of the time.
Here we list some frequently used NDArray APIs that can't be hybridized and provide you the work arounds.  

### Element-wise Operators

In NDArray APIs, the following arithmetic and comparison APIs are automatically broadcasted if the input NDArrays have different shapes.
However, that's not the case in Symbol API. It's not automatically broadcasted, and you have to manually specify to use another set of broadcast operators for Symbols expected to have different shapes.


| NDArray APIs  | Description  |
|---|---|
| [*NDArray.\__add\__*](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.__add__) | x.\__add\__(y) <=> x+y <=> mx.nd.add(x, y)  |
| [*NDArray.\__sub\__*](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.__sub__) | x.\__sub\__(y) <=> x-y <=> mx.nd.subtract(x, y)  |
| [*NDArray.\__mul\__*](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.__mul__) | x.\__mul\__(y) <=> x*y <=> mx.nd.multiply(x, y)  |
| [*NDArray.\__div\__*](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.__div__) | x.\__div\__(y) <=> x/y <=> mx.nd.divide(x, y)  |
| [*NDArray.\__mod\__*](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.__mod__) | x.\__mod\__(y) <=> x%y <=> mx.nd.modulo(x, y)  |
| [*NDArray.\__lt\__*](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.__lt__) |  x.\__lt\__(y) <=> x<y <=> x mx.nd.lesser(x, y) |
| [*NDArray.\__le\__*](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.__le__) |  x.\__le\__(y) <=> x<=y <=> mx.nd.less_equal(x, y) |
| [*NDArray.\__gt\__*](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.__gt__) |  x.\__gt\__(y) <=> x>y <=> mx.nd.greater(x, y) |
| [*NDArray.\__ge\__*](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.__ge__) |  x.\__ge\__(y) <=> x>=y <=> mx.nd.greater_equal(x, y)|
| [*NDArray.\__eq\__*](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.__eq__) |  x.\__eq\__(y) <=> x==y <=> mx.nd.equal(x, y) |
| [*NDArray.\__ne\__*](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.__ne__) |  x.\__ne\__(y) <=> x!=y <=> mx.nd.not_equal(x, y) |

The current workaround is to use corresponding broadcast operators for arithmetic and comparison to avoid potential hybridization failure when input shapes are different.

| Symbol APIs  | Description  |
|---|---|
|[*broadcast_add*](https://mxnet.incubator.apache.org/api/python/symbol/symbol.html#mxnet.symbol.broadcast_add) | Returns element-wise sum of the input arrays with broadcasting. |
|[*broadcast_sub*](https://mxnet.incubator.apache.org/api/python/symbol/symbol.html#mxnet.symbol.broadcast_sub) | Returns element-wise difference of the input arrays with broadcasting. |
|[*broadcast_mul*](https://mxnet.incubator.apache.org/api/python/symbol/symbol.html#mxnet.symbol.broadcast_mul) | Returns element-wise product of the input arrays with broadcasting. |
|[*broadcast_div*](https://mxnet.incubator.apache.org/api/python/symbol/symbol.html#mxnet.symbol.broadcast_div) | Returns element-wise division of the input arrays with broadcasting. |
|[*broadcast_mod*](https://mxnet.incubator.apache.org/api/python/symbol/symbol.html#mxnet.symbol.broadcast_mod) | Returns element-wise modulo of the input arrays with broadcasting. |
|[*broadcast_equal*](https://mxnet.incubator.apache.org/api/python/symbol/symbol.html#mxnet.symbol.broadcast_equal) | Returns the result of element-wise *equal to* (==) comparison operation with broadcasting. |
|[*broadcast_not_equal*](https://mxnet.incubator.apache.org/api/python/symbol/symbol.html#mxnet.symbol.broadcast_not_equal) | Returns the result of element-wise *not equal to* (!=) comparison operation with broadcasting. |
|[*broadcast_greater*](https://mxnet.incubator.apache.org/api/python/symbol/symbol.html#mxnet.symbol.broadcast_greater) | Returns the result of element-wise *greater than* (>) comparison operation with broadcasting. |
|[*broadcast_greater_equal*](https://mxnet.incubator.apache.org/api/python/symbol/symbol.html#mxnet.symbol.broadcast_greater_equal) | Returns the result of element-wise *greater than or equal to* (>=) comparison operation with broadcasting. |
|[*broadcast_lesser*](https://mxnet.incubator.apache.org/api/python/symbol/symbol.html#mxnet.symbol.broadcast_lesser) |	Returns the result of element-wise *lesser than* (<) comparison operation with broadcasting. |
|[*broadcast_lesser_equal*](https://mxnet.incubator.apache.org/api/python/symbol/symbol.html#mxnet.symbol.broadcast_lesser_equal) | Returns the result of element-wise *lesser than or equal to* (<=) comparison operation with broadcasting. |

For example, if you want to add a NDarray to your input x, use `broadcast_add` instead of `+`:

```python
def hybrid_forward(self, F, x):
    # avoid writing: return x + F.ones((1, 1))
    return F.broadcast_add(x, F.ones((1, 1)))
```

If you used `+`, it would still work before hybridization, but will throw an error of shape missmtach after hybridization.

### Shape

Gluon's imperative interface is very flexible and allows you to print the shape of the NDArray. However, Symbol does not have shape attributes. As a result, you need to avoid printing shapes in `hybrid_forward`.
Otherwise, you will get the following error:
```bash
AttributeError: 'Symbol' object has no attribute 'shape'
```

### Slice
`[]` in NDArray is used to get a slice from the array. However, `[]` in Symbol is used to get an output from a grouped symbol.
For example, you will get different results for the following method before and after hybridization.

```python
def hybrid_forward(self, F, x):
    return x[0]
```

The current workaround is to explicitly call [`slice`](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.slice) or [`slice_axis`](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.slice_axis) operators in `hybrid_forward`.


### Not implemented operators

Some of the often used operators in NDArray are not implemented in Symbol, and will cause hybridization failure.

#### NDArray.asnumpy
Symbol does not support the `asnumpy` function. You need to avoid calling `asnumpy` in `hybrid_forward`.

#### Array creation APIs

`mx.nd.array()` is used a lot, but Symbol does not have the `array` API. The current workaround is to use `F.ones`, `F.zeros`, or `F.full`, which exist in both the NDArray and Symbol APIs.

#### In-Place Arithmetic Operators

In-place arithmetic operators may be used in Gluon imperative mode, however if you expect to hybridize, you should write these operations explicitly instead.
For example, avoid writing `x += y` and use `x  = x + y`, otherwise you will get `NotImplementedError`. This applies to all the following operators:

| NDArray in-place arithmetic operators | Description |
|---|---|
|[*NDArray.\__iadd\__*](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.__iadd__) |	x.\__iadd\__(y) <=> x+=y |
|[*NDArray.\__isub\__*](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.__isub__) |	x.\__isub\__(y) <=> x-=y |
|[*NDArray.\__imul\__*](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.__imul__) |	x.\__imul\__(y) <=> x*=y |
|[*NDArray.\__idiv\__*](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.__idiv__) |	x.\__rdiv\__(y) <=> x/=y |
|[*NDArray.\__imod\__*](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.__imod__) |	x.\__rmod\__(y) <=> x%=y |



## Summary

The recommended practice is to utilize the flexibility of imperative NDArray API during experimentation. Once you finalized your model, make necessary changes mentioned above so you can call `hybridize` function to improve performance.

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->