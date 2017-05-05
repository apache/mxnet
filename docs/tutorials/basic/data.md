# MXNet Iterators - Load data for neural network training

This tutorial we focus on how to feeding data into a training and inference
program. Most training and inference modules in MXNet accepts data iterators,
which simplifies this procedure, especially when reading large datasets from
filesystems. Here we discuss the API conventions and several provided iterators.

## Basic Data Iterator

Data iterators in MXNet is similar to the iterator in Python. In Python, we can
use the built-in function `iter` with an iterable object (such as list) to
return an iterator. For example, in `x = iter([1, 2, 3])` we obtain an iterator
on the list `[1,2,3]`. If we repeatedly call `x.next()` (`__next__()` for Python
3), then we will get elements from the list one by one, and end with a
`StopIteration` exception.

MXNet's data iterator returns a batch of data in each `next` call. We first
introduce what a data batch looks like and then how to write a basic data
iterator.

### Data Batch

A data batch often contains *n* examples and the according labels. Here *n* is
often called as the batch size.

The following codes defines a valid data batch is able to be read by most
training/inference modules.

```python
class SimpleBatch(object):
    def __init__(self, data, label, pad=0):
        self.data = data
        self.label = label
        self.pad = pad
```

We explain what each attribute means:

- **`data`** is a list of NDArray, each of them has $n$ length first
  dimension. For example, if an example is an image with size $224 \times 224$
  and RGB channels, then the array shape should be `(n, 3, 224, 244)`.  Note
  that the image batch format used by MXNet is

  $$\textrm{batch_size} \times \textrm{num_channel} \times \textrm{height} \times
  \textrm{width}$$ The channels are often in RGB order.

  Each array will be copied into a free variable of the Symbol later. The
  mapping from arrays to free variables should be given by the `provide_data`
  attribute of the iterator, which will be discussed shortly.

- **`label`** is also a list of NDArray. Often each NDArray is a 1-dimensional
  array with shape `(n,)`. For classification, each class is represented by an
  integer starting from 0.

- **`pad`** is an integer shows how many examples are for merely used for
  padding, which should be ignored in the results. A nonzero padding is often
  used when we reach the end of the data and the total number of examples cannot
  be divided by the batch size.

### Symbol and Data Variables

Before moving the iterator, we first look at how to find which variables in a
Symbol are for input data. In MXNet, an operator (`mx.sym.*`) has one or more
input variables and output variables; some operators may have additional
auxiliary variables for internal states. For an input variable of an operator,
if do not assign it with an output of another operator during creating this
operator, then this input variable is free. We need to assign it with external
data before running.

The following codes define a simple multilayer perceptron (MLP) and then print
all free variables.


```python
import mxnet as mx
num_classes = 10
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=64)
net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=num_classes)
net = mx.sym.SoftmaxOutput(data=net, name='softmax')
print(net.list_arguments())
print(net.list_outputs())
```

As can be seen, we name a variable either by its operator's name if it is atomic
(e.g. `sym.Variable`) or by the `opname_varname` convention. The `varname` often
means what this variable is for:
- `weight` : the weight parameters
- `bias` : the bias parameters
- `output` : the output
- `label` : input label

On the above example, now we know that there are 4 variables for parameters, and
two for input data: `data` for examples and `softmax_label` for the according
labels.

The following example define a matrix factorization object function with rank 10
for recommendation systems. It has three input variables, `user` for user IDs,
`item` for item IDs, and `score` is the rating `user` gives to `item`.


```python
num_users = 1000
num_items = 1000
k = 10
user = mx.symbol.Variable('user')
item = mx.symbol.Variable('item')
score = mx.symbol.Variable('score')
# user feature lookup
user = mx.symbol.Embedding(data = user, input_dim = num_users, output_dim = k)
# item feature lookup
item = mx.symbol.Embedding(data = item, input_dim = num_items, output_dim = k)
# predict by the inner product, which is elementwise product and then sum
pred = user * item
pred = mx.symbol.sum_axis(data = pred, axis = 1)
pred = mx.symbol.Flatten(data = pred)
# loss layer
pred = mx.symbol.LinearRegressionOutput(data = pred, label = score)
```

### Data Iterators

Now we are ready to show how to create a valid MXNet data iterator. An iterator
should
1. return a data batch or raise a `StopIteration` exception if reaching the end
   when call `next()` in python 2 or `__next()__` in python 3
2. has `reset()` method to restart reading from the beginning
3. has `provide_data` and `provide_label` attributes, the former returns a list
   of `(str, tuple)` pairs, each pair stores an input data variable name and its
   shape. It is similar for `provide_label`, which provides information about
   input labels.

The following codes define a simple iterator that return some random data each
time.


```python
import numpy as np
class SimpleIter:
    def __init__(self, data_names, data_shapes, data_gen,
                 label_names, label_shapes, label_gen, num_batches=10):
        self._provide_data = zip(data_names, data_shapes)
        self._provide_label = zip(label_names, label_shapes)
        self.num_batches = num_batches
        self.data_gen = data_gen
        self.label_gen = label_gen
        self.cur_batch = 0

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            data = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_data, self.data_gen)]
            assert len(data) > 0, "Empty batch data."
            label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
            assert len(label) > 0, "Empty batch label."
            return SimpleBatch(data, label)
        else:
            raise StopIteration
```

Now we can feed the data iterator into a training problem. Here we used the
`Module` class.


```python
import logging
logging.basicConfig(level=logging.INFO)

n = 32
data = SimpleIter(['data'], [(n, 100)],
                  [lambda s: np.random.uniform(-1, 1, s)],
                  ['softmax_label'], [(n,)],
                  [lambda s: np.random.randint(0, num_classes, s)])

mod = mx.mod.Module(symbol=net)
mod.fit(data, num_epoch=5)
```

While for Symbol `pred`, we need to provide three inputs, two for examples and
one for label.


```python
data = SimpleIter(['user', 'item'],
                  [(n,), (n,)],
                  [lambda s: np.random.randint(0, num_users, s),
                   lambda s: np.random.randint(0, num_items, s)],
                  ['score'], [(n,)],
                  [lambda s: np.random.randint(0, 5, s)])

mod = mx.mod.Module(symbol=pred, data_names=['user', 'item'], label_names=['score'])
mod.fit(data, num_epoch=5)
```

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
