# Iterators - Loading data

This tutorial we focus on how to feeding data into a training and inference
program. Most training and inference modules in MXNet accepts data iterators,
which simplifies this procedure, especially when reading large datasets from
filesystems. Here we discuss the API conventions and several provided iterators.

Data iterators in MXNet is similar to the iterator in Python. In Python, we can
use the built-in function `iter` with an iterable object (such as list) to
return an iterator. For example, in `x = iter([1, 2, 3])` we obtain an iterator
on the list `[1,2,3]`. If we repeatedly call `x.next()` (`__next__()` for Python
3), then we will get elements from the list one by one, and end with a
`StopIteration` exception.

## Introduction

### Data Batch

A data iterator returns a batch of data in each `next` call.
A batch often contains *n* examples and the according labels. Here *n* is
called as the batch size.

The following codes define a simple data batch that is able to be read by most
training/inference modules.

```python
class SimpleBatch(object):
    def __init__(self, data, label, pad=0):
        self.data = data
        self.label = label
        self.pad = pad
```

We explain what each attribute means:

- `data` is a list of `NDArray`, each array contains *n* examples. For
  instance, if an example is presented by a length $k$ vector, then the shape of
  the array will be `(n, k)`.

  Each array will be copied into a free variable such as created by
  `mx.sym.Variable()` later. The mapping from arrays to free variables should be
  given by the `provide_data` attribute of the iterator, which will be discussed
  shortly.

- `label` is also a list of `NDArray`. Often each array is a 1-dimensional
  array with shape `(n,)`. For classification, each class is represented by an
  integer starting from 0.

- `pad` is an integer shows the number of examples added in the last of the
  batch that are merely used for padding. These examples should be ignored in
  the results, such as computing the gradient. A nonzero padding is often used
  when we reach the end of the data and the total number of examples cannot be
  divided by the batch size.

### Data Variables

Before showing the data iterator, we first discuss how to find free variables in
a symbol. A symbol often contains one or more explicit free variables and also
implicit ones.

The following codes define a multilayer perceptron.

```python
import mxnet as mx
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=64)
net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=10)
net = mx.sym.SoftmaxOutput(data=net, name='softmax')
```

We can get the names of the all free variables by calling `list_arguments`:

```python
net.list_arguments()
```

As can be seen, we name a variable either by its operator's name if it is atomic
(e.g. `Variable`) or by the `opname_varname` convention, where `opname` is the
operator's name and `varname` is assigned by the operator. The `varname`
often means what this variable is for:

- `weight` : the weight parameters
- `bias` : the bias parameters
- `output` : the output
- `label` : input label

On the above example, now we know that there are 6 variables for free
variables. Four of them are learnable parameters, `fc1_weight`, `fc1_bias`,
`fc2_weight`, and `fc2_bias`. These parameters are often initialized by
`mx.initializer` and updated by `mx.optimizer`. The rest two
are for input data: `data` for examples and `softmax_label` for the
according labels. Then it is the iterator's job to fed data into these two
variables.

### Data iterator

An iterator in _MXNet_ should

1. return a data batch or raise a `StopIteration` exception if reaching the end
   when call `next()` in python 2 or `__next()__` in python 3
2. has `reset()` method to restart reading from the beginning
3. has `provide_data` and `provide_label` attributes, the former returns a list
   of `(str, tuple)` pairs, each pair stores an input data variable name and its
   shape. It is similar for `provide_label`, which provides information about
   input labels.

On the above example,  assume the data batch size is *(n,k)* and label size is
*(n,1)*, the iterator for `net` should have `provide_data` to return
`[('data', (n,k))]` and `provide_label` to return
`[('softmax_label', (n,))]`. An training or inference program will then know how
to assign the arrays in a data batch into the input variables.

## Read array

When data are already in memory and stored by either `NDArray` or numpy ndarray,
we can create an iterator by `NDArrayIter`:

```python
import numpy as np
data = np.random.rand(100,3)
label = np.random.randint(0, 10, (100,))
data_iter = mx.io.NDArrayIter(data=data, label=label, batch_size=30)
for batch in data_iter:
    print([batch.data, batch.label, batch.pad])
```

## Read CSV

There is an iterator called to `CSVIter` to read data batches from CSV files. We
first dump `data` into a csv file, and then load the data

```python
np.savetxt('data.csv', data, delimiter=',')
data_iter = mx.io.CSVIter(data_csv='data.csv', data_shape=(3,), batch_size=30)
for batch in data_iter:
    print([batch.data, batch.pad])
```

Note that we have not given a label file, then `batch.label` is empty here.

## Read images

<!-- TODO(mli) move notebooks here -->

- [Read images](https://github.com/dmlc/mxnet-notebooks/blob/master/python/basic/image_io.ipynb)
- [Advanced image reading](https://github.com/dmlc/mxnet-notebooks/blob/master/python/basic/advanced_img_io.ipynb)

## Write your own data iterators

Sometimes the provided iterators are not enough for some application. There are
mainly two ways to develop a new iterator. One is creating from scratch, the
following codes define an iterator that creates a given number of data batches
through a data generator `data_gen`.

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
            label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
            return SimpleBatch(data, label)
        else:
            raise StopIteration
```

But in most cases we can reuse the existing data iterators. For example, in the
image caption application, an input example is an image while the label is a
sentence. Then we can create

- `image_iter` by `ImageRecordIter` so we can enjoy the provided multithreaded
pre-fetch and augmentation.
- `caption_iter` by `NDArrayIter` or bucketing iterator provided in the `rnn`
package.

Next we create an iterator whose `next()` function will both
`image_iter.next()` and `caption_iter.next()` and return the combined results.


<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
