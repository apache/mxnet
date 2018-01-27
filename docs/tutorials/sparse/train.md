
# Train a Linear Regression Model with Sparse Symbols
In previous tutorials, we introduced `CSRNDArray` and `RowSparseNDArray`,
the basic data structures for manipulating sparse data.
MXNet also provides `Sparse Symbol` API, which enables symbolic expressions that handle sparse arrays.
In this tutorial, we first focus on how to compose a symbolic graph with sparse operators,
then train a linear regression model using sparse symbols with the Module API.

## Prerequisites

To complete this tutorial, we need:

- MXNet. See the instructions for your operating system in [Setup and Installation](http://mxnet.io/install/index.html).  

- [Jupyter Notebook](http://jupyter.org/index.html) and [Python Requests](http://docs.python-requests.org/en/master/) packages.
```
pip install jupyter requests
```

- Basic knowledge of Symbol in MXNet. See the detailed tutorial for Symbol in [Symbol - Neural Network Graphs and Auto-differentiation](https://mxnet.incubator.apache.org/tutorials/basic/symbol.html).

- Basic knowledge of CSRNDArray in MXNet. See the detailed tutorial for CSRNDArray in [CSRNDArray - NDArray in Compressed Sparse Row Storage Format](https://mxnet.incubator.apache.org/versions/master/tutorials/sparse/csr.html).

- Basic knowledge of RowSparseNDArray in MXNet. See the detailed tutorial for RowSparseNDArray in [RowSparseNDArray - NDArray for Sparse Gradient Updates](https://mxnet.incubator.apache.org/versions/master/tutorials/sparse/row_sparse.html).

## Variables

Variables are placeholder for arrays. We can use them to hold sparse arrays too.

### Variable Storage Types

The `stype` attribute of a variable is used to indicate the storage type of the array.
By default, the `stype` of a variable is "default" which indicates the default dense storage format.
We can specify the `stype` of a variable as "csr" or "row_sparse" to hold sparse arrays.


```python
import mxnet as mx
# Create a variable to hold an NDArray
a = mx.sym.Variable('a')
# Create a variable to hold a CSRNDArray
b = mx.sym.Variable('b', stype='csr')
# Create a variable to hold a RowSparseNDArray
c = mx.sym.Variable('c', stype='row_sparse')
(a, b, c)
```




    (<Symbol a>, <Symbol b>, <Symbol c>)



### Bind with Sparse Arrays

The sparse symbols constructed above declare storage types of the arrays to hold.
To evaluate them, we need to feed the free variables with sparse data.

You can instantiate an executor from a sparse symbol by using the `simple_bind` method,
which allocate zeros to all free variables according to their storage types.
The executor provides `forward` method for evaluation and an attribute
`outputs` to get all the results. Later, we will show the use of the `backward` method and other methods computing the gradients and updating parameters. A simple example first:


```python
shape = (2,2)
# Instantiate an executor from sparse symbols
b_exec = b.simple_bind(ctx=mx.cpu(), b=shape)
c_exec = c.simple_bind(ctx=mx.cpu(), c=shape)
b_exec.forward()
c_exec.forward()
# Sparse arrays of zeros are bound to b and c
print(b_exec.outputs, c_exec.outputs)
```

    ([
    <CSRNDArray 2x2 @cpu(0)>], [
    <RowSparseNDArray 2x2 @cpu(0)>])


You can update the array held by the variable by accessing executor's `arg_dict` and assigning new values.


```python
b_exec.arg_dict['b'][:] = mx.nd.ones(shape).tostype('csr')
b_exec.forward()
# The array `b` holds are updated to be ones
eval_b = b_exec.outputs[0]
{'eval_b': eval_b, 'eval_b.asnumpy()': eval_b.asnumpy()}
```




    {'eval_b': 
     <CSRNDArray 2x2 @cpu(0)>, 'eval_b.asnumpy()': array([[ 1.,  1.],
            [ 1.,  1.]], dtype=float32)}



## Symbol Composition and Storage Type Inference

### Basic Symbol Composition

The following example builds a simple element-wise addition expression with different storage types.
The sparse symbols are available in the `mx.sym.sparse` package.


```python
# Element-wise addition of variables with "default" stype
d = mx.sym.elemwise_add(a, a)
# Element-wise addition of variables with "csr" stype
e = mx.sym.sparse.negative(b)
# Element-wise addition of variables with "row_sparse" stype
f = mx.sym.sparse.elemwise_add(c, c)
{'d':d, 'e':e, 'f':f}
```




    {'d': <Symbol elemwise_add0>,
     'e': <Symbol negative0>,
     'f': <Symbol elemwise_add1>}



### Storage Type Inference

What will be the output storage types of sparse symbols? In MXNet, for any sparse symbol, the result storage types are inferred based on storage types of inputs.
You can read the [Sparse Symbol API](http://mxnet.io/versions/master/api/python/symbol/sparse.html) documentation to find what output storage types are. In the example below we will try out the storage types introduced in the Row Sparse and Compressed Sparse Row tutorials: `default` (dense), `csr`, and `row_sparse`.


```python
add_exec = mx.sym.Group([d, e, f]).simple_bind(ctx=mx.cpu(), a=shape, b=shape, c=shape)
add_exec.forward()
dense_add = add_exec.outputs[0]
# The output storage type of elemwise_add(csr, csr) will be inferred as "csr"
csr_add = add_exec.outputs[1]
# The output storage type of elemwise_add(row_sparse, row_sparse) will be inferred as "row_sparse"
rsp_add = add_exec.outputs[2]
{'dense_add.stype': dense_add.stype, 'csr_add.stype':csr_add.stype, 'rsp_add.stype': rsp_add.stype}
```




    {'csr_add.stype': 'csr',
     'dense_add.stype': 'default',
     'rsp_add.stype': 'row_sparse'}



### Storage Type Fallback

For operators that don't specialize in certain sparse arrays, you can still use them with sparse inputs with some performance penalty. In MXNet, dense operators require all inputs and outputs to be in the dense format. If sparse inputs are provided, MXNet will convert sparse inputs into dense ones temporarily so that the dense operator can be used. If sparse outputs are provided, MXNet will convert the dense outputs generated by the dense operator into the provided sparse format. Warning messages will be printed when such a storage fallback event happens.


```python
# `log` operator doesn't support sparse inputs at all, but we can fallback on the dense implementation
csr_log = mx.sym.log(a)
# `elemwise_add` operator doesn't support adding csr with row_sparse, but we can fallback on the dense implementation
csr_rsp_add = mx.sym.elemwise_add(b, c)
fallback_exec = mx.sym.Group([csr_rsp_add, csr_log]).simple_bind(ctx=mx.cpu(), a=shape, b=shape, c=shape)
fallback_exec.forward()
fallback_add = fallback_exec.outputs[0]
fallback_log = fallback_exec.outputs[1]
{'fallback_add': fallback_add, 'fallback_log': fallback_log}
```




    {'fallback_add': 
     [[ 0.  0.]
      [ 0.  0.]]
     <NDArray 2x2 @cpu(0)>, 'fallback_log': 
     [[-inf -inf]
      [-inf -inf]]
     <NDArray 2x2 @cpu(0)>}



### Inspecting Storage Types of the Symbol Graph (Work in Progress)

When the environment variable `MXNET_INFER_STORAGE_TYPE_VERBOSE_LOGGING` is set to `1`, MXNet will log the storage type information of
operators' inputs and outputs in the computation graph. For example, we can inspect the storage types of
a linear classification network with sparse operators as follows:


```python
# Set logging level for executor
import mxnet as mx
import os
os.environ['MXNET_INFER_STORAGE_TYPE_VERBOSE_LOGGING'] = "1"
# Data in csr format
data = mx.sym.var('data', stype='csr', shape=(32, 10000))
# Weight in row_sparse format
weight = mx.sym.var('weight', stype='row_sparse', shape=(10000, 2))
bias = mx.symbol.Variable("bias", shape=(2,))
dot = mx.symbol.sparse.dot(data, weight)
pred = mx.symbol.broadcast_add(dot, bias)
y = mx.symbol.Variable("label")
output = mx.symbol.SoftmaxOutput(data=pred, label=y, name="output")
executor = output.simple_bind(ctx=mx.cpu())
```

## Training with Module APIs

In the following section we'll walk through how one can implement **linear regression** using sparse symbols and sparse optimizers.

The function you will explore is: *y = x<sub>1</sub>  +  2x<sub>2</sub> + ... 100x<sub>100*, where *(x<sub>1</sub>,x<sub>2</sub>, ..., x<sub>100</sub>)* are input features and *y* is the corresponding label.

### Preparing the Data

In MXNet, both [mx.io.LibSVMIter](https://mxnet.incubator.apache.org/versions/master/api/python/io/io.html#mxnet.io.LibSVMIter)
and [mx.io.NDArrayIter](https://mxnet.incubator.apache.org/versions/master/api/python/io/io.html#mxnet.io.NDArrayIter)
support loading sparse data in CSR format. In this example, we'll use the `NDArrayIter`.

You may see some warnings from SciPy. You don't need to worry about those for this example.


```python
# Random training data
feature_dimension = 100
train_data = mx.test_utils.rand_ndarray((1000, feature_dimension), 'csr', 0.01)
target_weight = mx.nd.arange(1, feature_dimension + 1).reshape((feature_dimension, 1))
train_label = mx.nd.dot(train_data, target_weight)
batch_size = 1
train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size, last_batch_handle='discard', label_name='label')
```

### Defining the Model

Below is an example of a linear regression model specifying the storage type of the variables.


```python
initializer = mx.initializer.Normal(sigma=0.01)
X = mx.sym.Variable('data', stype='csr')
Y = mx.symbol.Variable('label')
weight = mx.symbol.Variable('weight', stype='row_sparse', shape=(feature_dimension, 1), init=initializer)
bias = mx.symbol.Variable('bias', shape=(1, ))
pred = mx.sym.broadcast_add(mx.sym.sparse.dot(X, weight), bias)
lro = mx.sym.LinearRegressionOutput(data=pred, label=Y, name="lro")
```

The above network uses the following symbols:

1. `Variable X`: The placeholder for sparse data inputs. The `csr` stype indicates that the array to hold is in CSR format.

2. `Variable Y`: The placeholder for dense labels.

3. `Variable weight`: The placeholder for the weight to learn. The `stype` of weight is specified as `row_sparse` so that it is initialized as RowSparseNDArray,
   and the optimizer will perform sparse update rules on it. The `init` attribute specifies what initializer to use for this variable.

4. `Variable bias`: The placeholder for the bias to learn.

5. `sparse.dot`: The dot product operation of `X` and `weight`. The sparse implementation will be invoked to handle `csr` and `row_sparse` inputs.

6. `broadcast_add`: The broadcasting add operation to apply `bias`.

7. `LinearRegressionOutput`: The output layer which computes *l2* loss against its input and the labels provided to it.

### Training the model

Once we have defined the model structure, the next step is to create a module and initialize the parameters and optimizer.


```python
# Create module
mod = mx.mod.Module(symbol=lro, data_names=['data'], label_names=['label'])
# Allocate memory by giving the input data and label shapes
mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
# Initialize parameters by random numbers
mod.init_params(initializer=initializer)
# Use SGD as the optimizer, which performs sparse update on "row_sparse" weight
sgd = mx.optimizer.SGD(learning_rate=0.05, rescale_grad=1.0/batch_size, momentum=0.9)
mod.init_optimizer(optimizer=sgd)
```

Finally, we train the parameters of the model to fit the training data by using the `forward`, `backward`, and `update` methods in Module.


```python
# Use mean square error as the metric
metric = mx.metric.create('MSE')
# Train 10 epochs
for epoch in range(10):
    train_iter.reset()
    metric.reset()
    for batch in train_iter:
        mod.forward(batch, is_train=True)       # compute predictions
        mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
        mod.backward()                          # compute gradients
        mod.update()                            # update parameters
    print('Epoch %d, Metric = %s' % (epoch, metric.get()))
assert metric.get()[1] < 1, "Achieved MSE (%f) is larger than expected (1.0)" % metric.get()[1]    
```

    Epoch 0, Metric = ('mse', 886.16457029229127)
    Epoch 1, Metric = ('mse', 173.16523056503445)
    Epoch 2, Metric = ('mse', 71.625164168341811)
    Epoch 3, Metric = ('mse', 29.625375983519298)
    Epoch 4, Metric = ('mse', 12.45004676561909)
    Epoch 5, Metric = ('mse', 6.9090727975622368)
    Epoch 6, Metric = ('mse', 3.0759215722750142)
    Epoch 7, Metric = ('mse', 1.3106610134811276)
    Epoch 8, Metric = ('mse', 0.63063102482907718)
    Epoch 9, Metric = ('mse', 0.35979430613957991)




### Training the model with multiple machines

To train a sparse model with multiple machines, please refer to the example in [mxnet/example/sparse/](https://github.com/apache/incubator-mxnet/tree/master/example/sparse)

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
