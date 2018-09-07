# Hybridize Gluon models with control flows.

MXNet currently provides three control flow operators: `cond`, `foreach` and `while_loop`. Like other MXNet operators, they all have a version for NDArray and a version for Symbol. These two versions have exactly the same semantics. We can take advantage of this and use them in Gluon to hybridize models.

In this tutorial, we use a few examples to demonstrate the use of control flow operators in Gluon and show how a model that requires control flow is hybridized.

## Prepare running the code


```python
import mxnet as mx
from mxnet.gluon import HybridBlock
```

## foreach
`foreach` is a for loop that iterates over the first dimension of the input data (it can be an array or a list of arrays). It is defined with the following signature:

```python
foreach(body, data, init_states, name) => (outputs, states)
```

It runs the Python function defined in `body` for every slice from the input arrays. The signature of the `body` function is defined as follows:

```python
body(data, states) => (outputs, states)
```

The inputs of the `body` function have two parts: `data` is a slice of an array (if there is only one input array in `foreach`) or a list of slices (if there are a list of input arrays); `states` are the arrays from the previous iteration. The outputs of the `body` function also have two parts: `outputs` is an array or a list of arrays; `states` is the computation states of the current iteration. `outputs` from all iterations are concatenated as the outputs of `foreach`.

The following pseudocode illustrates the execution of `foreach`.

```python
def foreach(body, data, init_states):
    states = init_states
    outs = []

    for i in range(data.shape[0]):
        s = data[i]
        out, states = body(s, states)
        outs.append(out)
    outs = mx.nd.stack(*outs)
    return outs, states
```

### Example 1: `foreach` works like map
`foreach` can work like a map function of a functional language. In this case, the states of `foreach` can be an empty list, which means the computation doesn't carry computation states across iterations.

In this example, we use `foreach` to increase each element's value of an array by one.


```python
data = mx.nd.arange(5)
print(data)
```

    
    [ 0.  1.  2.  3.  4.]
    <NDArray 5 @cpu(0)>



```python
def add1(data, _):
    return data + 1, []

class Map(HybridBlock):
    def hybrid_forward(self, F, data):
        out, _ = F.contrib.foreach(add1, data, [])
        return out
    
map_layer = Map()
out = map_layer(data)
print(out)
```

    
    [[ 1.]
     [ 2.]
     [ 3.]
     [ 4.]
     [ 5.]]
    <NDArray 5x1 @cpu(0)>


We can hybridize the block and run the computation again. It should generate the same result.


```python
map_layer.hybridize()
out = map_layer(data)
print(out)
```

    
    [[ 1.]
     [ 2.]
     [ 3.]
     [ 4.]
     [ 5.]]
    <NDArray 5x1 @cpu(0)>


### Example 2: `foreach` works like scan
`foreach` can work like a scan function in a functional language. In this case, the outputs of the Python function is an empty list.


```python
def sum(data, state):
    return [], state + data

class Scan(HybridBlock):
    def hybrid_forward(self, F, data):
        _, state = F.contrib.foreach(sum, data, F.zeros((1)))
        return state
scan_layer = Scan()
state = scan_layer(data)
print(data)
print(state)
```

    
    [ 0.  1.  2.  3.  4.]
    <NDArray 5 @cpu(0)>
    
    [ 10.]
    <NDArray 1 @cpu(0)>



```python
scan_layer.hybridize()
state = scan_layer(data)
print(state)
```

    
    [ 10.]
    <NDArray 1 @cpu(0)>


### Example 3: `foreach` with both outputs and states
This is probably the most common use case of `foreach`. We extend the previous scan example and return both output and states.


```python
def sum(data, state):
    return state + data, state + data

class ScanV2(HybridBlock):
    def hybrid_forward(self, F, data):
        out, state = F.contrib.foreach(sum, data, F.zeros((1)))
        return out, state
scan_layer = ScanV2()
out, state = scan_layer(data)
print(out)
print(state)
```

    
    [[  0.]
     [  1.]
     [  3.]
     [  6.]
     [ 10.]]
    <NDArray 5x1 @cpu(0)>
    
    [ 10.]
    <NDArray 1 @cpu(0)>



```python
scan_layer.hybridize()
out, state = scan_layer(data)
print(out)
print(state)
```

    
    [[  0.]
     [  1.]
     [  3.]
     [  6.]
     [ 10.]]
    <NDArray 5x1 @cpu(0)>
    
    [ 10.]
    <NDArray 1 @cpu(0)>


### Example 4: use `foreach` to run an RNN on a variable-length sequence
Previous examples illustrate `foreach` with simple use cases. Here we show an example of processing variable-length sequences with `foreach`. The same idea is used by `dynamic_rnn` in TensorFlow for processing variable-length sequences.


```python
class DynamicRNNLayer(HybridBlock):
    def __init__(self, cell, prefix=None, params=None):
        super(DynamicRNNLayer, self).__init__(prefix=prefix, params=params)
        self.cell = cell
    def hybrid_forward(self, F, inputs, begin_state, valid_length):
        states = begin_state
        zeros = []
        for s in states:
            zeros.append(F.zeros_like(s))
        # the last state is the iteration number.
        states.append(F.zeros((1)))
        def loop_body(inputs, states):
            cell_states = states[:-1]
            # Get the iteration number from the states.
            iter_no = states[-1]
            out, new_states = self.cell(inputs, cell_states)
            # Copy the old state if we have reached the end of a sequence.
            for i, state in enumerate(cell_states):
                new_states[i] = F.where(F.broadcast_greater(valid_length, iter_no),
                                        new_states[i], state)
            new_states.append(iter_no + 1)
            return out, new_states

        outputs, states = F.contrib.foreach(loop_body, inputs, states)
        outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                 use_sequence_length=True, axis=0)
        # the last state is the iteration number. We don't need it.
        return outputs, states[:-1]


seq_len = 10
batch_size = 2
input_size = 5
hidden_size = 6

rnn_data = mx.nd.normal(loc=0, scale=1, shape=(seq_len, batch_size, input_size))
init_states = [mx.nd.normal(loc=0, scale=1, shape=(batch_size, hidden_size)) for i in range(2)]
valid_length = mx.nd.round(mx.nd.random.uniform(low=1, high=10, shape=(batch_size))) 

lstm = DynamicRNNLayer(mx.gluon.rnn.LSTMCell(hidden_size))
lstm.initialize()
res, states = lstm(rnn_data, [x for x in init_states], valid_length)

lstm.hybridize()
res, states = lstm(rnn_data, [x for x in init_states], valid_length)
```

## while_loop
`while_loop` defines a while loop. It has the following signature:

```python
while_loop(cond, body, loop_vars, max_iterations, name) => (outputs, states)
```

Instead of running over the first dimension of an array, `while_loop` checks a condition function in every iteration and runs a `body` function for computation. The signature of the `body` function is defined as follows:

```python
body(state1, state2, ...) => (outputs, states)
```

The inputs of the `body` function in `while_loop` are a little different from the one in `foreach`. It has a variable number of input arguments. Each input argument is a loop variable and the number of arguments is determined by the number of loop variables. The outputs of the `body` function also have two parts: `outputs` is an array or a list of arrays; `states` are loop variables and will be passed to the next iteration as inputs of `body`. Like `foreach`, both `outputs` and `states` can be an empty list. `outputs` from all iterations are concatenated as the outputs of `while_loop`.

### Example 5: scan with while_loop
`while_loop` is more general than `foreach`. We can also use it to iterate over an array and sum all of its values together. In this example, instead of summing over the entire array, we only sum over the first 4 elements.

**Note**: the output arrays of the current implementation of `while_loop` is determined by `max_iterations`. As such, even though the while loop in this example runs 4 iterations, it still outputs an array of 5 elements. The last element in the output array is actually filled with an arbitrary value.


```python
class ScanV2(HybridBlock):
    def hybrid_forward(self, F, data):
        def sum(state, i):
            s = state + data[i]
            return s, [s, i + 1]

        def sum_cond(state, i):
            return i < 4

        out, state = F.contrib.while_loop(sum_cond, sum,
                                          [F.zeros((1)), F.zeros((1))], max_iterations=5)
        return out, state
scan_layer = ScanV2()
out, state = scan_layer(data)
print(out)
print(state)
```

    
    [[ 0.]
     [ 1.]
     [ 3.]
     [ 6.]
     [ 0.]]
    <NDArray 5x1 @cpu(0)>
    [
    [ 6.]
    <NDArray 1 @cpu(0)>, 
    [ 4.]
    <NDArray 1 @cpu(0)>]


## cond
`cond` defines an if condition. It has the following signature:

```python
cond(pred, then_func, else_func, name)
```

`cond` checks `pred`, which is a symbol or an NDArray with one element. If its value is true, it calls `then_func`. Otherwise, it calls `else_func`. The signature of `then_func` and `else_func` are as follows:

```python
func() => [outputs]
```

`cond` requires all outputs from `then_func` and `else_func` have the same number of Symbols/NDArrays with the same shapes and data types.

### Example 6: skip RNN computation with cond
Example 4 shows how to process a batch with sequences of different lengths. It performs computation for all steps but discards some of the computation results.

In this example, we show how to skip computation after we have reached the end of a sequence, whose length is indicated by `length`. The code below only works for a batch with one sequence.


```python
class SkipRNNCell(HybridBlock):
    def __init__(self, cell, prefix=None, params=None):
        super(SkipRNNCell, self).__init__(prefix=prefix, params=params)
        self.cell = cell
    def hybrid_forward(self, F, i, length, data, states):
        def run_rnn():
            return self.cell(data, states)

        def copy_states():
            return F.zeros_like(data), states
        out, state = F.contrib.cond(i < length, run_rnn, copy_states)
        return out, state

class RNNLayer(HybridBlock):
    def __init__(self, cell, prefix=None, params=None):
        super(RNNLayer, self).__init__(prefix=prefix, params=params)
        self.cell = SkipRNNCell(cell)
    def hybrid_forward(self, F, length, data, init_states):
        def body(data, states):
            i = states[0]
            out, states = self.cell(i, length, data, states[1])
            return out, [i + 1, states]
        print()
        out, state = F.contrib.foreach(body, data, [F.zeros((1)), init_states])
        return out, state


seq_len = 5
batch_size = 1
input_size = 3
hidden_size = 3

rnn_data = mx.nd.normal(loc=0, scale=1, shape=(seq_len, batch_size, input_size))
init_states = [mx.nd.normal(loc=0, scale=1, shape=(batch_size, hidden_size)) for i in range(2)]

cell = mx.gluon.rnn.LSTMCell(hidden_size)
layer = RNNLayer(cell)
layer.initialize()

out, states = layer(mx.nd.array([3]), rnn_data, init_states)
print(rnn_data)
print(out)
```

    ()
    
    [[[-1.25296438  0.387312   -0.41055229]]
    
     [[ 1.28453672  0.21001032 -0.08666432]]
    
     [[ 1.46422136 -1.30581355  0.9344402 ]]
    
     [[ 0.5380863  -0.16038011  0.84187603]]
    
     [[-1.00553632  3.13221502 -0.4358989 ]]]
    <NDArray 5x1x3 @cpu(0)>
    
    [[[-0.02620504  0.1605694   0.29636264]]
    
     [[-0.00474182  0.08719197  0.17757624]]
    
     [[ 0.00631597  0.04674901  0.12468992]]
    
     [[ 0.          0.          0.        ]]
    
     [[ 0.          0.          0.        ]]]
    <NDArray 5x1x3 @cpu(0)>


<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
