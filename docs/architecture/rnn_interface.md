# Survey of Existing Interfaces and Implementations

Commonly used deep learning libraries with good RNN/LSTM support include [Theano](http://deeplearning.net/software/theano/library/scan.html) and its wrappers [Lasagne](http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html) and [Keras](http://keras.io/layers/recurrent/); [CNTK](https://cntk.codeplex.com/); [TensorFlow](https://www.tensorflow.org/versions/master/tutorials/recurrent/index.html); and various implementations in Torch, such as [this well-known character-level language model tutorial](https://github.com/karpathy/char-rnn), [this](https://github.com/Element-Research/rnn).

In this document, we present a comparative analysis of the approaches taken by these libraries.

## Theano

In Theano, RNN support comes via its [scan operator](http://deeplearning.net/software/theano/library/scan.html),
which allows construction of a loop where the number of iterations is specified
as a runtime value of a symbolic variable.
You can find an official example of an LSTM implementation with scan
[here](http://deeplearning.net/tutorial/lstm.html).

### Implementation

I'm not very familiar with the Theano internals,
but it seems from [theano/scan_module/scan_op.py#execute](https://github.com/Theano/Theano/blob/master/theano/scan_module/scan_op.py#L1225)
that the scan operator is implemented with a loop in Python
that performs one iteration at a time:

```python
    fn = self.fn.fn

    while (i < n_steps) and cond:
        # ...
        fn()
```

The `grad` function in Theano constructs a symbolic graph for computing gradients. So the `grad` for the scan operator is actually implemented by [constructing another scan operator](https://github.com/Theano/Theano/blob/master/theano/scan_module/scan_op.py#L2527):

```python
    local_op = Scan(inner_gfn_ins, inner_gfn_outs, info)
    outputs = local_op(*outer_inputs)
```

The [performance guide](http://deeplearning.net/software/theano/library/scan.html#optimizing-scan-s-performance) for Theano's scan operator suggests minimizing the usage of the scan. This might be due to the fact that the loop is executed in Python, which might be a bit slow (due to context switching and the performance of Python itself). Moreover, because no unrolling is performed, the graph optimizer can't see the big picture.

If I understand correctly, when multiple RNN/LSTM layers are stacked, instead of a single loop with each iteration computing the whole feedforward network operation, the computation sequentially does a separate loop for each layer that uses the scan operator. If all of the intermediate values are stored to support computing the gradients, this is fine. Otherwise, using a single loop could be more memory efficient.

### Lasagne

The documentation for RNN in Lasagne can be found [here](http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html). In Lasagne, a recurrent layer is just like a standard layer, except that the input shape is expected to be `(batch_size, sequence_length, feature_dimension)`. The output shape is then `(batch_size, sequence_length, output_dimension)`.

Both `batch_size` and `sequence_length` are specified as `None`, and inferred from the data. Alternatively, when memory is sufficient and the (maximum) sequence length is known beforehand, you can set `unroll_scan` to `False`. Then Lasagne will unroll the graph explicitly, instead of using the Theano `scan` operator. Explicitly unrolling is implemented in [utils.py#unroll_scan](https://github.com/Lasagne/Lasagne/blob/master/lasagne/utils.py#L340).

The recurrent layer also accepts a `mask_input`, to support variable length sequences (e.g., when sequences within a mini-batch have different lengths. The mask has the shape `(batch_size, sequence_length)`.

### Keras

The documentation for RNN in Keras can be found [here](http://keras.io/layers/recurrent/). The interface in Keras is similar to the interface in Lasagne. The input is expected to be of shape `(batch_size, sequence_length, feature_dimension)`, and the output shape (if `return_sequences` is `True`) is `(batch_size, sequence_length, feature_dimension)`.

Keras currently supports both a Theano and a TensorFlow back end. RNN for the Theano back end is [implemented with the scan operator](https://github.com/fchollet/keras/blob/master/keras/backend/theano_backend.py#L432). For TensorFlow, it seems to be [implemented via explicitly unrolling](https://github.com/fchollet/keras/blob/master/keras/backend/tensorflow_backend.py#L396). The documentation says that for the TensorFlow back end, the sequence length must be specified beforehand, and masking is currently not working (because `tf.reduce_any` is not functioning yet).

## Torch

[karpathy/char-rnn](https://github.com/karpathy/char-rnn) is implemented by [explicitly unrolling](https://github.com/karpathy/char-rnn/blob/master/model/RNN.lua#L15). On the contrary, [Element-Research/rnn](https://github.com/Element-Research/rnn) runs sequence iteration in Lua. It actually has a very modular design:

* The basic RNN/LSTM modules run only *one* time step per one call of `forward` (and accumulate/store necessary information to support backward computation, if needed). You could have detailed control when using this API directly.
* A collection of `Sequencer`s are defined to model common scenarios, like forwarding sequence, bi-directional sequence, attention models, etc.
* There are other utility modules, like masking to support variable length sequences, etc.

## CNTK

CNTK looks quite different from other common deep learning libraries. I don't understand it very well. I will talk with Yu to get more details.

It seems that the basic data types are matrices (although there is also a `TensorView` utility class). The mini-batch data for sequence data is packed in a matrix with N-row being `feature_dimension` and N-column being `sequence_length * batch_size` (see Figure 2.9 on page 50 of the [CNTKBook](http://research.microsoft.com/pubs/226641/CNTKBook-20151201.pdf)).

Recurrent networks are first-class citizens in CNTK. In section 5.2.1.8 of the CNTKBook, you can see an example of a customized computation node. The node needs to explicitly define the functions for standard forward and forward with a time index, which is used for RNN evaluation:

```cpp
    virtual void EvaluateThisNode()
    {
        EvaluateThisNodeS(FunctionValues(), Inputs(0)->
            FunctionValues(), Inputs(1)->FunctionValues());
    }
    virtual void EvaluateThisNode(const size_t timeIdxInSeq)
    {
        Matrix<ElemType> sliceInputValue = Inputs(1)->
            FunctionValues().ColumnSlice(timeIdxInSeq *
            m_samplesInRecurrentStep, m_samplesInRecurrentStep);
        Matrix<ElemType> sliceOutputValue =    m_functionValues.
            ColumnSlice(timeIdxInSeq * m_samplesInRecurrentStep,
            m_samplesInRecurrentStep);
        EvaluateThisNodeS(sliceOutputValue, Inputs(0)->
            FunctionValues(), sliceInput1Value);
    }
```

The function `ColumnSlice(start_col, num_col)` takes out the packed data for that time index, as described above (here `m_samplesInRecurrentStep` must be the mini-batch size).

The low-level API for recurrent connection seem to be a *delay node*. But I'm not sure how to use this low-level API. The [example of PTB language model](https://cntk.codeplex.com/SourceControl/latest#Examples/Text/PennTreebank/Config/rnn.config) uses a very high-level API (simply setting `recurrentLayer = 1` in the config).

## TensorFlow

The [current example of RNNLM](https://www.tensorflow.org/versions/master/tutorials/recurrent/index.html#recurrent-neural-networks) in TensorFlow uses explicit unrolling for a predefined number of time steps. The white-paper mentions that an advanced control flow API (Theano's scan-like) is planned.

## Next Steps

* [MXNet System Overview](http://mxnet.io/architecture/overview.html)
