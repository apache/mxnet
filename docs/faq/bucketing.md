# Bucketing in MXNet
When we train recurrent neural networks (RNNs), we _unroll_ the network in time.
For a single example of length T, we would unroll the network T steps.
In the unrolled view, the network is acyclic and thus we can think of it
as a feedforward neural network in which weights are tied across adjacent time steps.
The unrolled view allows us to train the network via the traditional backpropagation algorithm -
in this case, we call the algorithm _back-propagation through time_.

Things get complicated when we work with datasets where the example sequences have varying lengths.
In the _recurrent_ view, each example can pass through the same architecture.
But in the _unrolled_ view, each example requires a different number of unrollings, and thus corresponds to a _slightly_ different feedforward network.
If we train on one example at a time, we can simply unroll the desired amount on each training iteration.
But things get more complicated when we try to perform mini-batch training.
A naive approach might be to pad all the sequences so that they appear to have the length of the longest example.
However, this could be wasteful because on shorter sequences, most of the computations are done on padded data.

Borrowed from [TensorFlow's sequence training code](https://www.tensorflow.org/versions/r0.7/tutorials/seq2seq/index.html),
_bucketing_ offers an effective solution to make minibatches out of varying-length sequences.
Instead of unrolling the network to the maximum possible sequence length,
we unroll multiple instances of different lengths (e.g., length 5, 10, 20, 30).
During training, we use the most appropriate unrolled model
for each mini-batch of data.
Although the models *with different numbers of unrollings*
have different feedforward architectures,
their parameters are shared in time.
_MXNet_ reuses the internal memory buffers among all executors.

For simple RNNs, you can use a for loop to explicitly
go over the input sequences and perform a backpropagation-through-time
by maintaining the connection of the states and gradients through time.
However, this implementation approach results in slow processing.
This approach works with variable length sequences. For more complicated models (e.g., translation that uses a sequence-to-sequence model), explicitly unrolling is the easiest way. In this example, we introduce the MXNet APIs that allows us to implement bucketing.

## Variable-length Sequence Training for PTB

We use the [PennTreeBank language model example](https://github.com/dmlc/mxnet/tree/master/example/rnn) for this example. If you are not familiar with this example, see [this tutorial (in Julia)](http://dmlc.ml/mxnet/2015/11/15/char-lstm-in-julia.html) first.

In this example, we use a simple architecture
consisting of a word-embedding layer
followed by two LSTM layers.
In the original example,
the model is unrolled explicitly in time for a fixed length of 32.
 <!-- In this tutorial, we show how to use bucketing to implement variable-length sequence training. -->
To enable bucketing, MXNet needs to know how to construct a new unrolled symbolic architecture for a different sequence length. To achieve this, instead of constructing a model with a fixed `Symbol`, we use a callback function that generates a new `Symbol` on a *bucket key*.


```python
model = mx.mod.BucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = data_train.default_bucket_key,
        context             = contexts)
```

`sym_gen` must be a function that takes one argument, `bucket_key`, and returns a `Symbol` for this bucket. We'll use the sequence length as the bucket key. A bucket key could be anything. For example, in neural translation, because different combinations of input-output sequence lengths correspond to different unrolling, the bucket key could be a pair of lengths.

```python
def sym_gen(seq_len):
    return lstm_unroll(num_lstm_layer, seq_len, len(vocab),
                       num_hidden=num_hidden, num_embed=num_embed,
                       num_label=len(vocab))
```
The data iterator needs to report the `default_bucket_key`, which allows MXNet to do some parameter initialization before reading the data. Now the model is capable of training with different buckets by sharing the parameters and intermediate computation buffers between bucket executors.

To train, we still need to add some extra bits to our `DataIter`. Apart from reporting the `default_bucket_key` as mentioned previously, we also need to report the current `bucket_key` for each mini-batch. More specifically, the `DataBatch` object returned in each mini-batch by the `DataIter` should contain the following additional properties:

* `bucket_key`: The bucket key that corresponds to this batch of data.
In our example, it is the sequence length for this batch of data.
If the executors corresponding to this bucket key have not yet been created,
they will be constructed according to the symbolic model returned by `gen_sym` on this bucket key.
The executors will be cached for future use.
Note that generated `Symbol`s could be arbitrary,
but they should all have the same trainable parameters and auxiliary states.
* `provide_data`: The same information reported by the `DataIter` object.
Because each bucket now corresponds to a different architecture,
they could have different input data.
Also, make sure that the `provide_data` information returned by the `DataIter` object
is compatible with the architecture for `default_bucket_key`.
* `provide_label`: The same as `provide_data`.

The `DataIter` is responsible for grouping the data into different buckets.
Assuming that randomization is enabled, at each iteration,
`DataIter` chooses a random bucket (according to a distribution balanced by the bucket sizes),
and then randomly chooses sequences from that bucket to form a mini-batch.
It also applies padding for sequences of different length within the mini-batch as necessary.

For a full, working implementation of a `DataIter`
that reads text sequences by as described above, see [example/rnn/lstm_ptb_bucketing.py](https://github.com/dmlc/mxnet/blob/master/example/rnn/bucketing/lstm_bucketing.py).
In this example, you can use bucketing with a static configuration (e.g., `buckets = [10, 20, 30, 40, 50, 60]`), or let MXNet generate buckets automatically according to the characteristics of the dataset (`buckets = []`). The latter approach is implemented by adding a bucket as long as the number of sequences assigned to that bucket is exceeds some minimum count. For more information, see [default_gen_buckets()](https://github.com/dmlc/mxnet/blob/master/example/rnn/old/bucket_io.py#L43).

## Beyond Sequence Training

In this example, we briefly explained how the bucketing API works.
However, the API is not limited to bucketing by sequence lengths.
The bucket key can be an arbitrary object, as long
as the architecture returned by `gen_sym`
is compatible with (has the same set of parameters) as the object.
