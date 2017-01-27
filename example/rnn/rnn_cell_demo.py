"""A simple demo of new RNN cell with PTB language model."""

import os

import numpy as np
import mxnet as mx

from bucket_io import BucketSentenceIter, default_build_vocab


data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))


def Perplexity(label, pred):
    # TODO(tofix): we make a transpose of label here, because when
    # using the RNN cell, we called swap axis to the data.
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)

class RNNXavier(mx.init.Xavier):
    """Initialize the weight with Xavier or similar initialization scheme.
    
    Prepared specially for RNN parameters.
    Parameters
    ----------
    rnd_type: str, optional
        Use ```gaussian``` or ```uniform``` to init

    factor_type: str, optional
        Use ```avg```, ```in```, or ```out``` to init

    magnitude: float, optional
        scale of random number range
    """
    def __init__(self, rnd_type="uniform", factor_type="avg", magnitude=3):
        super().__init__(rnd_type, factor_type, magnitude)

    def _init_default(self, name, arr):
        super()._init_weight(name, arr)

if __name__ == '__main__':
    batch_size = 128
    buckets = [10, 20, 30, 40, 50, 60]
    num_hidden = 200
    num_embed = 200
    num_lstm_layer = 2

    num_epoch = 20
    learning_rate = 0.01
    momentum = 0.0

    contexts = [mx.context.gpu(i) for i in range(1)]
    vocab = default_build_vocab(os.path.join(data_dir, 'ptb.train.txt'))

    init_h = [('LSTM_init_h', (batch_size, num_lstm_layer, num_hidden))]
    init_c = [('LSTM_init_c', (batch_size, num_lstm_layer, num_hidden))]
    init_states = init_c + init_h

    data_train = BucketSentenceIter(os.path.join(data_dir, 'ptb.train.txt'),
                                    vocab, buckets, batch_size, init_states)
    data_val = BucketSentenceIter(os.path.join(data_dir, 'ptb.valid.txt'),
                                  vocab, buckets, batch_size, init_states)

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        embed = mx.sym.Embedding(data=data, input_dim=len(vocab),
                                 output_dim=num_embed, name='embed')

        # TODO(tofix)
        # The inputs and labels from IO are all in batch-major.
        # We need to transform them into time-major to use RNN cells.
        embed_tm = mx.sym.SwapAxis(embed, dim1=0, dim2=1)
        label_tm = mx.sym.SwapAxis(label, dim1=0, dim2=1)

        # TODO(tofix)
        # Create transformed RNN initial states. Normally we do
        # no need to do this. But the RNN symbol expects the state
        # to be time-major shape layout, while the current mxnet
        # IO and high-level training logic assume everything from
        # the data iter have batch_size as the first dimension.
        # So until we have extended our IO and training logic to
        # support this more general case, this dummy axis swap is
        # needed.
        rnn_h_init = mx.sym.SwapAxis(mx.sym.Variable('LSTM_init_h'),
                                     dim1=0, dim2=1)
        rnn_c_init = mx.sym.SwapAxis(mx.sym.Variable('LSTM_init_c'),
                                     dim1=0, dim2=1)

        # The original example, rnn_cell_demo.py, uses default Xavier as initalizer, 
        # which relies on variable name, cannot initialize LSTM_parameters. Thus it was
        # renamed to LSTM_bias, which can be initialized as zero. It weakens the converge
        # speed. Here I use revised initializer.py and introduce class RNNXavier to 
        # initialize LSTM_parameters. I attached the first epoch comparison:
        # The old LSTM_bias case:
# 2017-01-28 01:56:19,164 Epoch[0] Batch [50]    Speed: 1081.03 samples/sec    Train-Perplexity=4506.515945
# 2017-01-28 01:56:25,360 Epoch[0] Batch [100]    Speed: 1033.19 samples/sec    Train-Perplexity=797.195374
# 2017-01-28 01:56:31,394 Epoch[0] Batch [150]    Speed: 1060.61 samples/sec    Train-Perplexity=569.822128
# 2017-01-28 01:56:37,784 Epoch[0] Batch [200]    Speed: 1001.64 samples/sec    Train-Perplexity=469.483883
# 2017-01-28 01:56:43,424 Epoch[0] Batch [250]    Speed: 1134.96 samples/sec    Train-Perplexity=341.282116
# 2017-01-28 01:56:49,097 Epoch[0] Batch [300]    Speed: 1128.55 samples/sec    Train-Perplexity=327.141254
# 2017-01-28 01:56:54,667 Epoch[0] Batch [350]    Speed: 1149.03 samples/sec    Train-Perplexity=321.201624
# 2017-01-28 01:57:00,193 Epoch[0] Batch [400]    Speed: 1158.18 samples/sec    Train-Perplexity=293.513695
# 2017-01-28 01:57:04,366 Epoch[0] Train-Perplexity=335.999086
# 2017-01-28 01:57:04,366 Epoch[0] Time cost=51.624
# 2017-01-28 01:57:07,082 Epoch[0] Validation-Perplexity=276.119522
        # The new LSTM_parameters case (based on RNNXavier and revised initializer.py):          
# 2017-01-28 01:54:08,342 Epoch[0] Batch [50]    Speed: 1272.47 samples/sec    Train-Perplexity=3510.636976
# 2017-01-28 01:54:14,230 Epoch[0] Batch [100]    Speed: 1087.30 samples/sec    Train-Perplexity=849.972727
# 2017-01-28 01:54:20,326 Epoch[0] Batch [150]    Speed: 1049.99 samples/sec    Train-Perplexity=496.757170
# 2017-01-28 01:54:25,690 Epoch[0] Batch [200]    Speed: 1193.19 samples/sec    Train-Perplexity=324.072778
# 2017-01-28 01:54:31,388 Epoch[0] Batch [250]    Speed: 1123.19 samples/sec    Train-Perplexity=278.631529
# 2017-01-28 01:54:37,856 Epoch[0] Batch [300]    Speed: 989.55 samples/sec    Train-Perplexity=271.496667
# 2017-01-28 01:54:43,726 Epoch[0] Batch [350]    Speed: 1090.45 samples/sec    Train-Perplexity=210.853686
# 2017-01-28 01:54:49,338 Epoch[0] Batch [400]    Speed: 1140.62 samples/sec    Train-Perplexity=198.847126
# 2017-01-28 01:54:53,062 Epoch[0] Train-Perplexity=186.354118
# 2017-01-28 01:54:53,063 Epoch[0] Time cost=50.277
# 2017-01-28 01:54:55,770 Epoch[0] Validation-Perplexity=181.052411
        rnn_params = mx.sym.Variable('LSTM_parameters')

        # RNN cell takes input of shape (time, batch, feature)
        rnn = mx.sym.RNN(data=embed_tm, state_size=num_hidden,
                         num_layers=num_lstm_layer, mode='lstm',
                         name='LSTM', 
                         # The following params can be omitted
                         # provided we do not need to apply the
                         # workarounds mentioned above
                         state=rnn_h_init,
                         state_cell=rnn_c_init, 
                         parameters=rnn_params)

        # the RNN cell output is of shape (time, batch, dim)
        # if we need the states and cell states in the last time
        # step (e.g. when building encoder-decoder models), we
        # can set state_outputs=True, and the RNN cell will have
        # extra outputs: rnn['LSTM_output'], rnn['LSTM_state']
        # and for LSTM, also rnn['LSTM_state_cell']

        # now we collapse the time and batch dimension to do the
        # final linear logistic regression prediction
        hidden = mx.sym.Reshape(data=rnn, shape=(-1, num_hidden))
        label_cl = mx.sym.Reshape(data=label_tm, shape=(-1,))

        pred = mx.sym.FullyConnected(data=hidden, num_hidden=len(vocab),
                                     name='pred')
        sm = mx.sym.SoftmaxOutput(data=pred, label=label_cl, name='softmax')

        data_names = ['data', 'LSTM_init_h', 'LSTM_init_c']
        label_names = ['softmax_label']

        return (sm, data_names, label_names)

    if len(buckets) == 1:
        mod = mx.mod.Module(*sym_gen(buckets[0]), context=contexts)
    else:
        mod = mx.mod.BucketingModule(sym_gen, default_bucket_key=data_train.default_bucket_key,
                                     context=contexts)

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    mod.fit(data_train, eval_data=data_val, num_epoch=num_epoch,
            eval_metric=mx.metric.np(Perplexity),
            batch_end_callback=mx.callback.Speedometer(batch_size, 50),
            initializer=RNNXavier(factor_type="in", magnitude=2.34),
            optimizer='sgd',
            optimizer_params={'learning_rate': learning_rate,
                              'momentum': momentum, 'wd': 0.00001})
