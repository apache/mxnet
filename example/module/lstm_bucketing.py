# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rnn")))
import numpy as np
import mxnet as mx

from lstm import lstm_unroll
from bucket_io import BucketSentenceIter, default_build_vocab

import os.path
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rnn', 'data'))

def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)

if __name__ == '__main__':
    batch_size = 32
    buckets = [10, 20, 30, 40, 50, 60]
    #buckets = [32]
    num_hidden = 200
    num_embed = 200
    num_lstm_layer = 2

    #num_epoch = 25
    num_epoch = 2
    learning_rate = 0.01
    momentum = 0.0

    # dummy data is used to test speed without IO
    dummy_data = False

    contexts = [mx.context.gpu(i) for i in range(1)]

    vocab = default_build_vocab(os.path.join(data_dir, "ptb.train.txt"))

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    data_train = BucketSentenceIter(os.path.join(data_dir, "ptb.train.txt"), vocab,
                                    buckets, batch_size, init_states)
    data_val = BucketSentenceIter(os.path.join(data_dir, "ptb.valid.txt"), vocab,
                                  buckets, batch_size, init_states)

    if dummy_data:
        data_train = DummyIter(data_train)
        data_val = DummyIter(data_val)

    state_names = [x[0] for x in init_states]
    def sym_gen(seq_len):
        sym = lstm_unroll(num_lstm_layer, seq_len, len(vocab),
                          num_hidden=num_hidden, num_embed=num_embed,
                          num_label=len(vocab))
        data_names = ['data'] + state_names
        label_names = ['softmax_label']
        return (sym, data_names, label_names)

    if len(buckets) == 1:
        mod = mx.mod.Module(*sym_gen(buckets[0]), context=contexts)
    else:
        mod = mx.mod.BucketingModule(sym_gen, default_bucket_key=data_train.default_bucket_key, context=contexts)

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    mod.fit(data_train, eval_data=data_val, num_epoch=num_epoch,
            eval_metric=mx.metric.np(Perplexity),
            batch_end_callback=mx.callback.Speedometer(batch_size, 50),
            initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
            optimizer='sgd',
            optimizer_params={'learning_rate':0.01, 'momentum': 0.9, 'wd': 0.00001})

    # Now it is very easy to use the bucketing to do scoring or collect prediction outputs
    metric = mx.metric.np(Perplexity)
    mod.score(data_val, metric)
    for name, val in metric.get_name_value():
        logging.info('Validation-%s=%f', name, val)

