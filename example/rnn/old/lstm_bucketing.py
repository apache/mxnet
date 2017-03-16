# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx

from lstm import lstm_unroll
from bucket_io import BucketSentenceIter, default_build_vocab

def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)

if __name__ == '__main__':
    N = 8
    batch_size = 32*N
    #buckets = [10, 20, 30, 40, 50, 60]
    buckets = [32]
    #buckets = []
    num_hidden = 200
    num_embed = 200
    num_lstm_layer = 2

    num_epoch = 25
    learning_rate = 0.01
    momentum = 0.0

    # dummy data is used to test speed without IO
    dummy_data = False

    contexts = [mx.context.gpu(i) for i in range(N)]

    vocab = default_build_vocab("./data/ptb.train.txt")

    def sym_gen(seq_len):
        return lstm_unroll(num_lstm_layer, seq_len, len(vocab),
                           num_hidden=num_hidden, num_embed=num_embed,
                           num_label=len(vocab))

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    data_train = BucketSentenceIter("./data/ptb.train.txt", vocab,
                                    buckets, batch_size, init_states)
    data_val = BucketSentenceIter("./data/ptb.valid.txt", vocab,
                                  buckets, batch_size, init_states)

    if dummy_data:
        data_train = DummyIter(data_train)
        data_val = DummyIter(data_val)

    if len(buckets) == 1:
        # only 1 bucket, disable bucketing
        symbol = sym_gen(buckets[0])
    else:
        symbol = sym_gen

    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=symbol,
                                 num_epoch=num_epoch,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    model.fit(X=data_train, eval_data=data_val, kvstore='device',
              eval_metric = mx.metric.np(Perplexity),
              batch_end_callback=mx.callback.Speedometer(batch_size, 50),)

