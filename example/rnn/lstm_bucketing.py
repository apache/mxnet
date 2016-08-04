# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
#sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx

from lstm import lstm_unroll
from bucket_io import BucketSentenceIter, default_build_vocab, DummyIter

def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)

if __name__ == '__main__':
    batch_size = 20
    #buckets = [10, 20, 30, 40, 50, 60]
    #buckets = [32]
    buckets = []
    num_hidden = 200
    num_embed = 200
    num_lstm_layer = 2

    num_epoch = 25
    learning_rate = 0.005
    momentum = 0.9

    # dummy data is used to test speed without IO
    dummy_data = False

    contexts = [mx.context.cpu(i) for i in range(1)]

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

    gates_bn_hidden = 4*num_hidden
    gates_bn_gamma = []
    for layeridx in range(num_lstm_layer):
        gates_bn_gamma.append(mx.nd.ones((gates_bn_hidden), contexts[0])*0.1)

    arg_params={}
    for layeridx in range(num_lstm_layer):
        arg_params["lstm%d_gates_bn_gamma" % layeridx] = gates_bn_gamma[layeridx]

    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=symbol,
                                 num_epoch=num_epoch,
                                 optimizer='adagrad',
                                 learning_rate=learning_rate,
                                 wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
                                 arg_params=arg_params,
                                 allow_extra_params=False)
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    from mxnet.executor import Executor
    def batch_end_callback(batch_size,frequent):
            call_back = mx.callback.Speedometer(batch_size, frequent)
            def AverageL2Norm(d):
                return (mx.nd.norm(d)/np.sqrt(d.size)).asnumpy()[0]
            def decorator(parameter):
                call_back(parameter)
                if parameter.locals['nbatch'] % frequent == 0:
                    executor_manager = parameter.locals['executor_manager']
                    d= Executor._get_dict(executor_manager.aux_names, executor_manager.aux_arrays)
                    for key,value in sorted(list(zip(d.keys(),d.values()))):
                        print key,'AverageL2Norm(aux):',AverageL2Norm(value[0])
                    for (index,value) in enumerate(executor_manager.param_names):
                        print value,'AverageL2Norm(param,grad):',(AverageL2Norm(executor_manager.param_arrays[index][0]),
                                                                  AverageL2Norm(executor_manager.grad_arrays[index][0]))
                return False #Not to stop training
            return decorator

    model.fit(X=data_train, eval_data=data_val,
              eval_metric = mx.metric.np(Perplexity),
              batch_end_callback=batch_end_callback(batch_size, 2))

