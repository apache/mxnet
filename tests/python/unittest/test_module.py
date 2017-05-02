import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from functools import reduce

def test_module_dtype():
    dtype = np.float16
    dshape = (3, 8, 7)

    sym = mx.sym.Variable('data')
    sym = mx.sym.Activation(data=sym, act_type='relu', __layout__='TNC')

    mod = mx.mod.Module(sym, ('data',), None, context=[mx.cpu(0), mx.cpu(1)])
    mod.bind(data_shapes=[mx.io.DataDesc('data', dshape, dtype, layout='TNC')])
    mod.init_params()
    mod.forward(mx.io.DataBatch(data=[mx.nd.ones(dshape, dtype=dtype)],
                              label=None))
    mod.backward([mx.nd.ones(dshape, dtype=dtype)])

    for x in mod.get_outputs():
      assert x.dtype == dtype


def test_module_input_grads():
    a = mx.sym.Variable('a', __layout__='NC')
    b = mx.sym.Variable('b', __layout__='NC')
    c = mx.sym.Variable('c', __layout__='NC')

    c = a + 2 * b + 3 * c
    net = mx.mod.Module(c, data_names=['b', 'c', 'a'], label_names=None,
                        context=[mx.cpu(0), mx.cpu(1)])
    net.bind(data_shapes=[['b', (5, 5)], ['c', (5, 5)], ['a', (5, 5)]],
             label_shapes=None, inputs_need_grad=True)
    net.init_params()

    net.forward(data_batch=mx.io.DataBatch(data=[nd.ones((5, 5)),
                                                 nd.ones((5, 5)),
                                                 nd.ones((5, 5))]))
    net.backward(out_grads=[nd.ones((5, 5))])
    input_grads = net.get_input_grads()
    b_grad = input_grads[0].asnumpy()
    c_grad = input_grads[1].asnumpy()
    a_grad = input_grads[2].asnumpy()
    assert np.all(a_grad == 1), a_grad
    assert np.all(b_grad == 2), b_grad
    assert np.all(c_grad == 3), c_grad

def test_module_layout():
    sym = mx.sym.Variable('data')
    sym = mx.sym.Activation(data=sym, act_type='relu', __layout__='TNC')

    dshape = (3, 8, 7)
    mod = mx.mod.Module(sym, ('data',), None, context=[mx.cpu(0), mx.cpu(1)])
    mod.bind(data_shapes=[mx.io.DataDesc('data', dshape, layout='TNC')])
    mod.init_params()
    mod.forward(mx.io.DataBatch(data=[mx.nd.ones(dshape)],
                                label=None))
    mod.backward([mx.nd.ones(dshape)])
    assert mod.get_outputs()[0].shape == dshape

    hdshape = (3, 4, 7)
    for x in mod.get_outputs(merge_multi_context=False)[0]:
        assert x.shape == hdshape

def test_save_load():
    def dict_equ(a, b):
        assert set(a) == set(b)
        for k in a:
            assert (a[k].asnumpy() == b[k].asnumpy()).all()

    sym = mx.sym.Variable('data')
    sym = mx.sym.FullyConnected(sym, num_hidden=100)

    # single device
    mod = mx.mod.Module(sym, ('data',))
    mod.bind(data_shapes=[('data', (10, 10))])
    mod.init_params()
    mod.init_optimizer(optimizer_params={'learning_rate':0.1, 'momentum':0.9})
    mod.update()
    mod.save_checkpoint('test', 0, save_optimizer_states=True)

    mod2 = mx.mod.Module.load('test', 0, load_optimizer_states=True, data_names=('data',))
    mod2.bind(data_shapes=[('data', (10, 10))])
    mod2.init_optimizer(optimizer_params={'learning_rate':0.1, 'momentum':0.9})
    assert mod._symbol.tojson() == mod2._symbol.tojson()
    dict_equ(mod.get_params()[0], mod2.get_params()[0])
    dict_equ(mod._updater.states, mod2._updater.states)

    # multi device
    mod = mx.mod.Module(sym, ('data',), context=[mx.cpu(0), mx.cpu(1)])
    mod.bind(data_shapes=[('data', (10, 10))])
    mod.init_params()
    mod.init_optimizer(optimizer_params={'learning_rate':0.1, 'momentum':0.9})
    mod.update()
    mod.save_checkpoint('test', 0, save_optimizer_states=True)

    mod2 = mx.mod.Module.load('test', 0, load_optimizer_states=True, data_names=('data',))
    mod2.bind(data_shapes=[('data', (10, 10))])
    mod2.init_optimizer(optimizer_params={'learning_rate':0.1, 'momentum':0.9})
    assert mod._symbol.tojson() == mod2._symbol.tojson()
    dict_equ(mod.get_params()[0], mod2.get_params()[0])
    dict_equ(mod._kvstore._updater.states, mod2._updater.states)

def test_module_reshape():
    data = mx.sym.Variable('data')
    sym = mx.sym.FullyConnected(data, num_hidden=20, name='fc')

    dshape = (7, 20)
    mod = mx.mod.Module(sym, ('data',), None, context=[mx.cpu(0), mx.cpu(1)])
    mod.bind(data_shapes=[('data', dshape)])
    mod.init_params()
    mod.init_optimizer(optimizer_params={'learning_rate': 1})

    mod.forward(mx.io.DataBatch(data=[mx.nd.ones(dshape)],
                                label=None))
    mod.backward([mx.nd.ones(dshape)])
    mod.update()
    assert mod.get_outputs()[0].shape == dshape
    assert (mod.get_params()[0]['fc_bias'].asnumpy() == -1).all()

    dshape = (14, 20)
    mod.reshape(data_shapes=[('data', dshape)])
    mod.forward(mx.io.DataBatch(data=[mx.nd.ones(dshape)],
                                label=None))
    mod.backward([mx.nd.ones(dshape)])
    mod.update()
    assert mod.get_outputs()[0].shape == dshape
    assert (mod.get_params()[0]['fc_bias'].asnumpy() == -3).all()

def test_module_states():
    stack = mx.rnn.SequentialRNNCell()
    for i in range(2):
        stack.add(mx.rnn.LSTMCell(num_hidden=20, prefix='lstm_l%d_'%i))
    begin_state = stack.begin_state(func=mx.sym.Variable)
    _, states = stack.unroll(10, begin_state=begin_state, inputs=mx.sym.Variable('data'))

    state_names = [i.name for i in begin_state]
    mod = mx.mod.Module(mx.sym.Group(states), context=[mx.cpu(0), mx.cpu(1)],
                        label_names=None, state_names=state_names)
    mod.bind(data_shapes=[('data', (5, 10))], label_shapes=None, for_training=False)
    mod.init_params()
    batch = mx.io.DataBatch(data=[mx.nd.zeros((5, 10))], label=[])

    mod.set_states(value=1)
    mod.forward(batch)
    out = mod.get_outputs(merge_multi_context=False)
    out1 = mod.get_outputs(merge_multi_context=True)

    mod.set_states(states=out)
    mod.forward(batch)
    out2 = mod.get_outputs(merge_multi_context=True)

    for x1, x2 in zip(out1, out2):
        assert not mx.test_utils.almost_equal(x1.asnumpy(), x2.asnumpy(), rtol=1e-3)

def test_module_switch_bucket():
    vocab_dim = 5000
    num_hidden = 100
    num_embedding = 100
    num_layer = 2
    default_key = 10
    test_key = 5
    batch_size = 32
    contexts = [mx.cpu(0)]
    initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)

    #generate symbols for an LSTM network
    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        embed = mx.sym.Embedding(data=data, input_dim=vocab_dim,
                                 output_dim=num_embedding, name='embed')
        stack = mx.rnn.SequentialRNNCell()
        for i in range(num_layer):
            stack.add(mx.rnn.LSTMCell(num_hidden=num_hidden, prefix='lstm_l%d_'%i))
        outputs, states = stack.unroll(seq_len, inputs=embed, merge_outputs=True)

        pred = mx.sym.Reshape(outputs, shape=(-1, num_hidden))
        pred = mx.sym.FullyConnected(data=pred, num_hidden=vocab_dim, name='pred')

        label = mx.sym.Reshape(label, shape=(-1,))
        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('softmax_label',)

    def create_bucketing_module(key):
        model = mx.mod.BucketingModule(
            sym_gen             = sym_gen,
            default_bucket_key  = key,
            context             = contexts)
        model.bind([('data', (batch_size, key))],
                    [('softmax_label', (batch_size, key))], True, False)
        model.init_params(initializer=initializer)
        return model
    #initialize the bucketing module with the default bucket key
    bucketing_model = create_bucketing_module(default_key)
    #switch to test_key
    bucketing_model.switch_bucket(test_key, [('data', (batch_size, test_key))],
                                  [('softmax_label', (batch_size, test_key))])
    total_bytes_before = bucketing_model._buckets[default_key]._total_exec_bytes

    #remove test_key and switch again
    del bucketing_model._buckets[test_key]
    bucketing_model.switch_bucket(test_key, [('data', (batch_size, test_key))],
                                  [('softmax_label', (batch_size, test_key))])
    total_bytes_after = bucketing_model._buckets[default_key]._total_exec_bytes
    #the default bucket is expected to reuse the bytes allocated
    assert total_bytes_after == total_bytes_before

def test_monitor():
    # data iter
    mx.random.seed(11)
    data = mx.nd.array([[0.05, .10]]);
    label = mx.nd.array([[.01, 0.99]]);
    train_data = mx.io.NDArrayIter(data, label, batch_size=1)

    # symbols
    x = mx.symbol.Variable('data')
    x = mx.symbol.FullyConnected(name='fc_0', data=x, num_hidden=2)
    x = mx.symbol.Activation(name="act_0", data=x, act_type='sigmoid')
    x = mx.symbol.FullyConnected(name='fc_1', data=x, num_hidden=2)
    x = mx.symbol.Activation(name="act_1", data=x, act_type='sigmoid')
    x = mx.symbol.LinearRegressionOutput(data=x, name='softmax', grad_scale=2)

    # create monitor
    def mean_abs(x):
        sum_abs = mx.ndarray.sum(mx.ndarray.abs(x))
        return mx.ndarray.divide(sum_abs, reduce(lambda x, y: x * y, x.shape))
    mon = mx.mon.Monitor(1, stat_func=mean_abs, pattern='.*', sort=True)

    # create module
    mod = mx.mod.Module(x, context=[mx.cpu()]);
    mod.bind(train_data.provide_data, label_shapes=train_data.provide_label,
                    for_training=True)
    mod.install_monitor(mon)
    arg_params = {'fc_0_weight': mx.nd.array([[.15, .20], [.25, .30]]),
                  'fc_0_bias'  : mx.nd.array([.35, .35]),
                  'fc_1_weight': mx.nd.array([[.40, .45], [.50, .55]]),
                  'fc_1_bias'  : mx.nd.array([.60, .60])}
    mod.init_params(arg_params=arg_params)

    data_iter = iter(train_data)
    data_batch = next(data_iter)
    mon.tic()
    mod.forward_backward(data_batch)
    res = mon.toc()
    keys = ['act_0', 'act_1', 'data', 'fc_0', 'fc_1', 'softmax']
    mon_result_counts = [0, 0, 0, 0, 0, 0]
    assert(len(res) == 21)
    for n, k, v in res:
        for idx, key in enumerate(keys):
            if k.startswith(key):
                mon_result_counts[idx] += 1
                break
    assert(mon_result_counts == [2, 2, 1, 6, 6, 4])

if __name__ == '__main__':
    test_module_dtype()
    test_module_input_grads()
    test_module_states()
    test_module_reshape()
    test_save_load()
    test_module_layout()
    test_module_switch_bucket()
    test_monitor()
