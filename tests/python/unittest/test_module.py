import mxnet as mx

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

if __name__ == '__main__':
    test_module_states()
    test_module_reshape()
    test_save_load()
    test_module_layout()
    test_module_switch_bucket()
