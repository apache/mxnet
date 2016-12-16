import mxnet as mx

def test_module_layout():
    sym = mx.sym.Variable('data')
    sym = mx.sym.Activation(data=sym, act_type='relu', __layout__='TNC')

    dshape = (3L, 8L, 7L)
    mod = mx.mod.Module(sym, ('data',), None, context=[mx.cpu(0), mx.cpu(1)])
    mod.bind(data_shapes=[mx.io.DataDesc('data', dshape, layout='TNC')])
    mod.init_params()
    mod.forward(mx.io.DataBatch(data=[mx.nd.ones(dshape)],
                                label=None))
    mod.backward([mx.nd.ones(dshape)])
    assert mod.get_outputs()[0].shape == dshape

    hdshape = (3L, 4L, 7L)
    for x in mod.get_outputs(merge_multi_context=False)[0]:
        assert x.shape == hdshape

if __name__ == '__main__':
    test_module_layout()