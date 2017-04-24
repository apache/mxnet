import mxnet as mx


def warpctc_layer(net, label, num_label, seq_len, character_classes_count):
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    fc_seq = []
    for seqidx in range(seq_len):
        hidden = net[seqidx]
        hidden = mx.sym.FullyConnected(data=hidden,
                                       num_hidden=character_classes_count,
                                       weight=cls_weight,
                                       bias=cls_bias)
        fc_seq.append(hidden)
    net = mx.sym.Concat(*fc_seq, dim=0)

    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')

    net = mx.sym.WarpCTC(data=net, label=label, label_length=num_label, input_length=seq_len)

    return net
