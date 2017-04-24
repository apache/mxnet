import mxnet as mx


def slice_symbol_to_seq_symobls(net, seq_len, axis=1, squeeze_axis=True):
    net = mx.sym.SliceChannel(data=net, num_outputs=seq_len, axis=axis, squeeze_axis=squeeze_axis)
    hidden_all = []
    for seq_index in range(seq_len):
        hidden_all.append(net[seq_index])
    net = hidden_all
    return net
