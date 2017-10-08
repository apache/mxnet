import mxnet as mx
from weighted_softmax_ce import *


def wide_deep_model(num_linear_features, num_embed_features, num_cont_features, input_dims, hidden_units, positive_cls_weight):
    data = mx.symbol.Variable("data", stype='csr')
    label = mx.symbol.Variable("softmax_label")

    x = mx.symbol.slice_axis(data=data, axis=1, begin=0, end=num_linear_features)
    x = mx.symbol.cast_storage(x, 'csr')
    norm_init = mx.initializer.Normal(sigma=0.01)
    # weight with row_sparse storage type to enable sparse gradient updates
    weight = mx.symbol.Variable("weight", shape=(num_linear_features, 2),
                                init=norm_init, stype='row_sparse')
    bias = mx.symbol.Variable("bias", shape=(2,))
    dot = mx.symbol.sparse.dot(x, weight)
    linear_out = mx.symbol.broadcast_add(dot, bias)

    x = mx.symbol.slice_axis(data=data, axis=1, begin=num_linear_features, end=(num_linear_features+num_embed_features))
    embeds = mx.symbol.split(data=x, num_outputs=num_embed_features, squeeze_axis=1)

    x = mx.symbol.slice_axis(data=data, axis=1, begin=(num_linear_features+num_embed_features),
                             end=(num_linear_features+num_embed_features+num_cont_features))
    features = [x]

    for i, embed in enumerate(embeds):
        features.append(mx.symbol.Embedding(data=embed, input_dim=input_dims[i], output_dim=hidden_units[0]))

    hidden = mx.symbol.concat(*features, dim=1)
    hidden = mx.symbol.BatchNorm(data=hidden)
    hidden = mx.symbol.FullyConnected(data=hidden, num_hidden=hidden_units[1])
    hideen = mx.symbol.Activation(data=hidden, act_type='relu')
    hidden = mx.symbol.FullyConnected(data=hidden, num_hidden=hidden_units[2])
    hideen = mx.symbol.Activation(data=hidden, act_type='relu')
    deep_out = mx.symbol.FullyConnected(data=hidden, num_hidden=2)

    out = mx.symbol.Custom(linear_out+deep_out, label, op_type='weighted_softmax_ce_loss',
                           positive_cls_weight=positive_cls_weight, name='model')

    return mx.symbol.MakeLoss(out)
