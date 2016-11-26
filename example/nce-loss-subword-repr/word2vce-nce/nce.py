# pylint:skip-file
import sys
import mxnet as mx

def nce_loss(data, label, label_weight, embed_weight, vocab_size, num_hidden, num_label):
    label_embed = mx.sym.Embedding(data = label, input_dim = vocab_size,
                                   weight = embed_weight,
                                   output_dim = num_hidden, name = 'label_embed')
    label_embed = mx.sym.SliceChannel(data = label_embed,
                                      num_outputs = num_label,
                                      squeeze_axis = 1, name = 'label_slice')
    label_weight = mx.sym.SliceChannel(data = label_weight,
                                       num_outputs = num_label,
                                       squeeze_axis = 1)
    probs = []
    for i in range(num_label):
        vec = label_embed[i]
        vec = vec * data
        vec = mx.sym.sum(vec, axis = 1)
        sm = mx.sym.LogisticRegressionOutput(data = vec,
                                             label = label_weight[i])
        probs.append(sm)
    return probs

