import mxnet as mx

from negativesample import NegativeSamplingDataIter
import randomproj 
import crossentropy

def CosineLoss(a, b, label):
    a = mx.symbol.L2Normalization(a)
    b = mx.symbol.L2Normalization(b)
    dot = a * b
    dot = mx.symbol.sum_axis(dot, axis=1)
    dot = mx.symbol.Flatten(dot)
    cosine = 1 - dot
    return mx.symbol.MAERegressionOutput(data=cosine, label=label)

def SparseRandomProjection(indexes, values, input_dim, output_dim, ngram=1):
    return mx.symbol.Custom(indexes=indexes, values=values, vocab_size=input_dim,
                            output_dim=output_dim, op_type='SparseRandomProjection')

def SparseBagOfWordProjection(data, vocab_size, output_dim, ngram=1):
    return mx.symbol.Custom(indexes=data, vocab_size=vocab_size,
                            output_dim=output_dim, op_type='SparseBOWProj')

def CrossEntropyLoss(data, label):
    return mx.symbol.Custom(data=data, label=label,
                            op_type='CrossEntropyLoss')

