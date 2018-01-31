"""Predictor for classification/box prediction.

"""
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn


class ConvPredictor(HybridBlock):
    """Convolutional predictor.
    Convolutional predictor is widely used in object-detection. It can be used
    to predict classification scores (1 channel per class) or box predictor,
    which is usually 4 channels per box.
    The output is of shape (N, num_channel, H, W).

    Parameters
    ----------


    """
    def __init__(self, num_channel, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                 activation=None, use_bias=True, **kwargs):
        super(ConvPredictor, self).__init__(**kwargs)
        with self.name_scope():
            self.predictor = nn.Conv2D(
                num_channel, kernel, strides=stride, padding=pad,
                activation=activation, use_bias=use_bias)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.predictor(x)


class FCPredictor(HybridBlock):
    """Fully connected predictor.
    Fully connected predictor is used to ignore spatial information and will
    output fixed-sized predictions.


    Parameters
    ----------

    """
    def __init__(self, num_output, activation=None, use_bias=True, **kwargs):
        super(FCPredictor, self).__init__(**kwargs)
        with self.name_scope():
            self.predictor = nn.Dense(
                num_output, activation=activation, use_bias=use_bias)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.predictor(x)
