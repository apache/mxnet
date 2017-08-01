# coding: utf-8
# pylint: disable= arguments-differ
"""Alexnet, implemented in Gluon."""
__all__ = ['AlexNet', 'alexnet']

from ....context import cpu
from ...block import HybridBlock
from ... import nn

# Net
class AlexNet(HybridBlock):
    r"""AlexNet model from the `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Parameters
    ----------
    classes : int, default 1000
        Number of classes for the output layer.
    """
    def __init__(self, classes=1000, **kwargs):
        super(AlexNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            with self.features.name_scope():
                self.features.add(nn.Conv2D(64, kernel_size=11, strides=4,
                                            padding=2, activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(192, kernel_size=5, padding=2,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(384, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Flatten())

            self.classifier = nn.HybridSequential(prefix='')
            with self.classifier.name_scope():
                self.classifier.add(nn.Dense(4096, activation='relu'))
                self.classifier.add(nn.Dropout(0.5))
                self.classifier.add(nn.Dense(4096, activation='relu'))
                self.classifier.add(nn.Dropout(0.5))
                self.classifier.add(nn.Dense(classes))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Constructor
def alexnet(pretrained=False, ctx=cpu(), **kwargs):
    r"""AlexNet model from the `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    net = AlexNet(**kwargs)
    if pretrained:
        from ..model_store import get_model_file
        net.load_params(get_model_file('alexnet'), ctx=ctx)
    return net
