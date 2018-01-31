"""Feature extraction blocks.
Feature or Multi-Feature extraction is a key component in object detection.
Class predictor/Box predictor are usually applied on feature layer(s).
A good feature extraction mechanism is critical to performance.
"""
import mxnet as mx
from mxnet.symbol import Symbol
from mxnet.gluon import HybridBlock, SymbolBlock
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision
from mxnet.base import string_types

def parse_network(network, outputs, inputs, pretrained, ctx):
    """Parse network with specified outputs and other arguments.

    Parameters
    ----------
    network : str or HybridBlock or Symbol
        Logic chain: load from gluon.model_zoo.vision if network is string.
        Convert to Symbol if network is HybridBlock
    outputs : str or list of str
        The name of layers to be extracted as features
    pretrained : bool
        Use pretrained parameters as in gluon.model_zoo
    ctx : Context
        The context

    Returns
    -------
    inputs : list of Symbol
        Network input Symbols, usually ['data']
    outputs : list of Symbol
        Network output Symbols, usually as features
    params : ParameterDict
        Network parameters.
    """
    for i in range(len(inputs)):
        if isinstance(inputs[i], string_types):
            inputs[i] = mx.sym.var(inputs[i])
        assert isinstance(inputs[i], Symbol), "Network expects inputs are Symbols."
    if len(inputs) == 1:
        inputs = inputs[0]
    else:
        inputs = mx.sym.Group(inputs)
    params = None
    if isinstance(network, string_types):
        network = vision.get_model(network, pretrained=pretrained, ctx=ctx, prefix='')
    if isinstance(network, HybridBlock):
        params = network.collect_params()
        network = network(inputs)
    assert isinstance(network, Symbol), \
        "FeatureExtractor requires the network argument to be either " \
        "str, HybridBlock or Symbol, but got %s"%type(network)

    if isinstance(outputs, string_types):
        outputs = [outputs]
    assert len(outputs) > 0, "At least one outputs must be specified."
    outputs = [out if out.endswith('_output') else out + '_output' for out in outputs]
    outputs = [network.get_internals()[out] for out in outputs]
    return inputs, outputs, params


class FeatureExtractor(SymbolBlock):
    """Feature extractor.

    Parameters
    ----------
    network : str or HybridBlock or Symbol
        Logic chain: load from gluon.model_zoo.vision if network is string.
        Convert to Symbol if network is HybridBlock
    outputs : str or list of str
        The name of layers to be extracted as features
    inputs : list of str or list of Symbol
        The inputs of network.
    pretrained : bool
        Use pretrained parameters as in gluon.model_zoo
    ctx : Context
        The context
    """
    def __init__(self, network, outputs, inputs=['data'], pretrained=False, ctx=mx.cpu()):
        inputs, outputs, params = parse_network(network, outputs, inputs, pretrained, ctx)
        super(FeatureExtractor, self).__init__(outputs, inputs, params=params)


class FeatureExpander(SymbolBlock):
    """Feature extractor with additional layers to append.
    This is very common in SSD networks.

    """
    def __init__(self, network, outputs, num_filters, use_1x1_transition=True,
                 use_bn=True, reduce_ratio=1.0, min_depth=128, global_pool=False,
                 pretrained=False, ctx=mx.cpu(), inputs=['data']):
        inputs, outputs, params = parse_network(network, outputs, inputs, pretrained, ctx)
        # append more layers
        y = outputs[-1]
        for i, f in enumerate(num_filters):
            if use_1x1_transition:
                num_trans = max(min_depth, int(round(f * reduce_ratio)))
                y = mx.sym.Convolution(
                    y, num_filter=num_trans, kernel=(1, 1), no_bias=use_bn,
                    name='expand_trans_conv{}'.format(i))
                if use_bn:
                    y = mx.sym.BatchNorm(y, name='expand_trans_bn{}'.format(i))
                y = mx.sym.Activation(y, act_type='relu', name='expand_trans_relu{}'.format(i))
            y = mx.sym.Convolution(
                y, num_filter=f, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
                name='expand_conv{}'.format(i))
            if use_bn:
                y = mx.sym.BatchNorm(y, name='expand_bn{}'.format(i))
            y = mx.sym.Activation(y, act_type='relu', name='expand_reu{}'.format(i))
            outputs.append(y)
        if global_pool:
            outputs.append(mx.sym.Pooling(y, pool_type='avg', global_pool=True, kernel=(1, 1)))
        super(FeatureExpander, self).__init__(outputs, inputs, params)
