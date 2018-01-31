"""Single-shot Multi-box Detector.
"""
from collections import namedtuple
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon import Block, HybridBlock
from mxnet.gluon.model_zoo import vision
from block.feature import FeatureExpander
from block.anchor import SSDAnchorGenerator
from block.predictor import ConvPredictor

class SSDNet(Block):
    """

    """
    def __init__(self, network, features, num_filters, scale, ratios, base_size,
                 num_classes, strides=None, offsets=None, clip=None,
                 use_1x1_transition=True, use_bn=True, reduce_ratio=1.0,
                 min_depth=128, global_pool=False, pretrained=False,
                 ctx=mx.cpu(), **kwargs):
        super(SSDNet, self).__init__(**kwargs)
        num_layers = len(features) + len(num_filters) + int(global_pool)
        assert len(scale) == 2, "Must specify scale as (min_scale, max_scale)."
        min_scale, max_scale = scale
        sizes = [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
                 for i in range(num_layers)] + [1.0]
        sizes = [x * base_size for x in sizes]
        sizes = list(zip(sizes[:-1], sizes[1:]))
        assert isinstance(ratios, list), "Must provide ratios as list or list of list"
        if not isinstance(ratios[0], (tuple, list)):
            ratios = ratios * num_layers  # propagate to all layers if use same ratio
        assert num_layers == len(sizes) == len(ratios), \
            "Mismatched (number of layers) vs (sizes) vs (ratios): {}, {}, {}".format(
                num_layers, len(sizes), len(ratios))
        assert num_layers > 0, "SSD require at least one layer, suggest multiple."
        self._num_layers = num_layers
        self.num_classes = num_classes + 1
        self.features = FeatureExpander(
            network=network, outputs=features, num_filters=num_filters,
            use_1x1_transition=use_1x1_transition,
            use_bn=use_bn, reduce_ratio=reduce_ratio, min_depth=min_depth,
            global_pool=global_pool, pretrained=pretrained, ctx=ctx)


        with self.name_scope():
            self.class_predictors = nn.HybridSequential()
            self.box_predictors = nn.HybridSequential()
            self.anchor_generators = nn.Sequential()
            for i, s, r in zip(range(num_layers), sizes, ratios):
                self.anchor_generators.add(SSDAnchorGenerator(
                    s, r, im_size=(base_size, base_size), clip=clip))
                num_anchors = self.anchor_generators[-1].num_depth
                self.class_predictors.add(ConvPredictor(num_anchors * self.num_classes))
                self.box_predictors.add(ConvPredictor(num_anchors * 4))

    def forward(self, x, *args):
        features = self.features(x)
        cls_preds = [nd.flatten(nd.transpose(cp(feat), (0, 2, 3, 1)))
            for feat, cp in zip(features, self.class_predictors)]
        box_preds = [nd.flatten(nd.transpose(bp(feat), (0, 2, 3, 1)))
            for feat, bp in zip(features, self.box_predictors)]
        anchors = [nd.reshape(ag(feat), shape=(1, -1))
            for feat, ag in zip(features, self.anchor_generators)]
        # for i in range(len(features)):
        #     print(features[i].shape, cls_preds[i].shape, box_preds[i].shape, anchors[i].shape)
        # concat
        cls_preds = nd.concat(*cls_preds, dim=1).reshape((0, -1, self.num_classes))
        box_preds = nd.concat(*box_preds, dim=1).reshape((0, -1, 4))
        anchors = nd.concat(*anchors, dim=1).reshape((1, -1, 4))
        # sync device since anchors are always generated on cpu currently
        anchors = anchors.as_in_context(cls_preds.context)
        return [cls_preds, box_preds, anchors]


SSDConfig = namedtuple('SSDConfig', 'features num_filters scale ratios')

_factory = {
    'resnet18_v1_512': SSDConfig(
        ['stage3_activation1', 'stage4_activation1'], [512, 512, 256, 256],
        [0.1, 0.95], [[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 5),
    'resnet50_v1_512': SSDConfig(
        ['stage3_activation5', 'stage4_activation2'], [512, 512, 256, 256],
        [0.1, 0.95], [[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 5),
}

def get_ssd(name, base_size, classes, pretrained=0, ctx=mx.cpu(), **kwargs):
    """Get SSD models.

    Parameters
    ----------
    name : str
        Model name
    base_size : int

    """
    key = '{}_{}'.format(name, base_size)
    if not key in _factory:
        raise NotImplementedError("{} not defined in model_zoo".format(key))
    c = _factory[key]
    net = SSDNet(name, c.features, c.num_filters, c.scale, c.ratios, base_size,
                 num_classes=classes, pretrained=pretrained > 0, ctx=ctx, **kwargs)
    if pretrained > 1:
        # load trained ssd model
        raise NotImplementedError("Loading pretrained model for detection is not finished.")
    return net

def ssd_512_resnet18_v1(pretrained=0, classes=20, ctx=mx.cpu(), **kwargs):
    """SSD architecture with ResNet v1 18 layers.

    """
    return get_ssd('resnet18_v1', 512, classes, pretrained, ctx, **kwargs)

def ssd_512_resnet50_v1(pretrained=0, classes=20, ctx=mx.cpu(), **kwargs):
    """SSD architecture with ResNet v1 50 layers.

    """
    return get_ssd('resnet50_v1', 512, classes, pretrained, ctx, **kwargs)
