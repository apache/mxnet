"""Encoder and Decoder functions.
Encoders are used during training, which assign training targets.
Decoders are used during testing/validation, which convert predictions back to
normal boxes, etc.
"""
from __future__ import absolute_import

from mxnet import gluon
from .base import CornerToCenterBox
from .registry import register, alias, create


class BoxEncoder(gluon.Block):
    """A base class for box encoder."""
    def __init__(self):
        super(BoxEncoder, self).__init__()

be_register = registry.get_register_func(BoxEncoder, 'box_encoder')
be_alias = registry.get_alias_func(BoxEncoder, 'box_encoder')
be_create = registry.get_create_func(BoxEncoder, 'box_encoder')


class HybridBoxEncoder(gluon.HybridBlock):
    """A base class for hybrid box encoder."""
    def __init__(self):
        super(HybridBoxEncoder, self).__init__()

hbe_register = registry.get_register_func(HybridBoxEncoder, 'hybrid_box_encoder')
hbe_alias = registry.get_alias_func(HybridBoxEncoder, 'hybrid_box_encoder')
hbe_create = registry.get_create_func(HybridBoxEncoder, 'hybrid_box_encoder')


class BoxDecoder(gluon.Block):
    """A base class for box decoder."""
    def __init__(self):
        super(BoxDecoder, self).__init__()

bd_register = registry.get_register_func(BoxDecoder, 'box_decoder')
bd_alias = registry.get_alias_func(BoxDecoder, 'box_decoder')
bd_create = registry.get_create_func(BoxDecoder, 'box_decoder')


class HybridBoxDecoder(gluon.HybridBlock):
    """A base class for hybrid box decoder."""
    def __init__(self):
        super(HybridBoxDecoder, self).__init__()

hbd_register = registry.get_register_func(HybridBoxDecoder, 'hybrid_box_decoder')
hbd_alias = registry.get_alias_func(HybridBoxDecoder, 'hybrid_box_decoder')
hbd_create = registry.get_create_func(HybridBoxDecoder, 'hybrid_box_decoder')


class ClassEncoder(gluon.Block):
    """A base class for classification encoder."""
    def __init__(self):
        super(ClassEncoder, self).__init__()

ce_register = registry.get_register_func(ClassEncoder, 'class_encoder')
ce_alias = registry.get_alias_func(ClassEncoder, 'class_encoder')
ce_create = registry.get_create_func(ClassEncoder, 'class_encoder')


class HybridClassEncoder(gluon.HybridBlock):
    """A base class for hybrid classification encoder."""
    def __init__(self):
        super(HybridClassEncoder, self).__init__()

hce_register = registry.get_register_func(HybridClassEncoder, 'hybrid_class_encoder')
hce_alias = registry.get_alias_func(HybridClassEncoder, 'hybrid_class_encoder')
hce_create = registry.get_create_func(HybridClassEncoder, 'hybrid_class_encoder')


class ClassDecoder(gluon.Block):
    """A base class for classification decoder."""
    def __init__(self):
        super(ClassDecoder, self).__init__()

cd_register = registry.get_register_func(ClassDecoder, 'class_decoder')
cd_alias = registry.get_alias_func(ClassDecoder, 'class_decoder')
cd_create = registry.get_create_func(ClassDecoder, 'class_decoder')


class HybridClassDecoder(gluon.HybridBlock):
    """A base class for hybrid classification decoder."""
    def __init__(self):
        super(HybridClassDecoder, self).__init__()

hcd_register = registry.get_register_func(HybridClassDecoder, 'hybrid_class_decoder')
hcd_alias = registry.get_alias_func(HybridClassDecoder, 'hybrid_class_decoder')
hcd_create = registry.get_create_func(HybridClassDecoder, 'hybrid_class_decoder')


@register
@alias('rcnn_box_encoder')
class NormalizedBoxCenterEncoder(HybridBoxEncoder):
    """

    """
    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2)):
        super(NormalizedBoxCenterEncoder, self).__init__()
        assert len(stds) == 4, "Box Encoder requires 4 std values."
        self._stds = stds
        with self.name_scope():
            self.corner_to_center = CornerToCenterBox(split=True)

    def hybrid_forward(self, F, samples, matches, anchors, refs, *args, **kwargs):
        g = self.corner_to_center(F.pick(refs, matches, axis=1), axis=2, num_outputs=4)
        a = self.corner_to_center(anchors)
        t0 = (g[0] - a[0]) / a[2] / self._stds[0]
        t1 = (g[1] - a[1]) / a[3] / self._stds[1]
        t2 = F.log(g[2] / a[2]) / self._stds[2]
        t3 = F.log(g[3] / a[3]) / self._stds[3]
        codecs = F.concat(t0, t1, t2, t3, dim=2)
        temp = F.tile(samples, reps=(1, 1, 4)) > 0.5
        targets = F.where(temp, codecs, F.zeros_like(codecs))
        masks = F.where(temp, F.ones_like(temp), F.zeros_like(temp))
        return targets, masks


@register
@alias('rcnn_box_decoder')
class NormalizedBoxCenterDecoder(HybridBoxDecoder):
    """

    """
    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2)):
        super(NormalizedBoxCenterDecoder, self).__init__()
        assert len(stds) == 4, "Box Encoder requires 4 std values."
        self._stds = stds
        with self.name_scope():
            self.corner_to_center = CornerToCenterBox(split=True)

    def hybrid_forward(self, F, x, anchors, *args, **kwargs):
        a = self.corner_to_center(anchors)
        p = F.split(x, axis=2, num_outputs=4)
        ox = p[0] * self._stds[0] * a[2] + a[0]
        oy = p[1] * self._stds[1] * a[3] + a[1]
        ow = F.exp(p[2] * self._stds[2]) * a[2] / 2
        oh = F.exp(p[3] * self._stds[3]) * a[3] / 2
        return F.concat(ox - ow, oy - oh, ox + ow, oy + oh, dim=2)

@register
@alias('plus1_class_encoder')
class MultiClassEncoder(HybridClassEncoder):
    """

    """
    def __init__(self):
        super(MultiClassEncoder, self).__init__()

    def hybrid_forward(self, F, samples, matches, refs, *args, **kwargs):
        targets = F.where(samples > 0.5, F.pick(refs, matches, axis=1) + 1, F.zeros_like(refs))
        return targets

@register
@alias('plus1_class_decoder')
class MultiClassDecoder(HybridClassDecoder):
    """

    """
    def __init__(self, axis=-1):
        super(MultiClassDecoder, self).__init__()
        self._axis = axis

    def hybrid_forward(self, F, x, *args, **kwargs):
        cls_id = F.argmax(x, self._axis) - 1
        return cls_id
