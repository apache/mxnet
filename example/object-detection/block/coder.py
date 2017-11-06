"""Encoder and Decoder functions.
Encoders are used during training, which assign training targets.
Decoders are used during testing/validation, which convert predictions back to
normal boxes, etc.
"""
from mxnet import nd
from mxnet import gluon
from block.base import CornerToCenterBox
from block.registry import register, alias, create


class BoxEncoder(gluon.Block):
    """A base class for box encoder."""
    def __init__(self):
        super(BoxEncoder, self).__init__()


class HybridBoxEncoder(gluon.HybridBlock):
    """A base class for hybrid box encoder."""
    def __init__(self):
        super(HybridBoxEncoder, self).__init__()


class BoxDecoder(gluon.Block):
    """A base class for box decoder."""
    def __init__(self):
        super(BoxDecoder, self).__init__()


class HybridBoxDecoder(gluon.HybridBlock):
    """A base class for hybrid box decoder."""
    def __init__(self):
        super(HybridBoxDecoder, self).__init__()


class ClassEncoder(gluon.Block):
    """A base class for classification encoder."""
    def __init__(self):
        super(ClassEncoder, self).__init__()


class HybridClassEncoder(gluon.HybridBlock):
    """A base class for hybrid classification encoder."""
    def __init__(self):
        super(HybridClassEncoder, self).__init__()


class ClassDecoder(gluon.Block):
    """A base class for classification decoder."""
    def __init__(self):
        super(ClassDecoder, self).__init__()


class HybridClassDecoder(gluon.HybridBlock):
    """A base class for hybrid classification decoder."""
    def __init__(self):
        super(HybridClassDecoder, self).__init__()


@register
@alias('rcnn_box_encoder')
class NormalizedBoxCenterEncoder(BoxEncoder):
    """

    """
    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2)):
        super(NormalizedBoxCenterEncoder, self).__init__()
        assert len(stds) == 4, "Box Encoder requires 4 std values."
        self._stds = stds
        with self.name_scope():
            self.corner_to_center = CornerToCenterBox(split=True)

    def forward(self, samples, matches, anchors, refs, *args, **kwargs):
        F = nd
        # TODO(zhreshold): batch_pick, take multiple elements?
        ref_boxes = nd.repeat(refs.reshape((0, 1, -1, 4)), axis=1, repeats=matches.shape[1])
        ref_boxes = nd.split(ref_boxes, axis=-1, num_outputs=4, squeeze_axis=True)
        ref_boxes = nd.concat(*[F.pick(ref_boxes[i], matches, axis=2).reshape((0, -1, 1)) for i in range(4)], dim=2)
        g = self.corner_to_center(ref_boxes)
        a = self.corner_to_center(anchors)
        t0 = (g[0] - a[0]) / a[2] / self._stds[0]
        t1 = (g[1] - a[1]) / a[3] / self._stds[1]
        t2 = F.log(g[2] / a[2]) / self._stds[2]
        t3 = F.log(g[3] / a[3]) / self._stds[3]
        codecs = F.concat(t0, t1, t2, t3, dim=2)
        temp = F.tile(samples.reshape((0, -1, 1)), reps=(1, 1, 4)) > 0.5
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
class MultiClassEncoder(ClassEncoder):
    """

    """
    def __init__(self, ignore_label=-1):
        super(MultiClassEncoder, self).__init__()
        self._ignore_label = ignore_label

    def forward(self, samples, matches, refs, *args, **kwargs):
        refs = nd.repeat(refs.reshape((0, 1, -1)), axis=1, repeats=matches.shape[1])
        target_ids = nd.pick(refs, matches, axis=2) + 1
        targets = nd.where(samples > 0.5, target_ids, nd.ones_like(target_ids) * self._ignore_label)
        targets = nd.where(samples < -0.5, nd.zeros_like(targets), targets)
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
        pos_x = x.slice_axis(axis=self._axis, begin=1, end=-1)
        cls_id = F.argmax(pos_x, self._axis)
        scores = F.pick(pos_x, cls_id, axis=-1)
        return cls_id, scores
