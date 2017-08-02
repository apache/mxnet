"""Anchor generators.
The job of the anchor generator is to create (or load) a collection
of bounding boxes to be used as anchors.
Generated anchors are assumed to match some convolutional grid or list of grid
shapes.  For example, we might want to generate anchors matching an 8x8
feature map and a 4x4 feature map.  If we place 3 anchors per grid location
on the first feature map and 6 anchors per grid location on the second feature
map, then 3*8*8 + 6*4*4 = 288 anchors are generated in total.
To support fully convolutional settings, feature maps are passed as input,
however, only shapes are used to infer the anchors.
"""
from __future__ import division
import math
from mxnet import gluon
from mxnet import ndarray as nd
from .registry import register, alias, create


class ShapeExtractor(gluon.Block):
    """

    """
    def __init__(self, positions=[0]):
        super(ShapeExtractor, self).__init__()
        if not isinstance(positions, (list, tuple)):
            raise ValueError("positions must be list or tuple")
        self._positions = positions

    def forward(self, x, *args):
        if x is None:
            return x
        xshape = x.shape
        return nd.array([xshape[i] for i in self._positions])

@register
class GridAnchorGenerator(gluon.Block):
    """

    """
    def __init__(self, size_ratios, strides=None, offsets=None, clip=None,
                 im_size=(256.0, 256.0), layout='HWC'):
        super(GridAnchorGenerator, self).__init__()
        assert (isinstance(size_ratios, list) and size_ratios), (
            "Invalid size_ratios list.")
        for sr in size_ratios:
            assert (isinstance(sr, (list, tuple)) and len(sr) == 2), (
                "Each size_ratio pair must be length-2 tuple/list.")
        self._size_ratios = size_ratios
        if strides is not None:
            assert len(strides) == 2, "strides must be either None or length-2 vector"
        self._strides = strides
        if offsets is not None:
            assert len(offsets) == 2, "offsets must be either None or length-2 vector"
        self._offsets = offsets
        if clip is not None:
            assert len(clip) == 4, "clip must be either None or length-4 vector"
        self._clip = clip
        assert len(im_size) == 2, "im_size must be (height, width)"
        self._im_size = im_size
        assert layout == 'HWC' or layout == 'CHW', "layout must be 'HWC' or 'CHW'"
        self._layout = layout
        with self.name_scope():
            self.feat_size_extractor = ShapeExtractor([2, 3])
            self.im_size_extractor = ShapeExtractor([2, 3])

    @property
    def num_depth(self):
        """Returns the number of anchors per pixel/grid/location.
        """
        return len(self._size_ratios)

    def forward(self, x, img=None, *args):
        # input image size
        im_height, im_width = self._get_im_size(self.im_size_extractor(img))
        # feature size
        feat_height, feat_width = self._get_feat_size(self.feat_size_extractor(x))
        # stride
        stride_h, stride_w = self._get_strides(feat_height, feat_width, im_height, im_width)
        # offsets for center
        offset_h, offset_w = self._get_offsets(feat_height, feat_width, im_height, im_width)
        # generate anchors for each pixel/grid, as layout [HxWxC, 4]
        centers = [[(i * stride_w + offset_w) / im_width, (j * stride_h + offset_h) / im_height]
            for j in range(feat_height) for i in range(feat_width) for _ in self._size_ratios]
        shapes = [[s * math.sqrt(r) / im_width, s / math.sqrt(r) / im_height]
            for _ in range(feat_height) for _ in range(feat_width) for s, r in self._size_ratios]
        # convert to ndarray and as corner [xmin, ymin, xmax, ymax]
        shapes = nd.array(shapes) * 0.5
        centers = nd.array(centers)
        anchors = nd.concat(centers - shapes, centers + shapes, dim=1)

        if self._clip is not None:
            self._clip_anchors(anchors, self._clip)
        # print(anchors.shape)
        # anchors = anchors.reshape((feat_height, feat_width, self.num_depth, 4))
        # if self._layout == 'CHW':
        #     anchors = nd.transpose(anchors, (2, 0, 1, 3))

        return anchors

    def _get_im_size(self, im_size):
        """Get original image size given ndarray shape data."""
        im_height, im_width = self._im_size
        if im_size is not None:
            # infer image shape from data is available
            assert im_size.size == 2, (
                "Invalid data shape {}, expected (h, w)".format(im_size.shape))
            im_height, im_width = im_size.asnumpy().astype('int')
        return im_height, im_width

    def _get_feat_size(self, feat_size):
        """Get feature map size given ndarray shape data."""
        assert feat_size.size == 2, (
            "Invalid feat shape {}, expected (h, w)".format(feat_size.shape))
        feat_height, feat_width = feat_size.asnumpy().astype('int')
        return feat_height, feat_width

    def _get_strides(self, feat_height, feat_width, im_height, im_width):
        """Wrapping function for default grid strides."""
        if self._strides is None:
            stride_h = im_height / feat_height
            stride_w = im_width / feat_width
        else:
            stride_h, stride_w = self._strides
        return stride_h, stride_w

    def _get_offsets(self, feat_height, feat_width, im_height, im_width):
        """Wrapping function for default grid offsets."""
        if self._offsets is None:
            offset_h = 0.5 * im_height / feat_height
            offset_w = 0.5 * im_width / feat_width
        else:
            offset_h , offset_w = self._offsets
        return offset_h, offset_w

    def _clip_anchors(self, anchors, clip_window):
        """Clip all anchors to clip_window area.

        Parameters
        ----------
        anchors : NDArray
            N x 4 array
        clip_window : list or tuple
            [xmin, ymin, xmax, ymax] window

        Returns
        -------
        a NDArray with clipped anchor boxes
        """
        l, t, r, b = nd.split(anchors, axis=1, num_outputs=4)
        l = nd.maximum(clip_window[0], nd.minimum(clip_window[2], l))
        t = nd.maximum(clip_window[1], nd.minimum(clip_window[3], t))
        r = nd.maximum(clip_window[0], nd.minimum(clip_window[2], r))
        b = nd.maximum(clip_window[1], nd.minimum(clip_window[3], b))
        return nd.concat(l, t, r, b, dim=1)


@register
class SSDAnchorGenerator(GridAnchorGenerator):
    """

    """
    def __init__(self, sizes, ratios, strides=None, offsets=None, clip=None,
                 im_size=(300.0, 300.0), layout='HWC'):
        assert len(sizes) > 0
        assert len(ratios) > 0
        size_ratios = [(s, ratios[0]) for s in sizes] + [(sizes[0], r) for r in ratios[1:]]
        super(SSDAnchorGenerator, self).__init__(size_ratios, strides=strides,
                                                 offsets=offsets, clip=clip,
                                                 im_size=im_size, layout=layout)
