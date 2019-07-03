from __future__ import absolute_import
import numpy as _np
import mxnet as mx
from mxnet import np, npx
from mxnet.gluon import HybridBlock
from mxnet.test_utils import same, assert_almost_equal, rand_shape_nd, \
    rand_ndarray
from mxnet.test_utils import check_numeric_gradient
from common import with_seed
import random


@with_seed()
@npx.use_np_shape
def test_np_bitwise_or():
    @npx.use_np_shape
    class TestBitwiseOr(HybridBlock):
        def __init__(self):
            super(TestBitwiseOr, self).__init__()

        def hybrid_forward(self, F, a, b):
            return F.np.bitwise_or(a, b)

    for hybridize in [True, False]:
        for shape_x, shape_y in [[(1,), (1,)],  # single elements
                                 [(4, 5), (4, 5)],  # normal case
                                 [(3, 2), (3, 2)],  # tall matrices
                                 ((), ()),  # scalar only
                                 [(3, 0, 2), (3, 0, 2)],  # zero-dim
                                 ((3, 4, 5), (4, 5)),
                                 # trailing dim broadcasting
                                 ((3, 4, 5), ()),  # scalar broadcasting
                                 [(4, 3), (4, 1)],  # single broadcasting
                                 ((3, 4, 5), (3, 1, 5))
                                 # single broadcasting in the middle
                                 ]:
            test_bitwise_or = TestBitwiseOr()
            if hybridize:
                test_bitwise_or.hybridize()

            x = rand_ndarray(shape_x, dtype=_np.int32).as_np_ndarray()
            x.attach_grad()
            y = rand_ndarray(shape_y, dtype=_np.int32).as_np_ndarray()
            y.attach_grad()

            np_out = _np.bitwise_or(x.asnumpy(), y.asnumpy())
            with mx.autograd.record():
                mx_out = test_bitwise_or(x, y)
            assert mx_out.shape == np_out.shape
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
            mx_out.backward()

            # Test imperative once again
            mx_out = np.bitwise_or(x, y)
            np_out = _np.bitwise_or(x.asnumpy(), y.asnumpy())
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


if __name__ == '__main__':
    test_np_bitwise_or()
