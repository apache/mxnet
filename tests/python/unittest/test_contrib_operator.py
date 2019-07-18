# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: skip-file
from __future__ import print_function
import numpy as np
import mxnet as mx
import random
import itertools
from numpy.testing import assert_allclose, assert_array_equal
from mxnet.test_utils import *
import unittest

def test_box_nms_op():
    def test_box_nms_forward(data, expected, thresh=0.5, valid=0, topk=-1, coord=2, score=1, cid=0, bid=-1,
                             force=False, in_format='corner', out_format='corner'):
        for dtype in ['float16', 'float32', 'float64']:
            data = mx.nd.array(data, dtype=dtype)
            out = mx.contrib.nd.box_nms(data, overlap_thresh=thresh, valid_thresh=valid, topk=topk,
                                        coord_start=coord, score_index=score, id_index=cid, background_id=bid,
                                        force_suppress=force, in_format=in_format, out_format=out_format)
            assert_almost_equal(out.asnumpy(), expected.astype(dtype), rtol=1e-3, atol=1e-3)

    def test_box_nms_backward(data, grad, expected, thresh=0.5, valid=0, topk=-1, coord=2, score=1,
                              cid=0, bid=-1, force=False, in_format='corner', out_format='corner'):
        in_var = mx.sym.Variable('data')
        arr_data = mx.nd.array(data)
        arr_grad = mx.nd.empty(arr_data.shape)
        op = mx.contrib.sym.box_nms(in_var, overlap_thresh=thresh, valid_thresh=valid, topk=topk,
                                    coord_start=coord, score_index=score, id_index=cid, background_id=bid,
                                    force_suppress=force, in_format=in_format, out_format=out_format)
        exe = op.bind(ctx=default_context(), args=[arr_data], args_grad=[arr_grad])
        exe.forward(is_train=True)
        exe.backward(mx.nd.array(grad))
        assert_almost_equal(arr_grad.asnumpy(), expected)

    def corner_to_center(data):
        out = np.reshape(data, (-1, 6)).copy()
        out[:, 2] = (data[:, 2] + data[:, 4]) / 2.0
        out[:, 3] = (data[:, 3] + data[:, 5]) / 2.0
        out[:, 4] = data[:, 4] - data[:, 2]
        out[:, 5] = data[:, 5] - data[:, 3]
        invalid = np.where(data[:, 0] < 0)[0]
        out[invalid, :] = -1
        return out

    def center_to_corner(data):
        data = np.reshape(data, (-1, 6)).copy()
        out[:, 2] = data[:, 2] - data[:, 4] / 2.0
        out[:, 3] = data[:, 3] - data[:, 5] / 2.0
        out[:, 4] = data[:, 2] + data[:, 4] / 2.0
        out[:, 5] = data[:, 3] + data[:, 5] / 2.0
        invalid = np.where(data[:, 0] < 0)[0]
        out[invalid, :] = -1
        return out

    def swap_position(data, expected, coord=2, score=1, cid=0, new_col=0):
        data = np.reshape(data, (-1, 6))
        expected = np.reshape(expected, (-1, 6))
        new_coord = random.randint(0, 6 + new_col - 4)
        others = list(range(new_coord)) + list(range(new_coord + 4, 6 + new_col))
        random.shuffle(others)
        new_score = others[0]
        new_cid = others[1]
        new_data = np.full((data.shape[0], data.shape[1] + new_col), -1.0)
        new_expected = np.full((expected.shape[0], expected.shape[1] + new_col), -1.0)
        new_data[:, new_coord:new_coord+4] = data[:, coord:coord+4]
        new_data[:, new_score] = data[:, score]
        new_data[:, new_cid] = data[:, cid]
        new_expected[:, new_coord:new_coord+4] = expected[:, coord:coord+4]
        new_expected[:, new_score] = expected[:, score]
        new_expected[:, new_cid] = expected[:, cid]
        return new_data, new_expected, new_coord, new_score, new_cid

    # manually set up test cases
    boxes = [[0, 0.5, 0.1, 0.1, 0.2, 0.2], [1, 0.4, 0.1, 0.1, 0.2, 0.2],
             [0, 0.3, 0.1, 0.1, 0.14, 0.14], [2, 0.6, 0.5, 0.5, 0.7, 0.8]]

    # case1
    force = True
    thresh = 0.5
    expected = [[2, 0.6, 0.5, 0.5, 0.7, 0.8], [0, 0.5, 0.1, 0.1, 0.2, 0.2],
                [0, 0.3, 0.1, 0.1, 0.14, 0.14], [-1, -1, -1, -1, -1, -1]]
    grad = np.random.rand(4, 6)
    expected_in_grad = grad[(1, 3, 2, 0), :]
    expected_in_grad[1, :] = 0
    test_box_nms_forward(np.array(boxes), np.array(expected), force=force, thresh=thresh)
    test_box_nms_backward(np.array(boxes), grad, expected_in_grad, force=force, thresh=thresh)

    # case2: multi batch
    boxes2 = [boxes] * 3
    expected2 = [expected] * 3
    grad2 = np.array([grad.tolist()] * 3)
    expected_in_grad2 = np.array([expected_in_grad.tolist()] * 3)
    test_box_nms_forward(np.array(boxes2), np.array(expected2), force=force, thresh=thresh)
    test_box_nms_backward(np.array(boxes2), grad2, expected_in_grad2, force=force, thresh=thresh)
    # another new dim
    boxes2 = [boxes2] * 2
    expected2 = [expected2] * 2
    grad2 = np.array([grad2.tolist()] * 2)
    expected_in_grad2 = np.array([expected_in_grad2.tolist()] * 2)
    test_box_nms_forward(np.array(boxes2), np.array(expected2), force=force, thresh=thresh)
    test_box_nms_backward(np.array(boxes2), grad2, expected_in_grad2, force=force, thresh=thresh)

    # case3: thresh
    thresh = 0.1
    boxes3 = boxes
    expected3 = [[2, 0.6, 0.5, 0.5, 0.7, 0.8], [0, 0.5, 0.1, 0.1, 0.2, 0.2],
                [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]]
    grad3 = np.random.rand(4, 6)
    expected_in_grad3 = grad3[(1, 3, 2, 0), :]
    expected_in_grad3[(1, 2), :] = 0
    test_box_nms_forward(np.array(boxes3), np.array(expected3), force=force, thresh=thresh)
    test_box_nms_backward(np.array(boxes3), grad3, expected_in_grad3, force=force, thresh=thresh)

    # case4: non-force
    boxes4 = boxes
    force = False
    expected4 = [[2, 0.6, 0.5, 0.5, 0.7, 0.8], [0, 0.5, 0.1, 0.1, 0.2, 0.2],
                [1, 0.4, 0.1, 0.1, 0.2, 0.2], [-1, -1, -1, -1, -1, -1]]
    grad4 = np.random.rand(4, 6)
    expected_in_grad4 = grad4[(1, 2, 3, 0), :]
    expected_in_grad4[2, :] = 0
    test_box_nms_forward(np.array(boxes4), np.array(expected4), force=force, thresh=thresh)
    test_box_nms_backward(np.array(boxes4), grad4, expected_in_grad4, force=force, thresh=thresh)

    # case5: different coding
    boxes5 = corner_to_center(np.array(boxes4))
    test_box_nms_forward(np.array(boxes5), np.array(expected4), force=force, thresh=thresh,
        in_format='center')
    expected5 = corner_to_center(np.array(expected4))
    test_box_nms_forward(np.array(boxes4), np.array(expected5), force=force, thresh=thresh,
        out_format='center')
    test_box_nms_forward(np.array(boxes5), np.array(expected5), force=force, thresh=thresh,
        in_format='center', out_format='center')

    # case6: different position
    boxes6, expected6, new_coord, new_score, new_id = swap_position(np.array(boxes4),
        np.array(expected4), new_col=2)
    test_box_nms_forward(np.array(boxes6), np.array(expected6), force=force, thresh=thresh,
        coord=new_coord, score=new_score, cid=new_id)

    # case7: no id, should be same with force=True
    force = False
    thresh = 0.5
    test_box_nms_forward(np.array(boxes), np.array(expected), force=force, thresh=thresh, cid=-1)

    # case8: multi-batch thresh + topk
    boxes8 = [[[1, 1, 0, 0, 10, 10], [1, 0.4, 0, 0, 10, 10], [1, 0.3, 0, 0, 10, 10]],
              [[2, 1, 0, 0, 10, 10], [2, 0.4, 0, 0, 10, 10], [2, 0.3, 0, 0, 10, 10]],
              [[3, 1, 0, 0, 10, 10], [3, 0.4, 0, 0, 10, 10], [3, 0.3, 0, 0, 10, 10]]]
    expected8 = [[[1, 1, 0, 0, 10, 10], [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]],
                 [[2, 1, 0, 0, 10, 10], [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]],
                 [[3, 1, 0, 0, 10, 10], [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]]]
    grad8 = np.random.rand(3, 3, 6)
    expected_in_grad8 = np.zeros((3, 3, 6))
    expected_in_grad8[(0, 1, 2), (0, 0, 0), :] = grad8[(0, 1, 2), (0, 0, 0), :]
    force = False
    thresh = 0.5
    valid = 0.5
    topk = 2
    test_box_nms_forward(np.array(boxes8), np.array(expected8), force=force, thresh=thresh, valid=valid, topk=topk)
    test_box_nms_backward(np.array(boxes8), grad8, expected_in_grad8, force=force, thresh=thresh, valid=valid, topk=topk)

    # case9: background id filter out
    # default background id -1
    boxes9 = [[0, 0.5, 0.1, 0.1, 0.2, 0.2], [0, 0.4, 0.1, 0.1, 0.2, 0.2],
              [1, 0.3, 0.1, 0.1, 0.14, 0.14], [-1, 0.6, 0.5, 0.5, 0.7, 0.8]]
    expected9 = [[0, 0.5, 0.1, 0.1, 0.2, 0.2], [1, 0.3, 0.1, 0.1, 0.14, 0.14],
                 [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]]
    force = True
    thresh = 0.5
    grad9 = np.random.rand(4, 6)
    expected_in_grad9 = grad9[(0, 2, 1, 3), :]
    expected_in_grad9[(1, 3), :] = 0
    test_box_nms_forward(np.array(boxes9), np.array(expected9), force=force, thresh=thresh)
    test_box_nms_backward(np.array(boxes9), grad9, expected_in_grad9, force=force, thresh=thresh)
    # set background id
    background_id = 0
    expected9 = [[-1, 0.6, 0.5, 0.5, 0.7, 0.8], [1, 0.3, 0.1, 0.1, 0.14, 0.14],
                 [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]]
    grad9 = np.random.rand(4, 6)
    expected_in_grad9 = grad9[(2, 3, 1, 0), :]
    expected_in_grad9[(0, 1), :] = 0
    test_box_nms_forward(np.array(boxes9), np.array(expected9), force=force, thresh=thresh, bid=background_id)
    test_box_nms_backward(np.array(boxes9), grad9, expected_in_grad9, force=force, thresh=thresh, bid=background_id)

def test_box_iou_op():
    def numpy_box_iou(a, b, fmt='corner'):
        def area(left, top, right, bottom):
            return np.maximum(0, right - left) * np.maximum(0, bottom - top)

        assert a.shape[-1] == 4
        assert b.shape[-1] == 4
        oshape = a.shape[:-1] + b.shape[:-1]
        a = a.reshape((-1, 4))
        ashape = a.shape
        b = b.reshape((-1, 4))
        a = np.tile(a, reps=[1, b.shape[0]]).reshape((-1, 4))
        b = np.tile(b, reps=[ashape[0], 1]).reshape((-1, 4))
        if fmt == 'corner':
            al, at, ar, ab = np.split(a, 4, axis=-1)
            bl, bt, br, bb = np.split(b, 4, axis=-1)
        elif fmt == 'center':
            ax, ay, aw, ah = np.split(a, 4, axis=-1)
            bx, by, bw, bh = np.split(b, 4, axis=-1)
            al, at, ar, ab = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
            bl, bt, br, bb = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
        else:
            raise NotImplementedError("Fmt {} not supported".format(fmt))
        width = np.maximum(0, np.minimum(ar, br) - np.maximum(al, bl))
        height = np.maximum(0, np.minimum(ab, bb) - np.maximum(at, bt))
        intersect = width * height
        union = area(al, at, ar, ab) + area(bl, bt, br, bb) - intersect
        union[np.where(intersect <= 0)] = 1e-12
        iou = intersect / union
        return iou.reshape(oshape)

    def generate_boxes(dims):
        s1, off1, s2, off2 = np.random.rand(4) * 100
        xy = np.random.rand(*(dims + [2])) * s1 + off1
        wh = np.random.rand(*(dims + [2])) * s2 + off2
        xywh = np.concatenate([xy, wh], axis=-1)
        ltrb = np.concatenate([xy - wh / 2, xy + wh / 2], axis=-1)
        return xywh, ltrb


    for ndima in range(1, 6):
        for ndimb in range(1, 6):
            dims_a = np.random.randint(low=1, high=3, size=ndima).tolist()
            dims_b = np.random.randint(low=1, high=3, size=ndimb).tolist()
            # generate left, top, right, bottom
            xywh_a, ltrb_a = generate_boxes(dims_a)
            xywh_b, ltrb_b = generate_boxes(dims_b)

            iou_np = numpy_box_iou(ltrb_a, ltrb_b, fmt='corner')
            iou_np2 = numpy_box_iou(xywh_a, xywh_b, fmt='center')
            iou_mx = mx.nd.contrib.box_iou(mx.nd.array(ltrb_a), mx.nd.array(ltrb_b), format='corner')
            iou_mx2 = mx.nd.contrib.box_iou(mx.nd.array(xywh_a), mx.nd.array(xywh_b), format='center')
            assert_allclose(iou_np, iou_np2, rtol=1e-5, atol=1e-5)
            assert_allclose(iou_np, iou_mx.asnumpy(), rtol=1e-5, atol=1e-5)
            assert_allclose(iou_np, iou_mx2.asnumpy(), rtol=1e-5, atol=1e-5)

def test_bipartite_matching_op():
    def assert_match(inputs, x, y, threshold, is_ascend=False):
        for dtype in ['float16', 'float32', 'float64']:
            inputs = mx.nd.array(inputs, dtype=dtype)
            x = np.array(x, dtype=dtype)
            y = np.array(y, dtype=dtype)
            a, b = mx.nd.contrib.bipartite_matching(inputs, threshold=threshold, is_ascend=is_ascend)
            assert_array_equal(a.asnumpy().astype('int64'), x.astype('int64'))
            assert_array_equal(b.asnumpy().astype('int64'), y.astype('int64'))
    assert_match([[0.5, 0.6], [0.1, 0.2], [0.3, 0.4]], [1, -1, 0], [2, 0], 1e-12, False)
    assert_match([[0.5, 0.6], [0.1, 0.2], [0.3, 0.4]], [-1, 0, 1], [1, 2], 100, True)

def test_multibox_target_op():
    anchors = mx.nd.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], ctx=default_context()).reshape((1, -1, 4))
    cls_pred = mx.nd.array(list(range(10)), ctx=default_context()).reshape((1, -1, 2))
    label = mx.nd.array([1, 0.1, 0.1, 0.5, 0.6], ctx=default_context()).reshape((1, -1, 5))

    loc_target, loc_mask, cls_target = \
        mx.nd.contrib.MultiBoxTarget(anchors, label, cls_pred,
                                     overlap_threshold=0.5,
                                     negative_mining_ratio=3,
                                     negative_mining_thresh=0.4)
    expected_loc_target = np.array([[5.0, 2.5000005, 3.4657357, 4.581454, 0., 0., 0., 0.]])
    expected_loc_mask = np.array([[1, 1, 1, 1, 0, 0, 0, 0]])
    expected_cls_target = np.array([[2, 0]])
    assert_allclose(loc_target.asnumpy(), expected_loc_target, rtol=1e-5, atol=1e-5)
    assert_array_equal(loc_mask.asnumpy(), expected_loc_mask)
    assert_array_equal(cls_target.asnumpy(), expected_cls_target)

def test_gradient_multiplier_op():
    # We use the quadratic function in combination with gradient multiplier
    def f(x, a, b, c):
        return a * x**2 + b * x + c

    a = np.random.random_sample()
    b = np.random.random_sample()
    c = np.random.random_sample()
    m = np.random.random_sample() - 0.5

    data = mx.symbol.Variable('data')
    quad_sym = mx.sym.contrib.quadratic(data=data, a=a, b=b, c=c)
    gr_q_sym = mx.sym.contrib.gradientmultiplier(quad_sym, scalar=m)

    for dtype in [np.float16, np.float32, np.float64]:
        for ndim in range(1, 6):
            shape = rand_shape_nd(ndim, 5)
            data_np = np.random.randn(*shape).astype(dtype)
            expected = f(data_np, a, b, c)
            backward_expected = (2 * a * data_np + b) * m

            # check imperative forward
            output = mx.nd.contrib.quadratic(mx.nd.array(data_np), a=a, b=b, c=c)
            output = mx.nd.contrib.gradientmultiplier(output, scalar=m)
            assert_almost_equal(output.asnumpy(), expected,
                                rtol=1e-2 if dtype is np.float16 else 1e-5,
                                atol=1e-2 if dtype is np.float16 else 1e-5)
            # check forward
            check_symbolic_forward(gr_q_sym, [data_np], [expected],
                                    rtol=1e-2 if dtype is np.float16 else 1e-5,
                                    atol=1e-2 if dtype is np.float16 else 1e-5)
            # check backward
            check_symbolic_backward(gr_q_sym, [data_np], [np.ones(expected.shape)],
                                        [backward_expected],
                                        rtol=1e-2 if dtype is np.float16 else 1e-5,
                                        atol=1e-2 if dtype is np.float16 else 1e-5)
def test_multibox_prior_op():
    h = 561
    w = 728
    X = mx.nd.random.uniform(shape=(1, 3, h, w))
    Y = mx.contrib.nd.MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    assert_array_equal(Y.shape, np.array((1, 2042040, 4)))
    boxes = Y.reshape((h, w, 5, 4))
    assert_allclose(boxes.asnumpy()[250, 250, 0, :], np.array([0.055117, 0.071524, 0.63307 , 0.821524]), atol=1e-5, rtol=1e-5)
    # relax first ratio if user insists
    Y = mx.contrib.nd.MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[20, 2, 0.5])
    boxes = Y.reshape((h, w, 5, 4))
    assert_allclose(boxes.asnumpy()[250, 250, 0, :], np.array([-0.948249,  0.362671,  1.636436,  0.530377]), atol=1e-5, rtol=1e-5)

if __name__ == '__main__':
    import nose
    nose.runmodule()
