import mxnet as mx
import numpy as np

from rcnn.config import config
from helper.processing.bbox_transform import bbox_pred, clip_boxes


class Detector(object):
    def __init__(self, symbol, ctx=None,
                 arg_params=None, aux_params=None):
        self.symbol = symbol
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        self.arg_params = arg_params
        self.aux_params = aux_params
        self.executor = None

    def im_detect(self, im_array, roi_array):
        """
        perform detection of designated im, box, must follow minibatch.get_testbatch format
        :param im_array: numpy.ndarray [b c h w]
        :param roi_array: numpy.ndarray [roi_num 5]
        :return: scores, pred_boxes
        """
        # remove duplicate feature rois
        if config.TEST.DEDUP_BOXES > 0:
            roi_array = roi_array
            # rank roi by v .* (b, dx, dy, dw, dh)
            v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            # create hash and inverse index for rois
            hashes = np.round(roi_array * config.TEST.DEDUP_BOXES).dot(v)
            _, index, inv_index = np.unique(hashes, return_index=True, return_inverse=True)
            roi_array = roi_array[index, :]

        self.arg_params['data'] = mx.nd.array(im_array, self.ctx)
        self.arg_params['rois'] = mx.nd.array(roi_array, self.ctx)
        arg_shapes, out_shapes, aux_shapes = \
            self.symbol.infer_shape(data=self.arg_params['data'].shape, rois=self.arg_params['rois'].shape)
        arg_shapes_dict = {name: shape for name, shape in zip(self.symbol.list_arguments(), arg_shapes)}
        self.arg_params['cls_prob_label'] = mx.nd.zeros(arg_shapes_dict['cls_prob_label'], self.ctx)

        aux_names = self.symbol.list_auxiliary_states()
        self.aux_params = {k: mx.nd.zeros(s, self.ctx) for k, s in zip(aux_names, aux_shapes)}
        self.executor = self.symbol.bind(self.ctx, self.arg_params, args_grad=None,
                                         grad_req='null', aux_states=self.aux_params)
        output_dict = {name: nd for name, nd in zip(self.symbol.list_outputs(), self.executor.outputs)}

        self.executor.forward(is_train=False)
        scores = output_dict['cls_prob_output'].asnumpy()
        bbox_deltas = output_dict['bbox_pred_output'].asnumpy()

        pred_boxes = bbox_pred(roi_array[:, 1:], bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, im_array[0].shape[-2:])

        if config.TEST.DEDUP_BOXES > 0:
            # map back to original
            scores = scores[inv_index, :]
            pred_boxes = pred_boxes[inv_index, :]

        return scores, pred_boxes
