import mxnet as mx
import numpy as np
import os
import cPickle


class Detector(object):
    def __init__(self, symbol, ctx=None,
                 arg_params=None, aux_params=None):
        self.symbol = symbol
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        self.executor = None
        self.arg_params = arg_params
        self.aux_params = aux_params

    def im_detect(self, im, im_info):
        """
        perform detection of im, im_info
        :param im: numpy.ndarray [b, c, h, w]
        :param im_info: numpy.ndarray [b, 3]
        :return: boxes [b, 5], scores [b,]
        """
        self.arg_params['data'] = mx.nd.array(im, self.ctx)
        self.arg_params['im_info'] = mx.nd.array(im_info, self.ctx)
        arg_shapes, out_shapes, aux_shapes = \
            self.symbol.infer_shape(data=self.arg_params['data'].shape, im_info=self.arg_params['im_info'].shape)
        aux_names = self.symbol.list_auxiliary_states()
        self.aux_params = {k: mx.nd.zeros(s, self.ctx) for k, s in zip(aux_names, aux_shapes)}
        self.executor = self.symbol.bind(self.ctx, self.arg_params, args_grad=None,
                                         grad_req='null', aux_states=self.aux_params)
        output_dict = {name: nd for name, nd in zip(self.symbol.list_outputs(), self.executor.outputs)}

        self.executor.forward(is_train=False)
        boxes = output_dict['rois_output'].asnumpy()
        scores = output_dict['rois_score'].asnumpy()

        return boxes, scores


def generate_detections(detector, test_data, imdb, vis=False):
    """
    Generate detections results using RPN.
    :param detector: Detector
    :param test_data: data iterator, must be non-shuffled
    :param imdb: image database
    :param vis: controls visualization
    :return: list of detected boxes
    """
    assert not test_data.shuffle

    i = 0
    imdb_boxes = list()
    for databatch in test_data:
        if i % 10 == 0:
            print 'generating detections {}/{}'.format(i, imdb.num_images)

        boxes, scores = detector.im_detect(databatch.data['data'], databatch.data['im_info'])
        scale = databatch.data['im_info'][0, 2]
        # drop the batch index
        boxes = boxes[:, 1:].copy() / scale
        imdb_boxes.append(boxes)
        if vis:
            dets = np.hstack((boxes * scale, scores))
            vis_detection(databatch.data['data'], dets, thresh=0.9)
        i += 1

    assert len(imdb_boxes) == imdb.num_images, 'calculations not complete'
    rpn_folder = os.path.join(imdb.root_path, 'rpn_data')
    if not os.path.exists(rpn_folder):
        os.mkdir(rpn_folder)
    rpn_file = os.path.join(rpn_folder, imdb.name + '_rpn.pkl')
    with open(rpn_file, 'wb') as f:
        cPickle.dump(imdb_boxes, f, cPickle.HIGHEST_PROTOCOL)
    print 'wrote rpn proposals to {}'.format(rpn_file)
    return imdb_boxes


def vis_detection(im, dets, thresh=0.):
    """
    draw detected bounding boxes
    :param im: [b, c, h, w] oin rgb
    :param dets: only one class, [N * [4 coordinates score]]
    :param thresh: thresh for valid detections
    :return:
    """
    from rcnn.config import config
    from helper.processing.image_processing import transform_inverse
    import matplotlib.pyplot as plt
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    inds = np.argsort(dets[:, -1])[::-1]
    inds = inds[:20]

    class_name = 'obj'
    fig, ax = plt.subplots(figsize=(12, 12))
    im = transform_inverse(im, config.PIXEL_MEANS)
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2] - bbox[0],
                             bbox[3] - bbox[1], fill=False,
                             edgecolor='red', linewidth=3.5)
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5), fontsize=14, color='white')
    ax.set_title('{} detections with p({} | box) >= {:.1f}'.format(class_name, class_name, thresh), fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()
