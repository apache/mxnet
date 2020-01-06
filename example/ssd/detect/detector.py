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

import mxnet as mx
import numpy as np
from timeit import default_timer as timer
from dataset.testdb import TestDB
from dataset.iterator import DetIter
import logging
import cv2
from mxnet.io import DataBatch, DataDesc


class Detector(object):
    """
    SSD detector which hold a detection network and wraps detection API

    Parameters:
    ----------
    symbol : mx.Symbol
        detection network Symbol
    model_prefix : str
        name prefix of trained model
    epoch : int
        load epoch of trained model
    data_shape : int
        input data resize shape
    mean_pixels : tuple of float
        (mean_r, mean_g, mean_b)
    batch_size : int
        run detection with batch size
    ctx : mx.ctx
        device to use, if None, use mx.cpu() as default context
    """
    def __init__(self, symbol, model_prefix, epoch, data_shape, mean_pixels, \
                 batch_size=1, ctx=None):
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        load_symbol, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
        if symbol is None:
            symbol = load_symbol
        self.mod = mx.mod.Module(symbol, label_names=None, context=self.ctx)
        if not isinstance(data_shape, tuple):
            data_shape = (data_shape, data_shape)
        self.data_shape = data_shape
        self.mod.bind(data_shapes=[('data', (batch_size, 3, data_shape[0], data_shape[1]))])
        self.mod.set_params(args, auxs)
        self.mean_pixels = mean_pixels
        self.mean_pixels_nd = mx.nd.array(mean_pixels).reshape((3,1,1))

    def create_batch(self, frame):
        """
        :param frame: an (w,h,channels) numpy array (image)
        :return: DataBatch of (1,channels,data_shape,data_shape)
        """
        frame_resize = mx.nd.array(cv2.resize(frame, (self.data_shape[0], self.data_shape[1])))
        #frame_resize = mx.img.imresize(frame, self.data_shape[0], self.data_shape[1], cv2.INTER_LINEAR)
        # Change dimensions from (w,h,channels) to (channels, w, h)
        frame_t = mx.nd.transpose(frame_resize, axes=(2,0,1))
        frame_norm = frame_t - self.mean_pixels_nd
        # Add dimension for batch, results in (1,channels,w,h)
        batch_frame = [mx.nd.expand_dims(frame_norm, axis=0)]
        batch_shape = [DataDesc('data', batch_frame[0].shape)]
        batch = DataBatch(data=batch_frame, provide_data=batch_shape)
        return batch

    def detect_iter(self, det_iter, show_timer=False):
        """
        detect all images in iterator

        Parameters:
        ----------
        det_iter : DetIter
            iterator for all testing images
        show_timer : Boolean
            whether to print out detection exec time

        Returns:
        ----------
        list of detection results
        """
        num_images = det_iter._size
        if not isinstance(det_iter, mx.io.PrefetchingIter):
            det_iter = mx.io.PrefetchingIter(det_iter)
        start = timer()
        detections = self.mod.predict(det_iter).asnumpy()
        time_elapsed = timer() - start
        if show_timer:
            logging.info("Detection time for {} images: {:.4f} sec".format(
                num_images, time_elapsed))
        result = Detector.filter_positive_detections(detections)
        return result

    def detect_batch(self, batch):
        """
        Return detections for batch
        :param batch:
        :return:
        """
        self.mod.forward(batch, is_train=False)
        detections = self.mod.get_outputs()[0]
        positive_detections = Detector.filter_positive_detections(detections)
        return positive_detections

    def im_detect(self, im_list, root_dir=None, extension=None, show_timer=False):
        """
        wrapper for detecting multiple images

        Parameters:
        ----------
        im_list : list of str
            image path or list of image paths
        root_dir : str
            directory of input images, optional if image path already
            has full directory information
        extension : str
            image extension, eg. ".jpg", optional

        Returns:
        ----------
        list of detection results in format [det0, det1...], det is in
        format np.array([id, score, xmin, ymin, xmax, ymax]...)
        """
        test_db = TestDB(im_list, root_dir=root_dir, extension=extension)
        test_iter = DetIter(test_db, 1, self.data_shape, self.mean_pixels,
                            is_train=False)
        return self.detect_iter(test_iter, show_timer)

    def visualize_detection(self, img, dets, classes=[], thresh=0.6):
        """
        visualize detections in one image

        Parameters:
        ----------
        img : numpy.array
            image, in bgr format
        dets : numpy.array
            ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
        classes : tuple or list of str
            class names
        thresh : float
            score threshold
        """
        import matplotlib.pyplot as plt
        import random
        plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]
        colors = dict()
        for det in dets:
            (klass, score, x0, y0, x1, y1) = det
            if score < thresh:
                continue
            cls_id = int(klass)
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            xmin = int(x0 * width)
            ymin = int(y0 * height)
            xmax = int(x1 * width)
            ymax = int(y1 * height)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=3.5)
            plt.gca().add_patch(rect)
            class_name = str(cls_id)
            if classes and len(classes) > cls_id:
                class_name = classes[cls_id]
            plt.gca().text(xmin, ymin - 2,
                            '{:s} {:.3f}'.format(class_name, score),
                            bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                                    fontsize=12, color='white')
        plt.show()

    @staticmethod
    def filter_positive_detections(detections):
        """
        First column (class id) is -1 for negative detections
        :param detections:
        :return:
        """
        class_idx = 0
        assert(isinstance(detections, mx.nd.NDArray) or isinstance(detections, np.ndarray))
        detections_per_image = []
        # for each image
        for i in range(detections.shape[0]):
            result = []
            det = detections[i, :, :]
            for obj in det:
                if obj[class_idx] >= 0:
                    result.append(obj)
            detections_per_image.append(result)
        logging.info("%d positive detections", len(result))
        return detections_per_image

    def detect_and_visualize(self, im_list, root_dir=None, extension=None,
                             classes=[], thresh=0.6, show_timer=False):
        """
        wrapper for im_detect and visualize_detection

        Parameters:
        ----------
        im_list : list of str or str
            image path or list of image paths
        root_dir : str or None
            directory of input images, optional if image path already
            has full directory information
        extension : str or None
            image extension, eg. ".jpg", optional

        Returns:
        ----------

        """
        dets = self.im_detect(im_list, root_dir, extension, show_timer=show_timer)
        if not isinstance(im_list, list):
            im_list = [im_list]
        assert len(dets) == len(im_list)
        for k, det in enumerate(dets):
            img = cv2.imread(im_list[k])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.visualize_detection(img, det, classes, thresh)
