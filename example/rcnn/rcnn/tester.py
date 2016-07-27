import numpy as np
import cv2
import os
import cPickle
from rcnn.config import config
from helper.processing import image_processing
from helper.processing.nms import nms


def pred_eval(detector, test_data, imdb, vis=False):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param detector: Detector
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :return:
    """
    assert not test_data.shuffle

    thresh = 0.05
    # limit detections to max_per_image over all classes
    max_per_image = 100

    num_images = imdb.num_images
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    i = 0
    for databatch in test_data:
        if i % 10 == 0:
            print 'testing {}/{}'.format(i, imdb.num_images)

        if config.TEST.HAS_RPN:
            scores, boxes = detector.im_detect(databatch.data['data'], im_info=databatch.data['im_info'])
            scale = databatch.data['im_info'][0, 2]
        else:
            scores, boxes = detector.im_detect(databatch.data['data'], roi_array=databatch.data['rois'])
            # we used scaled image & roi to train, so it is necessary to transform them back
            # visualization should also be from the original size
            im_path = imdb.image_path_from_index(imdb.image_set_index[i])
            im = cv2.imread(im_path)
            im_height = im.shape[0]
            scale = float(databatch.data['data'].shape[2]) / float(im_height)

        for j in range(1, imdb.num_classes):
            indexes = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[indexes, j]
            cls_boxes = boxes[indexes, j * 4:(j + 1) * 4] / scale
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))
            keep = nms(cls_dets, config.TEST.NMS)
            all_boxes[j][i] = cls_dets[keep, :]

        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        boxes_this_image = [[]] + [all_boxes[j][i] for j in range(1, imdb.num_classes)]
        if vis:
            # visualize the testing scale
            for box in boxes_this_image:
                if isinstance(box, np.ndarray):
                    box[:, :4] *= scale
            vis_all_detection(databatch.data['data'], boxes_this_image,
                              imdb_classes=imdb.classes)
        i += 1

    cache_folder = os.path.join(imdb.cache_path, imdb.name)
    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)
    det_file = os.path.join(cache_folder, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f)

    imdb.evaluate_detections(all_boxes)


def vis_all_detection(im_array, detections, imdb_classes=None, thresh=0.7):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param imdb_classes: list of names in imdb
    :param thresh: threshold for valid detections
    :return:
    """
    import matplotlib.pyplot as plt
    import random
    im = image_processing.transform_inverse(im_array, config.PIXEL_MEANS)
    plt.imshow(im)
    for j in range(1, len(imdb_classes)):
        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        for i in range(dets.shape[0]):
            bbox = dets[i, :4]
            score = dets[i, -1]
            if score > thresh:
                rect = plt.Rectangle((bbox[0], bbox[1]),
                                     bbox[2] - bbox[0],
                                     bbox[3] - bbox[1], fill=False,
                                     edgecolor=color, linewidth=3.5)
                plt.gca().add_patch(rect)
                plt.gca().text(bbox[0], bbox[1] - 2,
                               '{:s} {:.3f}'.format(imdb_classes[j], score),
                               bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()
