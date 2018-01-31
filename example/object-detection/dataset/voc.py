"""Pascal VOC dataset."""
import os
import logging
import numpy as np
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
try:
    import cPickle as pickle
except ImportError:
    import pickle
from mxnet import image
from dataset.base import DetectionDataset
from dataset.utils import mkdirs_p


class VOCDetection(DetectionDataset):
    """Pascal VOC detection Dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder.
    sets : list of tuples
        List of combinations of (year, name), e.g. [(2007, 'trainval'), (2012, 'train')].
        For years, candidates can be: 2007, 2012.
        For names, candidates can be: 'train', 'val', 'trainval', 'test'.
    flag : {0, 1}, default 1
        If 0, always convert images to greyscale.

        If 1, always convert images to colored (RGB).
    transform : callable, optional
        A function that takes data and label and transforms them::

            transform = lambda data, label: (data.astype(np.float32)/255, label)
        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    index_map : dict, optional
        If provided as dict, class indecies are mapped by looking up in the dict.
        Otherwise will use alphabetic indexing for all classes from 0 to 19.
    preload : bool
        All labels will be parsed and loaded into memory at initialization.
        This will allow early check for errors, and will be faster.
    """
    def __init__(self, root, sets, flag=1, transform=None, index_map=None, preload=True):
        super(VOCDetection, self).__init__('voc')
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._flag = flag
        self._transform = transform
        self._sets = sets
        self._items = self._load_items(sets)
        self._anno_path = os.path.join('{}', 'Annotations', '{}.xml')
        self._image_path = os.path.join('{}', 'JPEGImages', '{}.jpg')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_classes)))
        self._label_cache = self._preload_labels() if preload else None
        self._comp = 'comp4'
        self._cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')
        self._result_dir = os.path.join(self._root, 'result')

    def __str__(self):
        detail = ','.join([str(s[0]) + s[1] for s in self._sets])
        return self.__class__.__name__ + '(' + detail + ')'

    def _load_items(self, sets):
        """Load individual image indices from sets."""
        ids = []
        for year, name in sets:
            root = os.path.join(self._root, 'VOC' + str(year))
            lf = os.path.join(root, 'ImageSets', 'Main', name + '.txt')
            with open(lf, 'r') as f:
                ids += [(root, line.strip()) for line in f.readlines()]
        return ids

    def _load_label(self, idx):
        """Parse xml file and return labels."""
        img_id = self._items[idx]
        anno_path = self._anno_path.format(*img_id)
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            difficult = int(obj.find('difficult').text)
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text) - 1) / width
            ymin = (float(xml_box.find('ymin').text) - 1) / height
            xmax = (float(xml_box.find('xmax').text) - 1) / width
            ymax = (float(xml_box.find('ymax').text) - 1) / height
            try:
                self._validator(xmin, ymin, xmax, ymax)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
            label.append([cls_id, xmin, ymin, xmax, ymax, difficult])
        return np.array(label)

    def _validator(self, xmin, ymin, xmax, ymax):
        """Validate labels."""
        assert xmin >= 0 and xmin < 1.0, "xmin must in [0, 1), given {}".format(xmin)
        assert ymin >= 0 and ymin < 1.0, "ymin must in [0, 1), given {}".format(ymin)
        assert xmax > xmin and ymin <= 1.0, "xmax must in (xmin, 1], given {}".format(xmax)
        assert ymax > ymin and ymax <= 1.0, "ymax must in (ymin, 1], given {}".format(ymax)

    def _preload_labels(self):
        """Preload all labels into memory."""
        logging.debug("Preloading {} labels into memory...".format(str(self)))
        return [self._load_label(idx) for idx in range(self.__len__())]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(*img_id)
        label = self._label_cache[idx] if self._label_cache else self._load_label(idx)
        img = image.imread(img_path, self._flag)
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def eval_results(self, results):
        """Evaluate results.


        """
        assert len(self._sets) == 1, "concatenated sets are not supposed to be evaluated."
        assert isinstance(results, np.ndarray), (
            "np.ndarray expected, given {}".format(type(results)))
        assert len(self._items) == results.shape[0], (
            "# image mismatch: {} vs. {}".format(len(self._items), results.shape[0]))
        self._write_results(results)
        return self.do_python_eval()

    def _get_filename_template(self):
        """Get filename template."""
        dir_name = os.path.join(self._result_dir, 'VOC' + str(self._sets[0][0]), self._comp)
        mkdirs_p(dir_name)
        return os.path.join(dir_name, self._comp + '_det_' + self._sets[0][1] + '_{:s}.txt')

    def _write_results(self, results):
        """Write results to disk in compliance with PASCAL formats."""
        for cls_name in self.classes:
            logging.info('Writing {} VOC results file'.format(cls_name))
            filename = self._get_filename_template().format(cls_name)
            buf = []
            for im_ind, index in enumerate(self._items):
                dets = results[im_ind]
                if dets.shape[0] < 1:
                    continue
                if im_ind not in self._im_shapes:
                    self._load_label(im_ind)
                w, h = self._im_shapes[im_ind]
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    if (int(dets[k, 0]) == self.index_map[cls_name]):
                        buf.append('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, 1],
                                   int(dets[k, 2] * w) + 1, int(dets[k, 3] * h) + 1,
                                   int(dets[k, 4] * w) + 1, int(dets[k, 5] * h) + 1))
            whole = ''.join(buf)
            with open(filename, 'wt') as f:
                f.write(whole)

    def do_python_eval(self):
        """Apply python evaluation functions."""
        data_path = os.path.join(self._root, 'VOC' + str(self._sets[0][0]))
        annopath = os.path.join(data_path, 'Annotations', '{}.xml')
        imageset_file = os.path.join(data_path, 'ImageSets', 'Main', self._sets[0][1] + '.txt')
        aps = []
        use_07_metric = True if int(self._sets[0][0]) < 2010 else False
        logging.info("Use VOC07 metric? " + ('Yes' if use_07_metric else 'No'))
        for cls_ind, cls_name in enumerate(self.classes):
            filename = self._get_filename_template().format(cls_name)
            rec, prec, ap = self._voc_eval(
                filename, annopath, imageset_file, cls_name, self._cache_dir,
                ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            logging.info("AP for {} = {:.4f}".format(cls_name, ap))
        mean_ap = np.mean(aps)
        logging.info("Mean AP = {:.4f}".format(mean_ap))
        return 'Mean AP', mean_ap

    def _parse_voc_rec(self, filename):
        """
        parse pascal voc record into a dictionary
        :param filename: xml file path
        :return: list of dict
        """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_dict = dict()
            obj_dict['name'] = obj.find('name').text
            obj_dict['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_dict['bbox'] = [int(bbox.find('xmin').text),
                                int(bbox.find('ymin').text),
                                int(bbox.find('xmax').text),
                                int(bbox.find('ymax').text)]
            objects.append(obj_dict)
        return objects


    def _voc_ap(self, rec, prec, use_07_metric=False):
        """
        average precision calculations
        [precision integrated to recall]
        :param rec: recall
        :param prec: precision
        :param use_07_metric: 2007 metric is 11-recall-point based AP
        :return: average precision
        """
        if use_07_metric:
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap += p / 11.
        else:
            # append sentinel values at both ends
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute precision integration ladder
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # look for recall value changes
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # sum (\delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


    def _voc_eval(self, detpath, annopath, imageset_file, classname, cache_dir, ovthresh=0.5, use_07_metric=False):
        """
        pascal voc evaluation
        :param detpath: detection results detpath.format(classname)
        :param annopath: annotations annopath.format(classname)
        :param imageset_file: text file containing list of images
        :param classname: category name
        :param cache_dir: caching annotations
        :param ovthresh: overlap threshold
        :param use_07_metric: whether to use voc07's 11 point ap computation
        :return: rec, prec, ap
        """
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
        cache_file = os.path.join(cache_dir, 'annotations.pkl')
        with open(imageset_file, 'r') as f:
            lines = f.readlines()
        image_filenames = [x.strip() for x in lines]

        # load annotations from cache
        if not os.path.isfile(cache_file):
            recs = {}
            for ind, image_filename in enumerate(image_filenames):
                recs[image_filename] = self._parse_voc_rec(annopath.format(image_filename))
                if ind % 1000 == 0:
                    logging.debug('reading annotations for {:d}/{:d}'.format(ind + 1, len(image_filenames)))
            logging.debug('saving annotations cache to {:s}'.format(cache_file))
            with open(cache_file, 'wb') as f:
                pickle.dump(recs, f)
        else:
            with open(cache_file, 'rb') as f:
                recs = pickle.load(f)

        # extract objects in :param classname:
        class_recs = {}
        npos = 0
        for image_filename in image_filenames:
            objects = [obj for obj in recs[image_filename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in objects])
            difficult = np.array([x['difficult'] for x in objects]).astype(np.bool)
            det = [False] * len(objects)  # stand for detected
            npos = npos + sum(~difficult)
            class_recs[image_filename] = {'bbox': bbox,
                                          'difficult': difficult,
                                          'det': det}

        # read detections
        detfile = detpath.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()

        if not lines:
            return 0.0, 0.0, 0.0

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        bbox = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_inds = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        bbox = bbox[sorted_inds, :]
        image_ids = [image_ids[x] for x in sorted_inds]

        # go down detections and mark true positives and false positives
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            r = class_recs[image_ids[d]]
            bb = bbox[d, :].astype(float)
            ovmax = -np.inf
            bbgt = r['bbox'].astype(float)

            if bbgt.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(bbgt[:, 0], bb[0])
                iymin = np.maximum(bbgt[:, 1], bb[1])
                ixmax = np.minimum(bbgt[:, 2], bb[2])
                iymax = np.minimum(bbgt[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (bbgt[:, 2] - bbgt[:, 0] + 1.) *
                       (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not r['difficult'][jmax]:
                    if not r['det'][jmax]:
                        tp[d] = 1.
                        r['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid division by zero in case first detection matches a difficult ground ruth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self._voc_ap(rec, prec, use_07_metric)

        return rec, prec, ap
