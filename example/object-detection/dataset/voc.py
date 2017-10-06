"""Pascal VOC dataset."""
import os
import numpy as np
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from mxnet import image
from dataset.base import DetectionDataset


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
        self._root = os.path.expanduser(root)
        self._flag = flag
        self._transform = transform
        self._items = self._load_items(sets)
        self._anno_path = os.path.join('{}', 'Annotations', '{}.xml')
        self._image_path = os.path.join('{}', 'JPEGImages', '{}.jpg')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_classes)))
        self._label_cache = self._preload_labels() if preload else None

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
