import os
import numpy as np
from imdb import Imdb


class YoloFormat(Imdb):
    """
    Base class for loading datasets as used in YOLO

    Parameters:
    ----------
    name : str
        name for this dataset
    classes : list or tuple of str
        class names in this dataset
    list_file : str
        filename of the image list file
    image_dir : str
        image directory
    label_dir : str
        label directory
    extension : str
        by default .jpg
    label_extension : str
        by default .txt
    shuffle : bool
        whether to shuffle the initial order when loading this dataset,
        default is True
    """
    def __init__(self, name, classes, list_file, image_dir, label_dir, \
                 extension='.jpg', label_extension='.txt', shuffle=True):
        if isinstance(classes, list) or isinstance(classes, tuple):
            num_classes = len(classes)
        elif isinstance(classes, str):
            with open(classes, 'r') as f:
                classes = [l.strip() for l in f.readlines()]
                num_classes = len(classes)
        else:
            raise ValueError("classes should be list/tuple or text file")
        assert num_classes > 0, "number of classes must > 0"
        super(YoloFormat, self).__init__(name + '_' + str(num_classes))
        self.classes = classes
        self.num_classes = num_classes
        self.list_file = list_file
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.extension = extension
        self.label_extension = label_extension

        self.image_set_index = self._load_image_set_index(shuffle)
        self.num_images = len(self.image_set_index)
        self.labels = self._load_image_labels()


    def _load_image_set_index(self, shuffle):
        """
        find out which indexes correspond to given image set (train or val)

        Parameters:
        ----------
        shuffle : boolean
            whether to shuffle the image list
        Returns:
        ----------
        entire list of images specified in the setting
        """
        assert os.path.exists(self.list_file), 'Path does not exists: {}'.format(self.list_file)
        with open(self.list_file, 'r') as f:
            image_set_index = [x.strip() for x in f.readlines()]
        if shuffle:
            np.random.shuffle(image_set_index)
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = os.path.join(self.image_dir, name) + self.extension
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index, :, :]

    def _label_path_from_index(self, index):
        """
        given image index, find out annotation path

        Parameters:
        ----------
        index: int
            index of a specific image

        Returns:
        ----------
        full path of annotation file
        """
        label_file = os.path.join(self.label_dir, index + self.label_extension)
        assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)
        return label_file

    def _load_image_labels(self):
        """
        preprocess all ground-truths

        Returns:
        ----------
        labels packed in [num_images x max_num_objects x 5] tensor
        """
        temp = []
        max_objects = 0

        # load ground-truths
        for idx in self.image_set_index:
            label_file = self._label_path_from_index(idx)
            with open(label_file, 'r') as f:
                label = []
                for line in f.readlines():
                    temp_label = line.strip().split()
                    assert len(temp_label) == 5, "Invalid label file" + label_file
                    cls_id = int(temp_label[0])
                    x = float(temp_label[1])
                    y = float(temp_label[2])
                    half_width = float(temp_label[3]) / 2
                    half_height = float(temp_label[4]) / 2
                    xmin = x - half_width
                    ymin = y - half_height
                    xmax = x + half_width
                    ymax = y + half_height
                    label.append([cls_id, xmin, ymin, xmax, ymax])
                temp.append(np.array(label))
                max_objects = max(max_objects, len(label))
        # add padding to labels so that the dimensions match in each batch
        assert max_objects > 0, "No objects found for any of the images"
        self.padding = max_objects
        labels = []
        for label in temp:
            label = np.lib.pad(label, ((0, max_objects-label.shape[0]), (0,0)), \
                               'constant', constant_values=(-1, -1))
            labels.append(label)
        return np.array(labels)
