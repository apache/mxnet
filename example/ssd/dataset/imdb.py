import numpy as np
import os.path as osp

class Imdb(object):
    """
    Base class for dataset loading

    Parameters:
    ----------
    name : str
        name of dataset
    """
    def __init__(self, name):
        self.name = name
        self.classes = []
        self.num_classes = 0
        self.image_set_index = []
        self.num_images = 0
        self.labels = None
        self.padding = 0

    def image_path_from_index(self, index):
        """
        load image full path given specified index

        Parameters:
        ----------
        index : int
            index of image requested in dataset

        Returns:
        ----------
        full path of specified image
        """
        raise NotImplementedError

    def label_from_index(self, index):
        """
        load ground-truth of image given specified index

        Parameters:
        ----------
        index : int
            index of image requested in dataset

        Returns:
        ----------
        object ground-truths, in format
        numpy.array([id, xmin, ymin, xmax, ymax]...)
        """
        raise NotImplementedError

    def save_imglist(self, fname=None, root=None, shuffle=False):
        """
        save imglist to disk

        Parameters:
        ----------
        fname : str
            saved filename
        """
        str_list = []
        for index in range(self.num_images):
            label = self.label_from_index(index)
            path = self.image_path_from_index(index)
            if root:
                path = osp.relpath(path, root)
            str_list.append('\t'.join([str(index), str(2), str(label.shape[1])] \
              + ["{0:.4f}".format(x) for x in label.ravel()] + [path,]) + '\n')
        if str_list:
            if shuffle:
                import random
                random.shuffle(str_list)
            if not fname:
                fname = self.name + '.lst'
            with open(fname, 'w') as f:
                for line in str_list:
                    f.write(line)
        else:
            raise RuntimeError("No image in imdb")
