"""Base detection dataset methods."""
import os
from mxnet.gluon.data import dataset


class DetectionDataset(dataset.Dataset):
    """Base detection Dataset.

    Parameters
    ----------
    name : str
        The name of dataset, by default, dataset/names/{}.names will be loaded,
        where names of classes is defined.
    root : str
        The root path of xxx.names, by defaut is 'dataset/names/'
    """
    def __init__(self, name, root=None):
        if root is None:
            root = os.path.join(os.path.dirname(__file__), 'names')
        else:
            assert isinstance(root, str), "Provided root must be str"
        name_path = os.path.join(root, name + '.names')
        with open(name_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.num_classes = len(self.classes)
