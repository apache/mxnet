import os
from imdb import Imdb


class TestDB(Imdb):
    """
    A simple wrapper class for converting list of image to Imdb during testing

    Parameters:
    ----------
    images : str or list of str
        image path or list of images, if directory and extension not
        specified, root_dir and extension are required
    root_dir : str or None
        directory of input images, optional if image path already
        has full directory information
    extension : str or None
        image extension, eg. ".jpg", optional
    """
    def __init__(self, images, root_dir=None, extension=None):
        if not isinstance(images, list):
            images = [images]
        num_images = len(images)
        super(TestDB, self).__init__("test" + str(num_images))
        self.image_set_index = images
        self.num_images = num_images
        self.root_dir = root_dir if root_dir else None
        self.extension = extension if extension else None


    def image_path_from_index(self, index):
        """
        given image index, return full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns
        ----------
        path of this image
        """
        name = self.image_set_index[index]
        if self.extension:
            name += self.extension
        if self.root_dir:
            name = os.path.join(self.root_dir, name)
        assert os.path.exists(name), 'Path does not exist: {}'.format(name)
        return name

    def label_from_index(self, index):
        return RuntimeError("Testdb does not support label loading")
