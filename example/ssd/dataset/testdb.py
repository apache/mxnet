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

import os
from .imdb import Imdb


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
