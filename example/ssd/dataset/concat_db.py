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

from imdb import Imdb
import random

class ConcatDB(Imdb):
    """
    ConcatDB is used to concatenate multiple imdbs to form a larger db.
    It is very useful to combine multiple dataset with same classes.
    Parameters
    ----------
    imdbs : Imdb or list of Imdb
        Imdbs to be concatenated
    shuffle : bool
        whether to shuffle the initial list
    """
    def __init__(self, imdbs, shuffle):
        super(ConcatDB, self).__init__('concatdb')
        if not isinstance(imdbs, list):
            imdbs = [imdbs]
        self.imdbs = imdbs
        self._check_classes()
        self.image_set_index = self._load_image_set_index(shuffle)

    def _check_classes(self):
        """
        check input imdbs, make sure they have same classes
        """
        try:
            self.classes = self.imdbs[0].classes
            self.num_classes = len(self.classes)
        except AttributeError:
            # fine, if no classes is provided
            pass

        if self.num_classes > 0:
            for db in self.imdbs:
                assert self.classes == db.classes, "Multiple imdb must have same classes"

    def _load_image_set_index(self, shuffle):
        """
        get total number of images, init indices

        Parameters
        ----------
        shuffle : bool
            whether to shuffle the initial indices
        """
        self.num_images = 0
        for db in self.imdbs:
            self.num_images += db.num_images
        indices = list(range(self.num_images))
        if shuffle:
            random.shuffle(indices)
        return indices

    def _locate_index(self, index):
        """
        given index, find out sub-db and sub-index

        Parameters
        ----------
        index : int
            index of a specific image

        Returns
        ----------
        a tuple (sub-db, sub-index)
        """
        assert index >= 0 and index < self.num_images, "index out of range"
        pos = self.image_set_index[index]
        for k, v in enumerate(self.imdbs):
            if pos >= v.num_images:
                pos -= v.num_images
            else:
                return (k, pos)

    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters
        ----------
        index: int
            index of a specific image

        Returns
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        pos = self.image_set_index[index]
        n_db, n_index = self._locate_index(index)
        return self.imdbs[n_db].image_path_from_index(n_index)

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters
        ----------
        index: int
            index of a specific image

        Returns
        ----------
        ground-truths of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        pos = self.image_set_index[index]
        n_db, n_index = self._locate_index(index)
        return self.imdbs[n_db].label_from_index(n_index)
