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
        self.image_set_index = None
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
        def progress_bar(count, total, suffix=''):
            import sys
            bar_len = 24
            filled_len = int(round(bar_len * count / float(total)))

            percents = round(100.0 * count / float(total), 1)
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
            sys.stdout.flush()

        str_list = []
        for index in range(self.num_images):
            progress_bar(index, self.num_images)
            label = self.label_from_index(index)
            if label.size < 1:
                continue
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

    def _load_class_names(self, filename, dirname):
        """
        load class names from text file

        Parameters:
        ----------
        filename: str
            file stores class names
        dirname: str
            file directory
        """
        full_path = osp.join(dirname, filename)
        classes = []
        with open(full_path, 'r') as f:
            classes = [l.strip() for l in f.readlines()]
        return classes
