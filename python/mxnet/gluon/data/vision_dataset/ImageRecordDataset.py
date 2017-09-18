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

# coding: utf-8
# pylint: disable=
"""Dataset container."""

import os
import gzip
import tarfile
import struct
import warnings
import numpy as np

from .. import dataset
from ...utils import download, check_sha1
from .... import nd, image, recordio

from .utils import _DownloadedDataset


class ImageRecordDataset(dataset.RecordFileDataset):
    """A dataset wrapping over a RecordIO file containing images.

    Each sample is an image and its corresponding label.

    Parameters
    ----------
    filename : str
        Path to rec file.
    flag : {0, 1}, default 1
        If 0, always convert images to greyscale.

        If 1, always convert images to colored (RGB).
    transform : function
        A user defined callback that transforms each instance. For example::

            transform=lambda data, label: (data.astype(np.float32)/255, label)
    """
    def __init__(self, filename, flag=1, transform=None):
        super(ImageRecordDataset, self).__init__(filename)
        self._flag = flag
        self._transform = transform

    def __getitem__(self, idx):
        record = super(ImageRecordDataset, self).__getitem__(idx)
        header, img = recordio.unpack(record)
        if self._transform is not None:
            return self._transform(image.imdecode(img, self._flag), header.label)
        return image.imdecode(img, self._flag), header.label
