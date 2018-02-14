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
import numpy as np
from dataset.imdb import Imdb
from dataset.pycocotools.coco import COCO


class Coco(Imdb):
    """
    Implementation of Imdb for MSCOCO dataset: https://http://mscoco.org

    Parameters:
    ----------
    anno_file : str
        annotation file for coco, a json file
    image_dir : str
        image directory for coco images
    shuffle : bool
        whether initially shuffle image list

    """
    def __init__(self, anno_file, image_dir, shuffle=True, names='mscoco.names'):
        assert os.path.isfile(anno_file), "Invalid annotation file: " + anno_file
        basename = os.path.splitext(os.path.basename(anno_file))[0]
        super(Coco, self).__init__('coco_' + basename)
        self.image_dir = image_dir

        self.classes = self._load_class_names(names,
            os.path.join(os.path.dirname(__file__), 'names'))

        self.num_classes = len(self.classes)
        self._load_all(anno_file, shuffle)
        self.num_images = len(self.image_set_index)


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
        image_file = os.path.join(self.image_dir, 'images', name)
        assert os.path.isfile(image_file), 'Path does not exist: {}'.format(image_file)
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
        return self.labels[index]

    def _load_all(self, anno_file, shuffle):
        """
        initialize all entries given annotation json file

        Parameters:
        ----------
        anno_file: str
            annotation json file
        shuffle: bool
            whether to shuffle image list
        """
        image_set_index = []
        labels = []
        coco = COCO(anno_file)
        img_ids = coco.getImgIds()
        for img_id in img_ids:
            # filename
            image_info = coco.loadImgs(img_id)[0]
            filename = image_info["file_name"]
            subdir = filename.split('_')[1]
            height = image_info["height"]
            width = image_info["width"]
            # label
            anno_ids = coco.getAnnIds(imgIds=img_id)
            annos = coco.loadAnns(anno_ids)
            label = []
            for anno in annos:
                cat_id = int(anno["category_id"])
                bbox = anno["bbox"]
                assert len(bbox) == 4
                xmin = float(bbox[0]) / width
                ymin = float(bbox[1]) / height
                xmax = xmin + float(bbox[2]) / width
                ymax = ymin + float(bbox[3]) / height
                label.append([cat_id, xmin, ymin, xmax, ymax, 0])
            if label:
                labels.append(np.array(label))
                image_set_index.append(os.path.join(subdir, filename))

        if shuffle:
            import random
            indices = range(len(image_set_index))
            random.shuffle(indices)
            image_set_index = [image_set_index[i] for i in indices]
            labels = [labels[i] for i in indices]
        # store the results
        self.image_set_index = image_set_index
        self.labels = labels
