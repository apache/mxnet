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

# pylint: skip-file
import mxnet as mx
import mxnet.ndarray as nd
from mxnet.test_utils import *
from mxnet.base import MXNetError
import numpy as np
import os
import tarfile
import sys
from common import assertRaises
import unittest

def test_im2rec():
    if not os.path.isdir("data/test_im2rec/"):
        os.makedirs('data/test_im2rec/')
    tar_filename = download('http://data.mxnet.io/data/test_images.tar.gz',
                            dirname='data/test_im2rec/')
    with tarfile.open(tar_filename, "r:gz") as gf:
        gf.extractall("data/test_im2rec/")
    filenames = next(os.walk("data/test_im2rec/test_images"))[2]
    filenames = [os.path.abspath(os.path.join("data/test_im2rec/test_images",i))
                 for i in filenames]
    # creating list file from test image data
    lst_filename = os.path.abspath("data/test_im2rec/img.lst")
    output_path = os.path.abspath("data/test_im2rec/")
    with open(lst_filename, 'w') as img_lst:
        for i,fpath in enumerate(filenames):
            img_lst.write(str(i) + "\t" + str(i) + "\t" + fpath + "\n")
    
    def test_validate_params():
        try:
            mx.io.im2rec._validate_params("notafile", output_path)
            assert False, "Invalid list file"
        except ValueError:
            assert True
        try:
            mx.io.im2rec._validate_params(lst_filename, "notafolder")
            assert False, "Invalid output folder"
        except ValueError:
            assert True
        try:
            mx.io.im2rec._validate_params(lst_filename, output_path, color=3)
            assert False, "Invalid color value"
        except ValueError:
            assert True
        try:
            mx.io.im2rec._validate_params(lst_filename, output_path, encoding='abc')
            assert False, "Invalid encoding"
        except ValueError:
            assert True
        try:
            mx.io.im2rec._validate_params(lst_filename, output_path, quality='abc')
            assert False, "Quality should be an int value"
        except ValueError:
            assert True

    def test_read_worker():
        try:
            try:
                import Queue as queue
            except ImportError:
                import queue as queue
            q_out = queue.PriorityQueue(5)
            img_path = os.path.abspath("data/test_im2rec/test_images/ILSVRC2012_val_00000007.JPEG")
            data_record = [0, 0, img_path, 0.0]
            mx.io.im2rec._read_worker(q_out, transformer, color=1, quality=95, encoding='.jpg',
                                    pass_through=False, pack_labels=True, exception_counter=0,
                                    data_record=data_record)
            output = q_out.get()
            header, img = mx.recordio.unpack_img(output[2])
            assert img == cv2.imread(img_path, 1)
        except:
            assert False, "Failed to process image"

    def test_im2rec():
        try:
            try:
                import Queue as queue
            except ImportError:
                import queue as queue
            output_path = mx.io.im2rec.im2rec(lst_filename, output_path,
                                              transformations=None, num_workers=mp.cpu_count() - 1,
                                              batch_size=4096, pack_labels=True, color=1,
                                              encoding='.jpg', quality=95, pass_through=False,
                                              error_limit=2)
            rec_file = os.path.join(output_path, 'img.rec')
            idx_file = os.path.join(output_path, 'img.idx')
            read_record = mx.recordio.MXIndexedRecordIO(rec_file, idx_file, 'r')
            for i, file in enumerate(filenames):
                item = read_record.read_idx(i)
                header, img = mx.recordio.unpack_img(item)
                assert img == cv2.imread(file, 1)
        except:
            assert False, "Failed to process image"

if __name__ == "__main__":
    test_im2rec()
