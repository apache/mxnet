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

"""convert image dataset to recordio."""
from collections import deque

import os
import logging
import itertools
import multiprocessing as mp
from multiprocessing.managers import SyncManager
from functools import partial

from ..ndarray import array
from .. import recordio

try:
    import Queue as queue
except ImportError:
    import queue as queue
try:
    import cv2
except ImportError:
    cv2 = None

class SharedObjectManager(SyncManager):
    """
    shared object manager
    """
    pass
SharedObjectManager.register("PriorityQueue", queue.PriorityQueue)

def _read_list(list_file, batch_size):
    """
    Helper function that reads the .lst file, binds it in
    a generator and returns a batched version of the generator.
    Parameters
    ----------
    list_file: str
    input list file.
    batch_size: int
    batch size of the generator
    Returns
    -------
    item iterator
    iterator that contains information in .lst file
    """
    with open(list_file, 'r') as input_file:
        while True:
            fetch_data = list(itertools.islice(input_file, batch_size))
            if not fetch_data:
                break
            batch = []
            for line in fetch_data:
                line = [i.strip() for i in line.strip().split('\t')]
                line_len = len(line)
                # check the data format of .lst file
                if line_len < 3:
                    logging.info("lst should have at least has three parts, \
                        but only has %s parts for %s", line_len, line)
                    continue
                try:
                    item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
                    batch.append(item)
                except IOError:
                    logging.info('Parsing lst met error for %s : ', line)
                    continue
            yield batch

def _read_worker(q_out, transformer, color, quality, encoding, pass_through, data_record):
    """
    Helper function that will be run by the read workers
    to fetch the image from the input queue apply
    transformations and put it into output priority queue.
    Parameters
    ----------
    q_out : priority queue
        priority queue
    transformer : transformer object
        transformer object
    color : int
        color
    quality : int
        quality
    encoding : str
        encoding
    pass_through: bool
        skip encoding while packing the image
    data_record : tuple
        image instance to work on.
    """
    i, item = data_record
    img_path = item[1]
    try:
        # construct the header of the record
        header = recordio.IRHeader(0, item[2:], item[0], 0)
        if pass_through:
            with open(img_path, 'rb') as f_im:
                img = f_im.read()
            packed_image = recordio.pack(header, img)
            q_out.put((i, packed_image, item))
        else:
            img = cv2.imread(img_path, color)
            if img is None:
                logging.info('Read a blank image for the file: %s', img_path)
                return
            img = transformer(array(img))
            packed_image = recordio.pack_img(header, img, quality=quality, img_fmt=encoding)
            q_out.put((i, packed_image, item))
    except IOError:
        logging.info('pack_img error on file: %s', img_path)
        return
    except AttributeError:
        logging.info("Using this API requires OpenCV. Unable to load cv2.")

def _validate_filenames(list_file, output_path):
    """
    Helper function to validate the file paths of
    the input list file and output .rec file path.
    Parameters
    --------
    list_file: input list file path
    output_path: path to the output directory
    """
    if not os.path.isfile(list_file):
        raise Exception("Input list file is invalid - \
            1. Wrong filename or file path \n2. List file should be of format *.lst")
    if not os.path.isdir(output_path):
        raise Exception("Output path should be a directory where the \
            rec files will be stored.")

def _count_elem(iter):
    """
    Helper function to count the number of elements in
    a generator.
    Parameters
    -----
    iter: generator object
    Returns
    -----
    count: total count of elements
    """
    cnt = itertools.count()
    deque(zip(iter, cnt), 0)
    return next(cnt)

def im2rec(list_file, transformer, output_path, num_workers=mp.cpu_count() - 1,
           batch_size=4096, color=1, encoding='.jpg', quality=95,
           pass_through=False):
    """
    API to convert the input image dataset into recordIO file format.

    Parameters
    ----------
    list_file : str
        List file name
    transformer : Transforms object
        Transforms object
    output_path : str
        output file path
    num_workers : int
        number of workers
    batch_size : int
        number of images to process in one batch.
    color : int
        color in which to load an image
        1 : color
        0 : grey scale
    encoding : str
        type of encoding for packing the image
    quality : int
        quality of the image being packed
    pass_through : bool
        should the image be packed without any encoding

    Returns
    -------
    output_path : str
        output path of the rec file

    """
    _validate_filenames(list_file, output_path)
    fname = os.path.basename(list_file).split('.')[0]
    shared_obj_mgr = SharedObjectManager()
    shared_obj_mgr.start()

    data_batch_iter = _read_list(list_file, batch_size)
    # A process-safe PriorityQueue
    out_q = shared_obj_mgr.PriorityQueue(batch_size)
    pool = mp.Pool(num_workers)

    for data_batch in data_batch_iter:
        pool.map(partial(_read_worker, out_q, transformer,
                         color, quality, encoding, pass_through), data_batch)
        out_rec = os.path.join(output_path, fname) + '.rec'
        out_idx = os.path.join(output_path, fname) + '.idx'
        record = recordio.MXIndexedRecordIO(os.path.join(out_idx), \
                    os.path.join(out_rec), 'w')
        while not out_q.empty():
            deq_item = out_q.get()
            _, buf, item = deq_item
            record.write_idx(item[0], buf)
        record.close()
        logging.info("Finished processing %s images and packing them", str(batch_size))
