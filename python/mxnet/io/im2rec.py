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
            for i, line in enumerate(fetch_data):
                line = [i.strip() for i in line.strip().split('\t')]
                line_len = len(line)
                # check the data format of .lst file
                if line_len < 3:
                    logging.info("lst should have at least has three parts, \
                        but only has %s parts for %s", line_len, line)
                    continue
                try:
                    # format of a line in the list file
                    # integer_image_index \t float_label_index \t path_to_image
                    item = [i] + [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
                    batch.append(item)
                except IOError:
                    logging.info('Parsing lst met error for %s : ', line)
                    continue
            yield batch

def _read_worker(q_out, transformations, color, quality, encoding,
                 pass_through, pack_labels, exception_counter,
                 data_record):
    """
    Helper function that will be run by the read workers
    to fetch the image from the input queue apply
    transformations and put it into output priority queue.
    Parameters
    ----------
    q_out : priority queue
        priority queue where processed images are stored
    transformations : transformer object
        transformer object
    color : int, default 1
        specify the color mode of the loaded image.
        1: Loads a color image
        0: Loads image in grayscale mode.
        -1: Loads image unchanged, including alpha channel.
    quality : int, default 95
        JPEG quality for encoding, the range is 1-100;
        or PNG compression for encoding, the range is 1-9
    encoding : str, default '.jpg'
        type of encoding for the images.
        it can be one of ['.jpg', '.png']
    pass_through : bool, default False
        should the image be packed without any transformation.
    pack_labels: bool
        should labels be packed with the image
    exception_counter: int
        process safe counter of exceptions swallowed so far.
    data_record : tuple
        image instance to work on.
    """
    order_num, i, data_item = data_record
    img_path = data_item[1]
    try:
        # construct the header of the record
        if pack_labels:
            header = recordio.IRHeader(0, data_item[2:], data_item[0], 0)
        else:
            header = recordio.IRHeader(0, data_item[0], 0)
        if pass_through:
            with open(img_path, 'rb') as img_file:
                img = img_file.read()
            packed_image = recordio.pack(header, img)
            q_out.put((order_num, i, packed_image, data_item))
        else:
            img = cv2.imread(img_path, color)
            if img is None:
                logging.info('Read a blank image for the file: %s', img_path)
                with exception_counter.get_lock():
                    exception_counter.value += 1
                return
            if transformations:
                img = transformations(array(img))
            packed_image = recordio.pack_img(header, img, quality=quality,
                                             img_fmt=encoding)
            q_out.put((order_num, i, packed_image, data_item))
    except IOError:
        logging.info('Unable to pack image for file: %s', img_path)
        with exception_counter.get_lock():
            exception_counter.value += 1
    except: # pylint: disable=bare-except
        logging.info('Transforms failed for the image: %s', img_path)
        with exception_counter.get_lock():
            exception_counter.value += 1

def _validate_params(list_file, output_path, pack_labels, color,
                     encoding, quality, pass_through):
    """
    Helper function to validate the file paths of
    the input list file and output .rec file path.
    Parameters
    """
    if not os.path.isfile(list_file):
        raise ValueError("Input list file is invalid - \n"
                         + "1. Wrong filename or file path \n"
                         + "2. List file should be of format *.lst")
    if not os.path.isdir(output_path):
        raise ValueError("Output path should be a directory where the "
                         + "rec files will be stored.")
    if not pass_through and cv2 is None:
        raise ValueError("This API usage requires OpenCV to be installed.")
    if color not in [1, 0, -1]:
        raise ValueError("Invalid value for color parameter. "
                         + "Should be 1, 0 or -1")
    if encoding not in ['.jpg', '.png']:
        raise ValueError("Encoding should be either .jpg or .png")
    if not isinstance(quality, int):
        raise ValueError("Quality should be an integer value. "
                         + "range for quality of JPEG encoding is 1-100, "
                         + "and range for PNG compression for encoding is 1-9")
    if not pack_labels:
        logging.info("pack_labels is set as False. We will create a copy of the"
                    + " list file to align with the created rec file.")

def im2rec(list_file, output_path, transformations=None, num_workers=mp.cpu_count() - 1,
           batch_size=4096, pack_labels=True, color=1, encoding='.jpg',
           quality=95, pass_through=False, error_limit=0):
    """
    API to convert the input image dataset into recordIO file format.

    Parameters
    ----------
    list_file : str
        List file name
    output_path : str
        output file path
    transformations : Transforms object
        Transforms object
    num_workers : int. default (number of cpu cores - 1)
        number of workers, ideally do not set this value to a number
        greater than the number of cpu cores in your machine.
    batch_size : int, default 4096
        number of images to process in one batch.
    pack_labels : bool, default True
        boolean to determine if multi-dimensional labels should be packed.
    color : int, default 1
        specify the color mode of the loaded image.
        1: Loads a color image
        0: Loads image in grayscale mode.
        -1: Loads image unchanged, including alpha channel.
    encoding : str, default '.jpg'
        type of encoding for the images.
        it can be one of ['.jpg', '.png']
    quality : int, default 95
        JPEG quality for encoding, the range is 1-100;
        or PNG compression for encoding, the range is 1-9
    pass_through : bool, default False
        should the image be packed without any transformation.
    error_limit: int, default 0
        While creating the dataset there might be errors due
        to corrupt images and inability to pack them in recordIO
        format. This parameter decides how many such errors can
        be swallowed while generating the rec file.

    Returns
    -------
    output_path : str
        output path of the rec file

    """
    _validate_params(list_file, output_path, pack_labels, color,
                     encoding, quality, pass_through)
    fname = os.path.basename(list_file).split('.')[0]
    shared_obj_mgr = SharedObjectManager()
    shared_obj_mgr.start()

    data_batch_iter = _read_list(list_file, batch_size)
    # A process-safe PriorityQueue
    out_q = shared_obj_mgr.PriorityQueue(batch_size)
    pool = mp.Pool(num_workers)
    # Process safe counter to track number of exceptions occured
    exception_counter = mp.Value('i', 0)
    out_rec = os.path.join(output_path, fname) + '.rec'
    out_idx = os.path.join(output_path, fname) + '.idx'
    record = recordio.MXIndexedRecordIO(os.path.join(out_idx),
                                        os.path.join(out_rec), 'w')
    if not pack_labels:
        lst_file_copy = os.path.join(output_path, fname) + '-copy.lst'
        lst_fhandle = open(lst_file_copy, 'w')
    for data_batch in data_batch_iter:
        pool.map(partial(_read_worker, out_q, transformations,
                         color, quality, encoding, pass_through,
                         pack_labels, exception_counter), data_batch)
        if exception_counter >= error_limit:
            logging.error("Number of images failed while converting to recordIO "
                          + "crossed error limit.")
            return None
        while not out_q.empty():
            deq_item = out_q.get()
            _, _, buf, item = deq_item
            if not pack_labels:
                labels = '\t'.join([str(i) for i in item[2:]])
                lst_fhandle.write(item[0] + '\t' + labels +
                                  '\t' + item[1] + '\n')
            record.write_idx(item[0], buf)
        logging.info("Finished processing %s images and packing them", str(batch_size))
    record.close()
    if pack_labels:
        logging.info("The recordIO files and their corresponding index files "
                     + "can be found here -\n%s \n%s", out_rec, out_idx)
    else:
        logging.info("The recordIO files and their corresponding index and list files "
                     + "can be found here -\n%s \n%s \n%s", out_rec, out_idx, lst_file_copy)
    return output_path
