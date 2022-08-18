#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from __future__ import print_function
import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../python"))
import mxnet as mx
import random
import argparse
import cv2
import time
import traceback

try:
    import multiprocessing
except ImportError:
    multiprocessing = None

def list_image(root, recursive, exts):
    """Traverses the root of directory that contains images and
    generates image list iterator.
    Parameters
    ----------
    root: string
    recursive: bool
    exts: string
    Returns
    -------
    image iterator that contains all the image under the specified path
    """

    i = 0
    if recursive:
        cat = {}
        for path, dirs, files in os.walk(root, followlinks=True):
            dirs.sort()
            files.sort()
            for fname in files:
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and (suffix in exts):
                    if path not in cat:
                        cat[path] = len(cat)
                    yield (i, os.path.relpath(fpath, root), cat[path])
                    i += 1
        for k, v in sorted(cat.items(), key=lambda x: x[1]):
            print(os.path.relpath(k, root), v)
    else:
        for fname in sorted(os.listdir(root)):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                yield (i, os.path.relpath(fpath, root), 0)
                i += 1

def write_list(path_out, image_list):
    """Hepler function to write image list into the file.
    The format is as below,
    integer_image_index \t float_label_index \t path_to_image
    Note that the blank between number and tab is only used for readability.
    Parameters
    ----------
    path_out: string
    image_list: list
    """
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            line = f'{item[0]}\t'
            for j in item[2:]:
                line += f'{j}\t'
            line += f'{item[1]}\n'
            fout.write(line)

def make_list(args):
    """Generates .lst file.
    Parameters
    ----------
    args: object that contains all the arguments
    """
    image_list = list_image(args.root, args.recursive, args.exts)
    image_list = list(image_list)
    if args.shuffle is True:
        random.seed(100)
        random.shuffle(image_list)
    N = len(image_list)
    chunk_size = (N + args.chunks - 1) // args.chunks
    for i in range(args.chunks):
        chunk = image_list[i * chunk_size:(i + 1) * chunk_size]
        if args.chunks > 1:
            str_chunk = f'_{i}'
        else:
            str_chunk = ''
        sep = int(chunk_size * args.train_ratio)
        sep_test = int(chunk_size * args.test_ratio)
        if args.train_ratio == 1.0:
            write_list(args.prefix + str_chunk + '.lst', chunk)
        else:
            if args.test_ratio:
                write_list(args.prefix + str_chunk + '_test.lst', chunk[:sep_test])
            if args.train_ratio + args.test_ratio < 1.0:
                write_list(args.prefix + str_chunk + '_val.lst', chunk[sep_test + sep:])
            write_list(args.prefix + str_chunk + '_train.lst', chunk[sep_test:sep_test + sep])

def read_list(path_in):
    """Reads the .lst file and generates corresponding iterator.
    Parameters
    ----------
    path_in: string
    Returns
    -------
    item iterator that contains information in .lst file
    """
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            line_len = len(line)
            # check the data format of .lst file
            if line_len < 3:
                print(f'lst should have at least has three parts, but only has {line_len} parts for {line}')
                continue
            try:
                item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
            except Exception as e:
                print(f'Parsing lst met error for {line}, detail: {e}')
                continue
            yield item

def image_encode(args, i, item, q_out):
    """Reads, preprocesses, packs the image and put it back in output queue.
    Parameters
    ----------
    args: object
    i: int
    item: list
    q_out: queue
    """
    fullpath = os.path.join(args.root, item[1])

    if len(item) > 3 and args.pack_label:
        header = mx.recordio.IRHeader(0, item[2:], item[0], 0)
    else:
        header = mx.recordio.IRHeader(0, item[2], item[0], 0)

    if args.pass_through:
        try:
            with open(fullpath, 'rb') as fin:
                img = fin.read()
            s = mx.recordio.pack(header, img)
            q_out.put((i, s, item))
        except Exception as e:
            traceback.print_exc()
            print('pack_img error:', item[1], e)
            q_out.put((i, None, item))
        return

    try:
        img = cv2.imread(fullpath, args.color)
    except:
        traceback.print_exc()
        print(f'imread error trying to load file: {fullpath} ')
        q_out.put((i, None, item))
        return
    if img is None:
        print(f'imread read blank (None) image for file: {fullpath}')
        q_out.put((i, None, item))
        return
    if args.center_crop:
        if img.shape[0] > img.shape[1]:
            margin = (img.shape[0] - img.shape[1]) // 2
            img = img[margin:margin + img.shape[1], :]
        else:
            margin = (img.shape[1] - img.shape[0]) // 2
            img = img[:, margin:margin + img.shape[0]]
    if args.resize:
        if img.shape[0] > img.shape[1]:
            newsize = (args.resize, img.shape[0] * args.resize // img.shape[1])
        else:
            newsize = (img.shape[1] * args.resize // img.shape[0], args.resize)
        img = cv2.resize(img, newsize)

    try:
        s = mx.recordio.pack_img(header, img, quality=args.quality, img_fmt=args.encoding)
        q_out.put((i, s, item))
    except Exception as e:
        traceback.print_exc()
        print(f'pack_img error on file: {fullpath}', e)
        q_out.put((i, None, item))
        return

def read_worker(args, q_in, q_out):
    """Function that will be spawned to fetch the image
    from the input queue and put it back to output queue.
    Parameters
    ----------
    args: object
    q_in: queue
    q_out: queue
    """
    while True:
        deq = q_in.get()
        if deq is None:
            break
        i, item = deq
        image_encode(args, i, item, q_out)

def write_worker(q_out, fname, working_dir):
    """Function that will be spawned to fetch processed image
    from the output queue and write to the .rec file.
    Parameters
    ----------
    q_out: queue
    fname: string
    working_dir: string
    """
    pre_time = time.time()
    count = 0
    fname = os.path.basename(fname)
    fname_rec = os.path.splitext(fname)[0] + '.rec'
    fname_idx = os.path.splitext(fname)[0] + '.idx'
    record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                           os.path.join(working_dir, fname_rec), 'w')
    buf = {}
    more = True
    while more:
        deq = q_out.get()
        if deq is not None:
            i, s, item = deq
            buf[i] = (s, item)
        else:
            more = False
        while count in buf:
            s, item = buf[count]
            del buf[count]
            if s is not None:
                record.write_idx(item[0], s)

            if count % 1000 == 0:
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', count)
                pre_time = cur_time
            count += 1

def parse_args():
    """Defines all arguments.
    Returns
    -------
    args object that contains all the params
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    parser.add_argument('prefix', help='prefix of input/output lst and rec files.')
    parser.add_argument('root', help='path to folder containing images.')

    cgroup = parser.add_argument_group('Options for creating image lists')
    cgroup.add_argument('--list', action='store_true',
                        help='If this is set im2rec will create image list(s) by traversing root folder\
        and output to <prefix>.lst.\
        Otherwise im2rec will read <prefix>.lst and create a database at <prefix>.rec')
    cgroup.add_argument('--exts', nargs='+', default=['.jpeg', '.jpg', '.png'],
                        help='list of acceptable image extensions.')
    cgroup.add_argument('--chunks', type=int, default=1, help='number of chunks.')
    cgroup.add_argument('--train-ratio', type=float, default=1.0,
                        help='Ratio of images to use for training.')
    cgroup.add_argument('--test-ratio', type=float, default=0,
                        help='Ratio of images to use for testing.')
    cgroup.add_argument('--recursive', action='store_true',
                        help='If true recursively walk through subdirs and assign an unique label\
        to images in each folder. Otherwise only include images in the root folder\
        and give them label 0.')
    cgroup.add_argument('--no-shuffle', dest='shuffle', action='store_false',
                        help='If this is passed, \
        im2rec will not randomize the image order in <prefix>.lst')
    rgroup = parser.add_argument_group('Options for creating database')
    rgroup.add_argument('--pass-through', action='store_true',
                        help='whether to skip transformation and save image as is')
    rgroup.add_argument('--resize', type=int, default=0,
                        help='resize the shorter edge of image to the newsize, original images will\
        be packed by default.')
    rgroup.add_argument('--center-crop', action='store_true',
                        help='specify whether to crop the center image to make it rectangular.')
    rgroup.add_argument('--quality', type=int, default=95,
                        help='JPEG quality for encoding, 1-100; or PNG compression for encoding, 1-9')
    rgroup.add_argument('--num-thread', type=int, default=1,
                        help='number of thread to use for encoding. order of images will be different\
        from the input list if >1. the input list will be modified to match the\
        resulting order.')
    rgroup.add_argument('--color', type=int, default=1, choices=[-1, 0, 1],
                        help='specify the color mode of the loaded image.\
        1: Loads a color image. Any transparency of image will be neglected. It is the default flag.\
        0: Loads image in grayscale mode.\
        -1:Loads image as such including alpha channel.')
    rgroup.add_argument('--encoding', type=str, default='.jpg', choices=['.jpg', '.png'],
                        help='specify the encoding of the images.')
    rgroup.add_argument('--pack-label', action='store_true',
        help='Whether to also pack multi dimensional label in the record file')
    args = parser.parse_args()
    args.prefix = os.path.abspath(args.prefix)
    args.root = os.path.abspath(args.root)
    return args

if __name__ == '__main__':
    args = parse_args()
    # if the '--list' is used, it generates .lst file
    if args.list:
        make_list(args)
    # otherwise read .lst file to generates .rec file
    else:
        if os.path.isdir(args.prefix):
            working_dir = args.prefix
        else:
            working_dir = os.path.dirname(args.prefix)
        files = [os.path.join(working_dir, fname) for fname in os.listdir(working_dir)
                    if os.path.isfile(os.path.join(working_dir, fname))]
        count = 0
        for fname in files:
            if fname.startswith(args.prefix) and fname.endswith('.lst'):
                print('Creating .rec file from', fname, 'in', working_dir)
                count += 1
                image_list = read_list(fname)
                # -- write_record -- #
                if args.num_thread > 1 and multiprocessing is not None:
                    q_in = [multiprocessing.Queue(1024) for i in range(args.num_thread)]
                    q_out = multiprocessing.Queue(1024)
                    # define the process
                    read_process = [multiprocessing.Process(target=read_worker, args=(args, q_in[i], q_out)) \
                                    for i in range(args.num_thread)]
                    # process images with num_thread process
                    for p in read_process:
                        p.start()
                    # only use one process to write .rec to avoid race-condtion
                    write_process = multiprocessing.Process(target=write_worker, args=(q_out, fname, working_dir))
                    write_process.start()
                    # put the image list into input queue
                    for i, item in enumerate(image_list):
                        q_in[i % len(q_in)].put((i, item))
                    for q in q_in:
                        q.put(None)
                    for p in read_process:
                        p.join()

                    q_out.put(None)
                    write_process.join()
                else:
                    print('multiprocessing not available, fall back to single threaded encoding')
                    try:
                        import Queue as queue
                    except ImportError:
                        import queue
                    q_out = queue.Queue()
                    fname = os.path.basename(fname)
                    fname_rec = os.path.splitext(fname)[0] + '.rec'
                    fname_idx = os.path.splitext(fname)[0] + '.idx'
                    record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                                           os.path.join(working_dir, fname_rec), 'w')
                    cnt = 0
                    pre_time = time.time()
                    for i, item in enumerate(image_list):
                        image_encode(args, i, item, q_out)
                        if q_out.empty():
                            continue
                        _, s, _ = q_out.get()
                        record.write_idx(item[0], s)
                        if cnt % 1000 == 0:
                            cur_time = time.time()
                            print('time:', cur_time - pre_time, ' count:', cnt)
                            pre_time = cur_time
                        cnt += 1
        if not count:
            print(f'Did not find and list file with prefix {args.prefix}')
