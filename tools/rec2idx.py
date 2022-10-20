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
import time
import ctypes
from mxnet.base import _LIB
from mxnet.base import check_call
import mxnet as mx
import argparse

class IndexCreator(mx.recordio.MXRecordIO):
    """Reads `RecordIO` data format, and creates index file
    that enables random access.

    Example usage:
    ----------
    >>> creator = IndexCreator('data/test.rec','data/test.idx')
    >>> creator.create_index()
    >>> creator.close()
    >>> !ls data/
    test.rec  test.idx

    Parameters
    ----------
    uri : str
        Path to the record file.
    idx_path : str
        Path to the index file, that will be created/overwritten.
    key_type : type
        Data type for keys (optional, default = int).
    """
    def __init__(self, uri, idx_path, key_type=int):
        self.key_type = key_type
        self.fidx = None
        self.idx_path = idx_path
        super(IndexCreator, self).__init__(uri, 'r')

    def open(self):
        super(IndexCreator, self).open()
        self.fidx = open(self.idx_path, 'w')

    def close(self):
        """Closes the record and index files."""
        if not self.is_open:
            return
        super(IndexCreator, self).close()
        self.fidx.close()

    def tell(self):
        """Returns the current position of read head.
        """
        pos = ctypes.c_size_t()
        check_call(_LIB.MXRecordIOReaderTell(self.handle, ctypes.byref(pos)))
        return pos.value

    def create_index(self):
        """Creates the index file from open record file
        """
        self.reset()
        counter = 0
        pre_time = time.time()
        while True:
            if counter % 1000 == 0:
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', counter)
            pos = self.tell()
            cont = self.read()
            if cont is None:
                break
            key = self.key_type(counter)
            self.fidx.write(f'{str(key)}\t{pos}\n')
            counter = counter + 1

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an index file from .rec file')
    parser.add_argument('record', help='path to .rec file.')
    parser.add_argument('index', help='path to index file.')
    args = parser.parse_args()
    args.record = os.path.abspath(args.record)
    args.index = os.path.abspath(args.index)
    return args


if __name__ == '__main__':
    args = parse_args()
    creator = IndexCreator(args.record, args.index)
    creator.create_index()
    creator.close()
