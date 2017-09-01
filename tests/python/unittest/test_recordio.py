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
import sys
import mxnet as mx
import numpy as np
import tempfile
import random
import string

def test_recordio():
    frec = tempfile.mktemp()
    N = 255

    writer = mx.recordio.MXRecordIO(frec, 'w')
    for i in range(N):
        if sys.version_info[0] < 3:
            writer.write(str(chr(i)))
        else:
            writer.write(bytes(str(chr(i)), 'utf-8'))
    del writer

    reader = mx.recordio.MXRecordIO(frec, 'r')
    for i in range(N):
        res = reader.read()
        if sys.version_info[0] < 3:
            assert res == str(chr(i))
        else:
            assert res == bytes(str(chr(i)), 'utf-8')

def test_indexed_recordio():
    fidx = tempfile.mktemp()
    frec = tempfile.mktemp()
    N = 255

    writer = mx.recordio.MXIndexedRecordIO(fidx, frec, 'w')
    for i in range(N):
        if sys.version_info[0] < 3:
            writer.write_idx(i, str(chr(i)))
        else:
            writer.write_idx(i, bytes(str(chr(i)), 'utf-8'))
    del writer

    reader = mx.recordio.MXIndexedRecordIO(fidx, frec, 'r')
    keys = reader.keys
    assert sorted(keys) == [i for i in range(N)]
    random.shuffle(keys)
    for i in keys:
        res = reader.read_idx(i)
        if sys.version_info[0] < 3:
            assert res == str(chr(i))
        else:
            assert res == bytes(str(chr(i)), 'utf-8')

def test_recordio_pack_label():
    frec = tempfile.mktemp()
    N = 255

    for i in range(1, N):
        for j in range(N):
            content = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(j))
            content = content.encode('utf-8')
            label = np.random.uniform(size=i).astype(np.float32)
            header = (0, label, 0, 0)
            s = mx.recordio.pack(header, content)
            rheader, rcontent = mx.recordio.unpack(s)
            assert (label == rheader.label).all()
            assert content == rcontent

if __name__ == '__main__':
    test_recordio_pack_label()
    test_recordio()
    test_indexed_recordio()
