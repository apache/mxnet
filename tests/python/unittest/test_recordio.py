# pylint: skip-file
import sys
import mxnet as mx
import tempfile
import random

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
    keys = reader.keys()
    assert sorted(keys) == [i for i in range(N)]
    random.shuffle(keys)
    for i in keys:
        res = reader.read_idx(i)
        if sys.version_info[0] < 3:
            assert res == str(chr(i))
        else:
            assert res == bytes(str(chr(i)), 'utf-8')

if __name__ == '__main__':
    test_recordio()
    test_indexed_recordio()