# pylint: skip-file
import mxnet as mx
import tempfile
import random
from builtins import bytes

def test_recordio():
    frec = tempfile.mktemp()
    N = 10

    writer = mx.recordio.MXRecordIO(frec, 'w')
    for i in range(N):
        writer.write(str(i))
    del writer

    reader = mx.recordio.MXRecordIO(frec, 'r')
    for i in range(N):
        res = reader.read()
        assert res == bytes(str(i), 'utf-8')


def test_indexed_recordio():
    fidx = tempfile.mktemp()
    frec = tempfile.mktemp()
    N = 10

    writer = mx.recordio.MXIndexedRecordIO(fidx, frec, 'w')
    for i in range(N):
        writer.write_idx(i, str(i))
    del writer

    reader = mx.recordio.MXIndexedRecordIO(fidx, frec, 'r')
    keys = reader.keys()
    assert sorted(keys) == [i for i in range(N)]
    random.shuffle(keys)
    for k in keys:
        res = reader.read_idx(k)
        assert res == bytes(str(k), 'utf-8')

if __name__ == '__main__':
    test_recordio()
    test_indexed_recordio()