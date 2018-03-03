"""
To write tf_record into file. Here we use it for tensorboard's event writting.
The code was borrowed from https://github.com/TeamHG-Memex/tensorboard_logger
"""

import struct
from .crc32c import crc32c


class RecordWriter(object):
    """Write records in the following format for a single record event_str:
    uint64 len(event_str)
    uint32 masked crc of len(event_str)
    byte event_str
    uint32 masked crc of event_str
    The implementation is ported from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/io/record_writer.cc
    Here we simply define a byte string _dest to buffer the record to be written to files.
    The flush and close mechanism is totally controlled in this class.
    In TensorFlow, _dest is a object instance of ZlibOutputBuffer (C++) which has its own flush
    and close mechanism defined."""
    def __init__(self, path):
        self._dest = None
        self._writer = None
        try:
            self._writer = open(path, 'wb')
        except (OSError, IOError) as e:
            raise ValueError('Failed to open file {}: {}'.format(path, str(e)))

    def __del__(self):
        self.close()

    def write_record(self, event_str):
        header = struct.pack('Q', len(event_str))
        header += struct.pack('I', masked_crc32c(header))
        footer = struct.pack('I', masked_crc32c(event_str))
        if self._dest is None:
            self._dest = header + event_str + footer
        else:
            self._dest += header + event_str + footer

    def flush(self):
        assert self._writer is not None
        if self._dest is not None:
            self._writer.write(self._dest)
            self._dest = None
        self._writer.flush()

    def close(self):
        if self._writer is not None:
            self.flush()
            self._writer.close()
            self._writer = None


def masked_crc32c(data):
    x = u32(crc32c(data))
    return u32(((x >> 15) | u32(x << 17)) + 0xa282ead8)


def u32(x):
    return x & 0xffffffff
