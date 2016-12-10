"""
To write tf_record into file. Here we use it for tensorboard's event writting.
The code was borrow from https://github.com/TeamHG-Memex/tensorboard_logger
"""

import re
import struct

from .crc32c import crc32c

_VALID_OP_NAME_START = re.compile('^[A-Za-z0-9.]')
_VALID_OP_NAME_PART = re.compile('[A-Za-z0-9_.\\-/]+')


class RecordWriter(object):
    def __init__(self, path, flush_secs=2):
        self._name_to_tf_name = {}
        self._tf_names = set()
        self.path = path
        self.flush_secs = flush_secs  # TODO. flush every flush_secs, not every time.
        self._writer = None
        self._writer = open(path, 'wb')

    def write(self, event_str):
        w = self._writer.write
        header = struct.pack('Q', len(event_str))
        w(header)
        w(struct.pack('I', masked_crc32c(header)))
        w(event_str)
        w(struct.pack('I', masked_crc32c(event_str)))
        self._writer.flush()

    def __del__(self):
        if self._writer is not None:
            self._writer.close()


def masked_crc32c(data):
    x = u32(crc32c(data))
    return u32(((x >> 15) | u32(x << 17)) + 0xa282ead8)


def u32(x):
    return x & 0xffffffff


def make_valid_tf_name(name):
    if not _VALID_OP_NAME_START.match(name):
        # Must make it valid somehow, but don't want to remove stuff
        name = '.' + name
    return '_'.join(_VALID_OP_NAME_PART.findall(name))

