# -*- coding: utf-8 -*-
import mxnet.ndarray as nd
from mxnet.base import numeric_types
from mxnet.recordio import RecordIOType
from recordio_pb2 import RecordHead, RecordData, RecordUnit, ExtraData
try:
    import cv2
except ImportError:
    cv2 = None

def pack_head(id, reserve=0, version=1.0):
    """ pack a head info into RecordHead format

    Parameters
    ----------
    id : int
        record id.
    reserve : int
        record reserve info
    version : float
        version of data structure

    Returns
    -------
    RecordHead
        The head format in recordio.proto.
    """

    head = RecordHead()
    head.id = id
    head.reserve = reserve
    head.version = version
    return head

def pack_ndarray_data(id, data):
    """ pack a NDArray into RecordData

    Parameters
    ----------
    id : int
        record id.
    data : NDArray
        ndarray data.

    Returns
    -------
    RecordData
        The data format in recordio.proto.
    """
    rec_data = RecordData()
    rec_data.id = id
    rec_data.type = RecordIOType.NDARRAY
    rec_data.value = nd.save_to_str(data)
    return rec_data

def pack_string_data(id, data, is_binary=True):
    """ pack a string data info into RecordData

    Parameters
    ----------
    id : int
        record id.
    data : str
        string data.
    is_binary : bool
        whether save the data as binary,
        NDArray format data will read in RecordIter when using binary,

    Returns
    -------
    RecordData
        The data format in recordio.proto.
    """

    rec_data = RecordData()
    rec_data.id = id
    rec_data.type = RecordIOType.BINARY if is_binary else RecordIOType.STRING
    rec_data.value = data
    return rec_data

def pack_ndarray_extra(key, data):
    """ pack a NDArray into RecordData

    Parameters
    ----------
    id : int
        record id.
    data : NDArray
        ndarray data.

    Returns
    -------
    RecordData
        The data format in recordio.proto.
    """
    rec_data = ExtraData()
    rec_data.key = key
    rec_data.type = RecordIOType.NDARRAY
    rec_data.value = nd.save_to_str(data)
    return rec_data

def pack_string_extra(key, data, is_binary=True):
    """ pack a string extra data info into RecordData

    Parameters
    ----------
    id : int
        record id.
    data : str
        string data.
    is_binary : bool
        whether save the data as binary,
        NDArray format data will read in RecordIter when using binary,

    Returns
    -------
    RecordData
        The data format in recordio.proto.
    """


    rec_data = ExtraData()
    rec_data.key = key
    rec_data.type = RecordIOType.BINARY if is_binary else RecordIOType.STRING
    rec_data.value = data
    return rec_data

def pack(header, data, label, extra=[]):
    """ pack a data into string for MXRecordIO

    Parameters
    ----------
    header : RecordHead
        Header of the image record.
    data : RecordData or list of RecordData
        record data list
    label : float or list of float
        label list
    extra : ExtraData or list of ExtraData

    Returns
    -------
    s : str
        The packed string.

    Examples
    --------
    >>> label = 4 # label can also be a 1-D array, for example: label = [1,2,3]
    >>> id = 2574
    >>> header = recordio_pb_pack.pack_head(id)
    >>> with open(path, 'r') as file:
        ...     s = file.read()
    >>> packed_s = recordio_pb_pack.pack(header, s, label)
    """
    if isinstance(data, list):
        for d in data:
            if not isinstance(d, RecordData):
                raise TypeError('type %s not supported' % str(type(d)))
    elif isinstance(data, RecordData):
        data = [data]
    else:
        raise TypeError('type %s not supported' % str(type(data)))

    if isinstance(label, list):
        for l in label:
            if not isinstance(l, numeric_types):
                raise TypeError('type %s not supported' % str(type(l)))
    elif isinstance(label, numeric_types):
        label = [label]
    else:
        raise TypeError('type %s not supported' % str(type(label)))

    if isinstance(extra, list):
        for e in extra:
            if not isinstance(e, ExtraData):
                raise TypeError('type %s not supported' % str(type(e)))
    elif isinstance(extra, ExtraData):
        extra = [extra]
    else:
        raise TypeError('type %s not supported' % str(type(extra)))

    rec_unit = RecordUnit()
    rec_unit.head.CopyFrom(header)

    # body
    rec_body = rec_unit.body
    for d in data:
        rec_data = rec_body.data.add()
        rec_data.CopyFrom(d)
    for l in label:
        rec_body.label.append(l)
    for e in extra:
        ed = rec_body.extra.add()
        ed.CopyFrom(e)
    return rec_unit.SerializeToString()

def pack_img_data(id, img, quality=95, img_fmt='.jpg'):
    """Pack an image into string.

    Parameters
    ----------
    id : uint64
        id of the image record.
    img : numpy.ndarray
        Image to be packed.
    quality : int
        Quality for JPEG encoding in range 1-100, or compression for PNG encoding in range 1-9.
    img_fmt : str
        Encoding of the image (.jpg for JPEG, .png for PNG).

    Returns
    -------
    s : str
        The packed string.

    Examples
    --------
    >>> id = 2574
    >>> label = 4 # label can also be a 1-D array, for example: label = [1,2,3]
    >>> header = recordio_pb_pack.pack_head(id)
    >>> img = cv2.imread('test.jpg')
    >>> packed_img = recordio_pb_pack.pack_img_data(id, img)
    >>> packed_s = recordio_pb_pack.pack(header, packed_img, label)
    """
    assert cv2 is not None
    jpg_formats = ['.JPG', '.JPEG']
    png_formats = ['.PNG']
    encode_params = None
    if img_fmt.upper() in jpg_formats:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif img_fmt.upper() in png_formats:
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, quality]

    ret, buf = cv2.imencode(img_fmt, img, encode_params)
    assert ret, 'failed to encode image'
    return pack_string_data(id, buf.tostring())
