# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""## Generation of summaries.
### Class for writing Summaries
@@FileWriter
@@FileWriterCache
### Summary Ops
@@tensor_summary
@@scalar
@@histogram
@@audio
@@image
@@merge
@@merge_all
## Utilities
@@get_summary_description
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import io
import wave
import struct
import re as _re
import numpy as np
from .proto.summary_pb2 import Summary
from .proto.summary_pb2 import HistogramProto
from .proto.summary_pb2 import SummaryMetadata
from .proto.tensor_pb2 import TensorProto
from .proto.tensor_shape_pb2 import TensorShapeProto
from .proto.plugin_pr_curve_pb2 import PrCurvePluginData
from .utils import _make_numpy_array
try:
    from PIL import Image
except:
    Image = None

_INVALID_TAG_CHARACTERS = _re.compile(r'[^-/\w\.]')


def _clean_tag(name):
    # In the past, the first argument to summary ops was a tag, which allowed
    # arbitrary characters. Now we are changing the first argument to be the node
    # name. This has a number of advantages (users of summary ops now can
    # take advantage of the tf name scope system) but risks breaking existing
    # usage, because a much smaller set of characters are allowed in node names.
    # This function replaces all illegal characters with _s, and logs a warning.
    # It also strips leading slashes from the name.
    if name is not None:
        new_name = _INVALID_TAG_CHARACTERS.sub('_', name)
        new_name = new_name.lstrip('/')  # Remove leading slashes
        if new_name != name:
            logging.info('Summary name %s is illegal; using %s instead.' % (name, new_name))
            name = new_name
    return name


def scalar_summary(name, scalar):
    """Outputs a `Summary` protocol buffer containing a single scalar value.
    The generated Summary has a Tensor.proto containing the input Tensor.
    Args:
      name: A name for the generated node. Will also serve as the series name in
        TensorBoard.
      tensor: A real numeric Tensor containing a single value.
      collections: Optional list of graph collections keys. The new summary op is
        added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    Returns:
      A scalar `Tensor` of type `string`. Which contains a `Summary` protobuf.
    Raises:
      ValueError: If tensor has the wrong shape or type.
    """
    name = _clean_tag(name)
    scalar = _make_numpy_array(scalar)
    assert(scalar.squeeze().ndim == 0), 'scalar should be 0D'
    scalar = float(scalar)
    return Summary(value=[Summary.Value(tag=name, simple_value=scalar)])


def histogram_summary(name, values, bins):
    # pylint: disable=line-too-long
    """Outputs a `Summary` protocol buffer with a histogram.
    The generated
    [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
    has one summary value containing a histogram for `values`.
    This op reports an `InvalidArgument` error if any value is not finite.
    Args:
      name: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      values: A real numeric `Tensor`. Any shape. Values to use to
        build the histogram.
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    name = _clean_tag(name)
    values = _make_numpy_array(values)
    hist = make_histogram(values.astype(float), bins)
    return Summary(value=[Summary.Value(tag=name, histo=hist)])


def make_histogram(values, bins):
    """Convert values into a histogram proto using logic from histogram.cc."""
    values = values.reshape(-1)
    counts, limits = np.histogram(values, bins=bins)
    limits = limits[1:]

    sum_sq = values.dot(values)
    return HistogramProto(min=values.min(),
                          max=values.max(),
                          num=len(values),
                          sum=values.sum(),
                          sum_squares=sum_sq,
                          bucket_limit=limits,
                          bucket=counts)


def image_summary(tag, tensor):
    """Outputs a `Summary` protocol buffer with images.
    The summary has up to `max_images` summary values containing images. The
    images are built from `tensor` which must be 3-D with shape `[height, width,
    channels]` and where `channels` can be:
    *  1: `tensor` is interpreted as Grayscale.
    *  3: `tensor` is interpreted as RGB.
    *  4: `tensor` is interpreted as RGBA.
    The `name` in the outputted Summary.Value protobufs is generated based on the
    name, with a suffix depending on the max_outputs setting:
    *  If `max_outputs` is 1, the summary value tag is '*name*/image'.
    *  If `max_outputs` is greater than 1, the summary value tags are
       generated sequentially as '*name*/image/0', '*name*/image/1', etc.
    Args:
      tag: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      tensor: A 3-D `uint8` or `float32` `Tensor` of shape `[height, width,
        channels]` where `channels` is 1, 3, or 4.
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    tag = _clean_tag(tag)
    tensor = _make_numpy_array(tensor, 'IMG')
    tensor = tensor.astype(np.float32)
    tensor = (tensor * 255).astype(np.uint8)
    image = make_image(tensor)
    return Summary(value=[Summary.Value(tag=tag, image=image)])


def make_image(tensor):
    """Convert an numpy representation image to Image protobuf"""
    if Image is None:
        raise ImportError('need to install PIL for visualizing images')
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)


def audio_summary(tag, tensor, sample_rate=44100):
    tensor = _make_numpy_array(tensor)
    tensor = tensor.squeeze()
    assert(tensor.ndim == 1), 'input tensor should be 1 dimensional.'

    tensor_list = [int(32767.0 * x) for x in tensor]
    fio = io.BytesIO()
    wave_writer = wave.open(fio, 'wb')
    wave_writer.setnchannels(1)
    wave_writer.setsampwidth(2)
    wave_writer.setframerate(sample_rate)
    tensor_enc = b''
    for v in tensor_list:
        tensor_enc += struct.pack('<h', v)

    wave_writer.writeframes(tensor_enc)
    wave_writer.close()
    audio_string = fio.getvalue()
    fio.close()
    audio = Summary.Audio(sample_rate=sample_rate,
                          num_channels=1,
                          length_frames=len(tensor_list),
                          encoded_audio_string=audio_string,
                          content_type='audio/wav')
    return Summary(value=[Summary.Value(tag=tag, audio=audio)])


def text_summary(tag, text):
    plugin_data = [SummaryMetadata.PluginData(plugin_name='text')]
    smd = SummaryMetadata(plugin_data=plugin_data)
    tensor = TensorProto(dtype='DT_STRING',
                         string_val=[text.encode(encoding='utf_8')],
                         tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]))
    return Summary(value=[Summary.Value(node_name=tag, metadata=smd, tensor=tensor)])


def pr_curve_summary(tag, labels, predictions, num_thresholds=127, weights=None):
    if num_thresholds > 127:  # strange, value > 127 breaks protobuf
        num_thresholds = 127
    data = compute_curve(labels, predictions, num_thresholds=num_thresholds, weights=weights)
    pr_curve_plugin_data = PrCurvePluginData(version=0, num_thresholds=num_thresholds).SerializeToString()
    plugin_data = [SummaryMetadata.PluginData(plugin_name='pr_curves', content=pr_curve_plugin_data)]
    smd = SummaryMetadata(plugin_data=plugin_data)
    tensor = TensorProto(dtype='DT_FLOAT',
                         float_val=data.reshape(-1).tolist(),
                         tensor_shape=TensorShapeProto(
                             dim=[TensorShapeProto.Dim(size=data.shape[0]), TensorShapeProto.Dim(size=data.shape[1])]))
    return Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor)])


# https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/summary.py
def compute_curve(labels, predictions, num_thresholds=None, weights=None):
    _MINIMUM_COUNT = 1e-7

    if weights is None:
        weights = 1.0

    # Compute bins of true positives and false positives.
    bucket_indices = np.int32(np.floor(predictions * (num_thresholds - 1)))
    float_labels = labels.astype(np.float)
    histogram_range = (0, num_thresholds - 1)
    tp_buckets, _ = np.histogram(
        bucket_indices,
        bins=num_thresholds,
        range=histogram_range,
        weights=float_labels * weights)
    fp_buckets, _ = np.histogram(
        bucket_indices,
        bins=num_thresholds,
        range=histogram_range,
        weights=(1.0 - float_labels) * weights)

    # Obtain the reverse cumulative sum.
    tp = np.cumsum(tp_buckets[::-1])[::-1]
    fp = np.cumsum(fp_buckets[::-1])[::-1]
    tn = fp[0] - fp
    fn = tp[0] - tp
    precision = tp / np.maximum(_MINIMUM_COUNT, tp + fp)
    recall = tp / np.maximum(_MINIMUM_COUNT, tp + fn)
    return np.stack((tp, fp, tn, fn, precision, recall))
