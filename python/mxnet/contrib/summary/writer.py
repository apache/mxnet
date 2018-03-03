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
"""Provides an API for generating Event protocol buffers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import json
import os
import logging
from .proto import event_pb2
from .proto import summary_pb2
from .event_file_writer import EventFileWriter
from .summary import scalar_summary, histogram_summary, image_summary, audio_summary, text_summary, pr_curve_summary
from .graph import graph
from .graph_onnx import gg
from .utils import _save_ndarray, _make_sprite_image, _make_tsv, _add_embedding_info, _make_numpy_array
from ...ndarray import NDArray


class SummaryToEventTransformer(object):
    """Abstractly implements the SummaryWriter API.
    This API basically implements a number of endpoints (add_summary,
    add_session_log, etc). The endpoints all generate an event protobuf, which is
    passed to the contained _event_writer.
    @@__init__
    @@add_summary
    @@add_session_log
    @@add_graph
    @@add_meta_graph
    @@add_run_metadata
    """
    # TODO(junwu): The original interface has two more arguments: graph and graph_def.
    # Need to check its compatibility with using ONNX for visualizing MXNet graphs.
    def __init__(self, event_writer, graph=None, graph_def=None):
        """Creates a `SummaryWriter` and an event file.
        On construction the summary writer creates a new event file in `logdir`.
        This event file will contain `Event` protocol buffers constructed when you
        call one of the following functions: `add_summary()`, `add_session_log()`,
        `add_event()`, or `add_graph()`.
        If you pass a `Graph` to the constructor it is added to
        the event file. (This is equivalent to calling `add_graph()` later).
        TensorBoard will pick the graph from the file and display it graphically so
        you can interactively explore the graph you built. You will usually pass
        the graph from the session in which you launched it:
        ```python
        ...create a graph...
        # Launch the graph in a session.
        sess = tf.Session()
        # Create a summary writer, add the 'graph' to the event file.
        writer = tf.summary.FileWriter(<some-directory>, sess.graph)
        ```
        Args:
          event_writer: An EventWriter. Implements add_event method.
          graph: A `Graph` object, such as `sess.graph`.
          graph_def: DEPRECATED: Use the `graph` argument instead.
        Adapted from
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/writer.py#L40
        """
        self._event_writer = event_writer

    def add_summary(self, summary, global_step=None):
        """Adds a `Summary` protocol buffer to the event file.
        This method wraps the provided summary in an `Event` protocol buffer
        and adds it to the event file.
        You can pass the result of evaluating any summary op, using
        [`Session.run()`](client.md#Session.run) or
        [`Tensor.eval()`](framework.md#Tensor.eval), to this
        function. Alternatively, you can pass a `tf.Summary` protocol
        buffer that you populate with your own data. The latter is
        commonly done to report evaluation results in event files.
        Args:
          summary: A `Summary` protocol buffer, optionally serialized as a string.
          global_step: Number. Optional global step value to record with the
            summary.
        """
        if isinstance(summary, bytes):
            summ = summary_pb2.Summary()
            summ.ParseFromString(summary)
            summary = summ
        # TODO(junwu): some code is missing here, check its validity
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/writer.py#L125
        event = event_pb2.Event(summary=summary)
        self._add_event(event, global_step)

    def add_graph_onnx(self, graph):
        """Adds a `Graph` protocol buffer to the event file.
        """
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self._add_event(event, None)

    def add_graph(self, graph):
        """Adds a `Graph` protocol buffer to the event file.
        """
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self._add_event(event, None)

    def add_session_log(self, session_log, global_step=None):
        """Adds a `SessionLog` protocol buffer to the event file.
        This method wraps the provided session in an `Event` protocol buffer
        and adds it to the event file.
        Args:
          session_log: A `SessionLog` protocol buffer.
          global_step: Number. Optional global step value to record with the
            summary.
        """
        event = event_pb2.Event(session_log=session_log)
        self._add_event(event, global_step)

    def _add_event(self, event, step):
        event.wall_time = time.time()
        if step is not None:
            event.step = int(step)
        self._event_writer.add_event(event)


class FileWriter(SummaryToEventTransformer):
    """Writes `Summary` protocol buffers to event files.
    The `FileWriter` class provides a mechanism to create an event file in a
    given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    @@__init__
    @@add_summary
    @@add_session_log
    @@add_event
    @@add_graph
    @@add_run_metadata
    @@get_logdir
    @@flush
    @@close
    Adapted from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/writer.py#L281
    It's recommended to use SummaryWriter as user-level APIs.
    """

    def __init__(self, logdir, graph=None, max_queue=10, flush_secs=120, graph_def=None, filename_suffix=None):
        """Creates a `FileWriter` and an event file.
        On construction the summary writer creates a new event file in `logdir`.
        This event file will contain `Event` protocol buffers constructed when you
        call one of the following functions: `add_summary()`, `add_session_log()`,
        `add_event()`, or `add_graph()`.
        If you pass a `Graph` to the constructor it is added to
        the event file. (This is equivalent to calling `add_graph()` later).
        TensorBoard will pick the graph from the file and display it graphically so
        you can interactively explore the graph you built. You will usually pass
        the graph from the session in which you launched it:
        ```python
        ...create a graph...
        # Launch the graph in a session.
        sess = tf.Session()
        # Create a summary writer, add the 'graph' to the event file.
        writer = tf.summary.FileWriter(<some-directory>, sess.graph)
        ```
        The other arguments to the constructor control the asynchronous writes to
        the event file:
        *  `flush_secs`: How often, in seconds, to flush the added summaries
           and events to disk.
        *  `max_queue`: Maximum number of summaries or events pending to be
           written to disk before one of the 'add' calls block.
        Args:
          logdir: A string. Directory where event file will be written.
          graph: A `Graph` object, such as `sess.graph`.
          max_queue: Integer. Size of the queue for pending events and summaries.
          flush_secs: Number. How often, in seconds, to flush the
            pending events and summaries to disk.
          graph_def: DEPRECATED: Use the `graph` argument instead.
        """
        event_writer = EventFileWriter(logdir, max_queue, flush_secs, filename_suffix)
        super(FileWriter, self).__init__(event_writer, graph, graph_def)

    def __enter__(self):
        """Make usable with "with" statement."""
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        """Make usable with "with" statement."""
        self.close()

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self._event_writer.get_logdir()

    def add_event(self, event):
        """Adds an event to the event file.
        Args:
          event: An `Event` protocol buffer.
        """
        self._event_writer.add_event(event)

    def flush(self):
        """Flushes the event file to disk.
        Call this method to make sure that all pending events have been written to
        disk.
        """
        self._event_writer.flush()

    def close(self):
        """Flushes the event file to disk and close the file.
        Call this method when you do not need the summary writer anymore.
        """
        self._event_writer.close()

    def reopen(self):
        """Reopens the EventFileWriter.
        Can be called after `close()` to add more events in the same directory.
        The events will go into a new events file.
        Does nothing if the EventFileWriter was not closed.
        """
        self._event_writer.reopen()


class SummaryWriter(object):
    """Writes `Summary` directly to event files.
    The `SummaryWriter` class provides a high-level api to create an event file in a
    given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    Adapted from
    https://github.com/lanpa/tensorboard-pytorch/blob/master/tensorboardX/writer.py#L220
    """
    def __init__(self, logdir, max_queue=10, flush_secs=120, filename_suffix=None):
        """
        Args:
            logdir (string): save location, default is: runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each
              run. Use hierarchical folder structure to compare between runs easily. e.g. 'runs/exp1', 'runs/exp2'
            comment (string): comment that appends to the default log_dir
        """
        self._file_writer = FileWriter(logdir=logdir, max_queue=max_queue,
                                       flush_secs=flush_secs, filename_suffix=filename_suffix)
        self._default_bins = None
        self._text_tags = []
        self._scalar_dict = {}  # {writer_id : [[timestamp, step, value],...],...}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_or_create_tf_bins(self):
        if self._default_bins is None:
            v = 1E-12
            buckets = []
            neg_buckets = []
            while v < 1E20:
                buckets.append(v)
                neg_buckets.append(-v)
                v *= 1.1
            self._default_bins = neg_buckets[::-1] + [0] + buckets
        return self._default_bins

    def get_logdir(self):
        return self._file_writer.get_logdir()

    def _append_to_scalar_dict(self, tag, scalar_value, global_step, timestamp):
        """This adds an entry to the self._scalar_dict datastructure with format
        {writer_id : [[timestamp, step, value], ...], ...}.
        """
        if tag not in self._scalar_dict.keys():
            self._scalar_dict[tag] = []
        self._scalar_dict[tag].append([timestamp, global_step, float(_make_numpy_array(scalar_value))])

    def add_scalar(self, tag, scalar_value, global_step=None):
        """Add scalar data to summary.

        Args:
            tag (string): Data identifier
            scalar_value (float): Value to save
            global_step (int): Global step value to record
        """
        self._file_writer.add_summary(scalar_summary(tag, scalar_value), global_step)
        self._append_to_scalar_dict(tag, scalar_value, global_step, time.time())

    def add_histogram(self, tag, values, global_step=None, bins=10):
        """Add histogram to summary.

        Args:
            tag (string): Data identifier
            values (numpy.array): Values to build histogram
            global_step (int): Global step value to record
            bins (int) or (string): one of {'tf','auto', 'fd', ...},
              this determines how the bins are made. You can find
              other options in: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
        """
        if bins == 'tf':
            bins = self._get_or_create_tf_bins()
        self._file_writer.add_summary(histogram_summary(tag, values, bins), global_step)

    def add_image(self, tag, image, global_step=None):
        """Add image data to summary.

        Note that this requires the ``pillow`` package.

        Args:
            tag (string): Data identifier
            image (MXNet NDArray or NumPy ndarray): Image data
            global_step (int): Global step value to record
        Shape:
            img_tensor: :math:`(3, H, W)`. Use ``torchvision.utils.make_grid()`` to prepare it is a good idea.
        """
        self._file_writer.add_summary(image_summary(tag, image), global_step)

    def add_audio(self, tag, audio, global_step=None, sample_rate=44100):
        """Add audio data to summary.

        Args:
            tag (string): Data identifier
            audio (MXNet NDArray or np.ndarray): audio data squeezable to 1D tensor
            global_step (int): Global step value to record
            sample_rate (int): sample rate in Hz

        Shape:
            snd_tensor: :math:`(1, L)`. The values should between [-1, 1].
        """
        self._file_writer.add_summary(audio_summary(tag, audio, sample_rate=sample_rate), global_step)

    def add_text(self, tag, text, global_step=None):
        """Add text data to summary.

        Args:
            tag (string): Data identifier
            text (string): String to save
            global_step (int): Global step value to record

        Examples::

            writer.add_text('lstm', 'This is an lstm', 0)
            writer.add_text('rnn', 'This is an rnn', 10)
        """
        self._file_writer.add_summary(text_summary(tag, text), global_step)
        if tag not in self._text_tags:
            self._text_tags.append(tag)
            extension_dir = self.get_logdir() + '/plugins/tensorboard_text/'
            if not os.path.exists(extension_dir):
                os.makedirs(extension_dir)
            with open(extension_dir + 'tensors.json', 'w') as fp:
                json.dump(self._text_tags, fp)

    def add_graph_onnx(self, prototxt):
        self._file_writer.add_graph_onnx(gg(prototxt))

    def add_graph(self, model, input_to_model, verbose=False):
        # prohibit second call?
        # no, let tensorboard handles it and show its warning message.
        """Add graph data to summary.

        Args:
            model (torch.nn.Module): model to draw.
            input_to_model (torch.autograd.Variable): a variable or a tuple of variables to be fed.

        """
        # TODO(junwu): add support for MXNet graphs
        import torch
        from distutils.version import LooseVersion
        if LooseVersion(torch.__version__) >= LooseVersion("0.4"):
            pass
        else:
            if LooseVersion(torch.__version__) >= LooseVersion("0.3"):
                print('You are using PyTorch==0.3, switching to calling add_graph_onnx()')
                torch.onnx.export(model, input_to_model,
                                  os.path.join(self.get_logdir(), "{}.proto".format(0)), verbose=True)
                self.add_graph_onnx(os.path.join(self.get_logdir(), "{}.proto".format(0)))
                return
            if not hasattr(torch.autograd.Variable, 'grad_fn'):
                print('add_graph() only supports PyTorch v0.2.')
                return
        self._file_writer.add_graph(graph(model, input_to_model, verbose))

    def add_embedding(self, embedding, labels=None, images=None, global_step=None, tag='default'):
        """Add embedding projector data to summary. See the following reference for the meanings
        of labels and images.
        Ref: https://www.tensorflow.org/versions/r1.2/get_started/embedding_viz

        Args:
            embedding (MXNet NDArray or NumPy ndarray): A matrix whose each row is the feature vector of the data point
            labels (list): A list of labels, each element will be converted to string
            images (MXNet NDArray or NumPy ndarray): Images correspond to each data point
            global_step (int): Global step value to record
            tag (string): Name for the embedding
        Shape:
            embedding: :math:`(N, D)`, where N is number of data and D is number of features in
            feature dim

            images: :math:`(N, C, H, W)`

        Examples::

            import keyword
            meta = []
            while len(meta)<100:
                meta = meta+keyword.kwlist # get some strings
            meta = meta[:100]

            for i, v in enumerate(meta):
                meta[i] = v+str(i)

            images = np.random.normal(size=(100, 3, 10, 32))
            for i in range(100):
                images[i] *= i / 100.0

            writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
            writer.add_embedding(torch.randn(100, 5), label_img=label_img)
            writer.add_embedding(torch.randn(100, 5), metadata=meta)
        """
        embedding_shape = embedding.shape
        if len(embedding_shape) != 2:
            raise ValueError('expected 2D NDArray as embedding data, while received an array with ndim=%d'
                             % len(embedding_shape))
        if global_step is None:
            global_step = 0
            # clear pbtxt?
        save_path = os.path.join(self.get_logdir(), str(global_step).zfill(5))
        try:
            os.makedirs(save_path)
        except OSError:
            logging.warn('embedding dir exists, did you set global_step for add_embedding()?')
        if labels is not None:
            if embedding_shape[0] != len(labels):
                raise ValueError('expected equal values of embedding first dim and length of labels,'
                                 ' while received %d and %d for each' % (embedding_shape[0], len(labels)))
            if isinstance(labels, NDArray):
                labels = labels.asnumpy().flatten()
            _make_tsv(labels, save_path)
        if images is not None:
            img_labels_shape = images.shape
            if embedding_shape[0] != img_labels_shape[0]:
                raise ValueError('expected equal first dim size of embedding and images,'
                                 ' while received %d and %d for each' % (embedding_shape[0], img_labels_shape[0]))
            _make_sprite_image(images, save_path)
        _save_ndarray(embedding, save_path)
        # new function to append to the config file a new embedding
        _add_embedding_info(labels, images, self.get_logdir(), str(global_step).zfill(5), tag)

    def add_pr_curve(self, tag, labels, predictions, global_step=None, num_thresholds=127, weights=None):
        """Adds precision recall curve.

        Args:
            tag (string): Data identifier
            labels (torch.Tensor): Ground truth data. Binary label for each element.
            predictions (torch.Tensor): The probability that an element be classified as true. Value should in [0, 1]
            global_step (int): Global step value to record
            num_thresholds (int): Number of thresholds used to draw the curve.

        """
        labels = _make_numpy_array(labels)
        predictions = _make_numpy_array(predictions)
        self._file_writer.add_summary(pr_curve_summary(tag, labels, predictions, num_thresholds, weights), global_step)

    def flush(self):
        self._file_writer.flush()

    def close(self):
        if self._file_writer is None:
            return  # ignore double close
        self._file_writer.flush()
        self._file_writer.close()

    def reopen(self):
        self._file_writer.reopen()
