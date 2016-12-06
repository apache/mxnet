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

from . import summary_pb2
from . import event_pb2
from .event_file_writer import EventFileWriter


class SummaryToEventTransformer(object):
    """Abstractly implements the SummaryWriter API.
    This API basically implements a number of endpoints (add_summary,
    add_session_log, etc). The endpoints all generate an event protobuf, which is
    passed to the contained event_writer.
    @@__init__
    @@add_summary
    @@add_session_log
    @@add_graph
    @@add_meta_graph
    @@add_run_metadata
    """

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
        """
        self.event_writer = event_writer
        # For storing used tags for session.run() outputs.
        self._session_run_tags = {}
        # TODO[MXNet]. pass this an empty graph to check whether it's necessary.
        # currently we don't support graph in MXNet using tensorboard.

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
        event = event_pb2.Event(summary=summary)
        self._add_event(event, global_step)

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
        self.event_writer.add_event(event)


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
    """

    def __init__(self,
                 logdir,
                 graph=None,
                 max_queue=10,
                 flush_secs=120,
                 graph_def=None):
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
        event_writer = EventFileWriter(logdir, max_queue, flush_secs)
        super(FileWriter, self).__init__(event_writer, graph, graph_def)

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self.event_writer.get_logdir()

    def add_event(self, event):
        """Adds an event to the event file.
        Args:
          event: An `Event` protocol buffer.
        """
        self.event_writer.add_event(event)

    def flush(self):
        """Flushes the event file to disk.
        Call this method to make sure that all pending events have been written to
        disk.
        """
        self.event_writer.flush()

    def close(self):
        """Flushes the event file to disk and close the file.
        Call this method when you do not need the summary writer anymore.
        """
        self.event_writer.close()

    def reopen(self):
        """Reopens the EventFileWriter.
        Can be called after `close()` to add more events in the same directory.
        The events will go into a new events file.
        Does nothing if the EventFileWriter was not closed.
        """
        self.event_writer.reopen()
