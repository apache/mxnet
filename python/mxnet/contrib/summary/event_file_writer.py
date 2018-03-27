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

"""Writes events to disk in a logdir."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os.path
import socket
import threading
import time

import six

from .proto import event_pb2
from .record_writer import RecordWriter


class EventsWriter(object):
    """Writes `Event` protocol buffers to an event file. This class is ported from
    EventsWriter defined in
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/events_writer.cc"""
    def __init__(self, file_prefix):
        """
        Events files have a name of the form
        '/file/path/events.out.tfevents.[timestamp].[hostname][file_suffix]'
        """
        self._file_prefix = file_prefix
        self._file_suffix = ''
        self._filename = None
        self._recordio_writer = None
        self._num_outstanding_events = 0

    def __del__(self):
        self.close()

    def _init_if_needed(self):
        if self._recordio_writer is not None:
            return
        self._filename = self._file_prefix + ".out.tfevents." + str(time.time())[:10] \
                         + "." + socket.gethostname() + self._file_suffix
        self._recordio_writer = RecordWriter(self._filename)
        logging.basicConfig(filename=self._filename)
        logging.info('Successfully opened events file: %s', self._filename)
        event = event_pb2.Event()
        event.wall_time = time.time()
        self.write_event(event)
        self.flush()  # flush the first event

    def init_with_suffix(self, file_suffix):
        """Initializes the events writer with file_suffix"""
        self._file_suffix = file_suffix
        self._init_if_needed()

    def write_event(self, event):
        """Appends event to the file."""
        # Check if event is of type event_pb2.Event proto.
        if not isinstance(event, event_pb2.Event):
            raise TypeError("Expected an event_pb2.Event proto, "
                            " but got %s" % type(event))
        return self._write_serialized_event(event.SerializeToString())

    def _write_serialized_event(self, event_str):
        if self._recordio_writer is None:
            self._init_if_needed()
        self._num_outstanding_events += 1
        self._recordio_writer.write_record(event_str)

    def flush(self):
        """Flushes the event file to disk."""
        if self._num_outstanding_events == 0 or self._recordio_writer is None:
            return
        self._recordio_writer.flush()
        if self._num_outstanding_events != 1:
            logging.info('Wrote %d events to disk', self._num_outstanding_events)
        else:
            logging.info('Wrote %d event to disk', self._num_outstanding_events)
        self._num_outstanding_events = 0

    def close(self):
        """Flushes the pending events and closes the writer after it is done."""
        self.flush()
        if self._recordio_writer is not None:
            self._recordio_writer.close()
            self._recordio_writer = None


class EventFileWriter(object):
    """This class is adapted from EventFileWriter in Tensorflow:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/event_file_writer.py
    Writes `Event` protocol buffers to an event file.
    The `EventFileWriter` class creates an event file in the specified directory,
    and asynchronously writes Event protocol buffers to the file. The Event file
    is encoded using the tfrecord format, which is similar to RecordIO.
    """

    def __init__(self, logdir, max_queue=10, flush_secs=120, filename_suffix=None):
        """Creates a `EventFileWriter` and an event file to write to.
        On construction the summary writer creates a new event file in `logdir`.
        This event file will contain `Event` protocol buffers, which are written to
        disk via the add_event method.
        The other arguments to the constructor control the asynchronous writes to
        the event file:
        """
        self._logdir = logdir
        if not os.path.exists(self._logdir):
            os.makedirs(self._logdir)
        self._event_queue = six.moves.queue.Queue(max_queue)
        self._ev_writer = EventsWriter(os.path.join(self._logdir, "events"))
        self._flush_secs = flush_secs
        self._sentinel_event = self._get_sentinel_event()
        if filename_suffix is not None:
            self._ev_writer.init_with_suffix(filename_suffix)
        self._closed = False
        self._worker = _EventLoggerThread(self._event_queue, self._ev_writer,
                                          self._flush_secs, self._sentinel_event)

        self._worker.start()

    def _get_sentinel_event(self):
        """Generate a sentinel event for terminating worker."""
        return event_pb2.Event()

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self._logdir

    def reopen(self):
        """Reopens the EventFileWriter.
        Can be called after `close()` to add more events in the same directory.
        The events will go into a new events file.
        Does nothing if the `EventFileWriter` was not closed.
        """
        if self._closed:
            self._worker = _EventLoggerThread(self._event_queue, self._ev_writer,
                                              self._flush_secs, self._sentinel_event)
            self._worker.start()
            self._closed = False

    def add_event(self, event):
        """Adds an event to the event file."""
        if not self._closed:
            self._event_queue.put(event)

    def flush(self):
        """Flushes the event file to disk.
        Call this method to make sure that all pending events have been written to disk.
        """
        self._event_queue.join()
        self._ev_writer.flush()

    def close(self):
        """Flushes the event file to disk and close the file.
        Call this method when you do not need the summary writer anymore.
        """
        if not self._closed:
            self.add_event(self._sentinel_event)
            self.flush()
            self._worker.join()
            self._ev_writer.close()
            self._closed = True


class _EventLoggerThread(threading.Thread):
    """Thread that logs events. Copied from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/event_file_writer.py#L133"""

    def __init__(self, queue, ev_writer, flush_secs, sentinel_event):
        """Creates an _EventLoggerThread."""
        threading.Thread.__init__(self)
        self.daemon = True
        self._queue = queue
        self._ev_writer = ev_writer
        self._flush_secs = flush_secs
        # The first event will be flushed immediately.
        self._next_event_flush_time = 0
        self._sentinel_event = sentinel_event

    def run(self):
        while True:
            event = self._queue.get()
            if event is self._sentinel_event:
                self._queue.task_done()
                break
            try:
                self._ev_writer.write_event(event)
                # Flush the event writer every so often.
                now = time.time()
                if now > self._next_event_flush_time:
                    self._ev_writer.flush()
                    # Do it again in two minutes.
                    self._next_event_flush_time = now + self._flush_secs
            finally:
                self._queue.task_done()
