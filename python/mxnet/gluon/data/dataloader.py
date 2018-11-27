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

# coding: utf-8
# pylint: disable=ungrouped-imports
"""Dataset generator."""
__all__ = ['DataLoader']

import pickle
import io
import sys
import multiprocessing
import multiprocessing.queues
from multiprocessing.reduction import ForkingPickler
import threading
import numpy as np

try:
    from Queue import Empty as QueueEmpty
except ImportError:
    from queue import Empty as QueueEmpty

try:
    import multiprocessing.resource_sharer
except ImportError:
    pass

from . import sampler as _sampler
from ... import nd, context

if sys.platform == 'darwin' or sys.platform == 'win32':
    def rebuild_ndarray(*args):
        """Rebuild ndarray from pickled shared memory"""
        # pylint: disable=no-value-for-parameter
        return nd.NDArray(nd.ndarray._new_from_shared_mem(*args))

    def reduce_ndarray(data):
        """Reduce ndarray to shared memory handle"""
        return rebuild_ndarray, data._to_shared_mem()
else:
    def rebuild_ndarray(pid, fd, shape, dtype):
        """Rebuild ndarray from pickled shared memory"""
        # pylint: disable=no-value-for-parameter
        if sys.version_info[0] == 2:
            fd = multiprocessing.reduction.rebuild_handle(fd)
        else:
            fd = fd.detach()
        return nd.NDArray(nd.ndarray._new_from_shared_mem(pid, fd, shape, dtype))

    def reduce_ndarray(data):
        """Reduce ndarray to shared memory handle"""
        # keep a local ref before duplicating fd
        data = data.as_in_context(context.Context('cpu_shared', 0))
        pid, fd, shape, dtype = data._to_shared_mem()
        if sys.version_info[0] == 2:
            fd = multiprocessing.reduction.reduce_handle(fd)
        else:
            fd = multiprocessing.reduction.DupFd(fd)
        return rebuild_ndarray, (pid, fd, shape, dtype)

ForkingPickler.register(nd.NDArray, reduce_ndarray)


class ConnectionWrapper(object):
    """Connection wrapper for multiprocessing that supports sending
    NDArray via shared memory."""

    def __init__(self, conn):
        self._conn = conn

    def send(self, obj):
        """Send object"""
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(obj)
        self.send_bytes(buf.getvalue())

    def recv(self):
        """Receive object"""
        buf = self.recv_bytes()
        return pickle.loads(buf)

    def __getattr__(self, name):
        """Emmulate conn"""
        attr = self.__dict__.get('_conn', None)
        return getattr(attr, name)


class Queue(multiprocessing.queues.Queue):
    """Wrapper for multiprocessing queue that dumps NDArray with shared memory."""
    def __init__(self, *args, **kwargs):
        if sys.version_info[0] <= 2:
            super(Queue, self).__init__(*args, **kwargs)
        else:
            super(Queue, self).__init__(*args, ctx=multiprocessing.get_context(),
                                        **kwargs)
        self._reader = ConnectionWrapper(self._reader)
        self._writer = ConnectionWrapper(self._writer)
        self._send = self._writer.send
        self._recv = self._reader.recv


class SimpleQueue(multiprocessing.queues.SimpleQueue):
    """Wrapper for multiprocessing SimpleQueue that dumps NDArray with shared memory.
       SimpleQueue don't use threading internally.
    """
    def __init__(self, *args, **kwargs):
        if sys.version_info[0] <= 2:
            super(SimpleQueue, self).__init__(*args, **kwargs)
        else:
            super(SimpleQueue, self).__init__(*args, ctx=multiprocessing.get_context(),
                                              **kwargs)
        self._reader = ConnectionWrapper(self._reader)
        self._writer = ConnectionWrapper(self._writer)
        self._send = self._writer.send
        self._recv = self._reader.recv

def default_batchify_fn(data):
    """Collate data into batch."""
    if isinstance(data[0], nd.NDArray):
        return nd.stack(*data)
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [default_batchify_fn(i) for i in data]
    else:
        data = np.asarray(data)
        return nd.array(data, dtype=data.dtype)


def default_mp_batchify_fn(data):
    """Collate data into batch. Use shared memory for stacking."""
    if isinstance(data[0], nd.NDArray):
        out = nd.empty((len(data),) + data[0].shape, dtype=data[0].dtype,
                       ctx=context.Context('cpu_shared', 0))
        return nd.stack(*data, out=out)
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [default_mp_batchify_fn(i) for i in data]
    else:
        data = np.asarray(data)
        return nd.array(data, dtype=data.dtype,
                        ctx=context.Context('cpu_shared', 0))


def _as_in_context(data, ctx):
    """Move data into new context."""
    if isinstance(data, nd.NDArray):
        return data.as_in_context(ctx)
    elif isinstance(data, (list, tuple)):
        return [_as_in_context(d, ctx) for d in data]
    return data


def worker_loop(dataset, key_queue, data_queue, batchify_fn):
    """Worker loop for multiprocessing DataLoader."""
    while True:
        idx, samples = key_queue.get()
        if idx is None:
            break
        batch = batchify_fn([dataset[i] for i in samples])
        data_queue.put((idx, batch))

def fetcher_loop(data_queue, data_buffer, pin_memory=False, data_buffer_lock=None):
    """Fetcher loop for fetching data from queue and put in reorder dict."""
    while True:
        idx, batch = data_queue.get()
        if idx is None:
            break
        if pin_memory:
            batch = _as_in_context(batch, context.cpu_pinned())
        else:
            batch = _as_in_context(batch, context.cpu())
        if data_buffer_lock is not None:
            with data_buffer_lock:
                data_buffer[idx] = batch
        else:
            data_buffer[idx] = batch


class _MultiWorkerIter(object):
    """Interal multi-worker iterator for DataLoader.
    Re-acquiring this iterator by `iter()` function will reset it 
    with all workers alive in order to save re-initialization overhead.

    Parameters
    ----------
    num_workers : int, default 0
        The number of multiprocessing workers to use for data preprocessing.
    dataset : Dataset
        Source dataset. Note that numpy and mxnet arrays can be directly used
        as a Dataset.
    batchify_fn : callable
        Callback function to allow users to specify how to merge samples
        into a batch.
    batch_sampler : Sampler
        A sampler that returns mini-batches. Do not specify batch_size,
        shuffle, sampler, and last_batch if batch_sampler is specified.
    pin_memory : boolean, default False
        If ``True``, the dataloader will copy NDArrays into pinned memory
        before returning them. Copying from CPU pinned memory to GPU is faster
        than from normal CPU memory.

    """
    def __init__(self, num_workers, dataset, batchify_fn, batch_sampler, pin_memory=False,
                 worker_fn=worker_loop):
        assert num_workers > 0, "_MultiWorkerIter is not for {} workers".format(num_workers)
        self._num_workers = num_workers
        self._dataset = dataset
        self._batchify_fn = batchify_fn
        self._batch_sampler = batch_sampler
        self._key_queue = Queue()
        self._data_queue = Queue() if sys.version_info[0] <= 2 else SimpleQueue()

        self._data_buffer = {}
        self._data_buffer_lock = threading.Lock()

        self._rcvd_idx = 0
        self._sent_idx = 0
        self._iter = iter(self._batch_sampler)
        self._shutdown = False

        workers = []
        for _ in range(self._num_workers):
            worker = multiprocessing.Process(
                target=worker_fn,
                args=(self._dataset, self._key_queue, self._data_queue, self._batchify_fn))
            worker.daemon = True
            worker.start()
            workers.append(worker)
        self._workers = workers

        self._fetcher = threading.Thread(
            target=fetcher_loop,
            args=(self._data_queue, self._data_buffer, pin_memory, self._data_buffer_lock))
        self._fetcher.daemon = True
        self._fetcher.start()

        self._reset()

    def __len__(self):
        """Get length of iterator.

        Returns
        -------
        int
            Length of iterator, equals to batch_sampler length.

        """
        return len(self._batch_sampler)

    def __del__(self):
        self.shutdown()

    def _reset(self):
        """Reset iterator with multiprocessing workers alive. Internal use. """
        assert not self._shutdown, "call reset after shutdown is forbidden"
        # clear key queue
        removed_idx = set()
        while True:
            try:
                idx, _ = self._key_queue.get(False)
                removed_idx.add(idx)
            except QueueEmpty:
                break

        # clear data queue
        while self._rcvd_idx < self._sent_idx:
            if self._rcvd_idx in removed_idx:
                self._rcvd_idx += 1
            elif self._rcvd_idx in self._data_buffer:
                _ = self._data_buffer.pop(self._rcvd_idx)
                self._rcvd_idx += 1
        assert not self._data_buffer, "data buffer should be empty"

        # reset indices and samples
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._iter = iter(self._batch_sampler)

        # pre-fetch
        for _ in range(2 * self._num_workers):
            self._push_next()

    def _push_next(self):
        """Assign next batch workload to workers. Internal use only. """
        r = next(self._iter, None)
        if r is None:
            return
        self._key_queue.put((self._sent_idx, r))
        self._sent_idx += 1

    def __next__(self):
        """Return next sample, will raise `StopIteration` reaching end.

        Returns
        -------
        NDArray
            Batched sample data.

        """
        assert not self._shutdown, "call __next__ after shutdown is forbidden"
        if self._rcvd_idx == self._sent_idx:
            assert not self._data_buffer, "Data buffer should be empty at this moment"
            raise StopIteration

        while True:
            if self._rcvd_idx in self._data_buffer:
                with self._data_buffer_lock:
                    batch = self._data_buffer.pop(self._rcvd_idx)
                self._rcvd_idx += 1
                self._push_next()
                return batch

    def next(self):
        """Compatible portal for __next__ in python2.

        Returns
        -------
        type
            Description of returned object.

        """
        return self.__next__()

    def __iter__(self):
        """Requiring iterator will reset current instance, but keep all workers
        alive, thus save re-init time of forking processes.

        Returns
        -------
        iterator
            Iterator of self.

        """
        self._reset()
        return self

    def shutdown(self):
        """
        Shutdown internal workers by pushing terminate signals. Once shutdown,
        you cannot use this instance again, you will need to obtain a new
        _MultiWorkerIter by `iter(dataloader)`.
        """
        if not self._shutdown:
            # send shutdown signal to the fetcher and join data queue first
            # Remark:   loop_fetcher need to be joined prior to the workers.
            #           otherwise, the the fetcher may fail at getting data
            self._data_queue.put((None, None))
            self._fetcher.join()
            # send shutdown signal to all worker processes
            for _ in range(self._num_workers):
                self._key_queue.put((None, None))
            # force shut down any alive worker processes
            for w in self._workers:
                if w.is_alive():
                    w.terminate()
            self._shutdown = True


class _SameProcessIter(object):
    """Same Process Iterator.
    Re-acquire this iterator by `iter()` function will reset it.

    Parameters
    ----------
    dataset : Dataset
        Source dataset. Note that numpy and mxnet arrays can be directly used
        as a Dataset.
    batchify_fn : callable
        Callback function to allow users to specify how to merge samples
        into a batch.
    batch_sampler : Sampler
        A sampler that returns mini-batches. Do not specify batch_size,
        shuffle, sampler, and last_batch if batch_sampler is specified.
    pin_memory : boolean, default False
        If ``True``, the dataloader will copy NDArrays into pinned memory
        before returning them. Copying from CPU pinned memory to GPU is faster
        than from normal CPU memory.

    """
    def __init__(self, dataset, batchify_fn, batch_sampler, pin_memory=False):
        self._dataset = dataset
        self._batchify_fn = batchify_fn
        self._batch_sampler = batch_sampler
        self._pin_memory = pin_memory
        self._reset()

    def __len__(self):
        """Get length of iterator.

        Returns
        -------
        int
            Length of iterator, equals to batch_sampler length.

        """
        return len(self._batch_sampler)

    def _reset(self):
        """Reset iterator."""
        self._iter = iter(self._batch_sampler)

    def __next__(self):
        """Return next sample, will raise `StopIteration` reaching end.

        Returns
        -------
        NDArray
            Batched sample data.

        """
        try:
            batch = next(self._iter)
        except StopIteration:
            raise StopIteration
        else:
            ret = self._batchify_fn([self._dataset[idx] for idx in batch])
            if self._pin_memory:
                ret = _as_in_context(ret, context.cpu_pinned())
            return ret

    def next(self):
        """Compatible portal for __next__ in python2.

        Returns
        -------
        type
            Description of returned object.

        """
        return self.__next__()

    def __iter__(self):
        """Requiring iterator will reset current instance.

        Returns
        -------
        iterator
            Iterator of self.

        """
        self._reset()
        return self


class DataLoader(object):
    """Loads data from a dataset and returns mini-batches of data.

    Parameters
    ----------
    dataset : Dataset
        Source dataset. Note that numpy and mxnet arrays can be directly used
        as a Dataset.
    batch_size : int
        Size of mini-batch.
    shuffle : bool
        Whether to shuffle the samples.
    sampler : Sampler
        The sampler to use. Either specify sampler or shuffle, not both.
    last_batch : {'keep', 'discard', 'rollover'}
        How to handle the last batch if batch_size does not evenly divide
        `len(dataset)`.

        keep - A batch with less samples than previous batches is returned.
        discard - The last batch is discarded if its incomplete.
        rollover - The remaining samples are rolled over to the next epoch.
    batch_sampler : Sampler
        A sampler that returns mini-batches. Do not specify batch_size,
        shuffle, sampler, and last_batch if batch_sampler is specified.
    batchify_fn : callable
        Callback function to allow users to specify how to merge samples
        into a batch. Defaults to `default_batchify_fn`::

            def default_batchify_fn(data):
                if isinstance(data[0], nd.NDArray):
                    return nd.stack(*data)
                elif isinstance(data[0], tuple):
                    data = zip(*data)
                    return [default_batchify_fn(i) for i in data]
                else:
                    data = np.asarray(data)
                    return nd.array(data, dtype=data.dtype)

    num_workers : int, default 0
        The number of multiprocessing workers to use for data preprocessing.
    pin_memory : boolean, default False
        If ``True``, the dataloader will copy NDArrays into pinned memory
        before returning them. Copying from CPU pinned memory to GPU is faster
        than from normal CPU memory.
    """
    def __init__(self, dataset, batch_size=None, shuffle=False, sampler=None,
                 last_batch=None, batch_sampler=None, batchify_fn=None,
                 num_workers=0, pin_memory=False):
        self._dataset = dataset
        self._pin_memory = pin_memory

        if batch_sampler is None:
            if batch_size is None:
                raise ValueError("batch_size must be specified unless " \
                                 "batch_sampler is specified")
            if sampler is None:
                if shuffle:
                    sampler = _sampler.RandomSampler(len(dataset))
                else:
                    sampler = _sampler.SequentialSampler(len(dataset))
            elif shuffle:
                raise ValueError("shuffle must not be specified if sampler is specified")

            batch_sampler = _sampler.BatchSampler(
                sampler, batch_size, last_batch if last_batch else 'keep')
        elif batch_size is not None or shuffle or sampler is not None or \
                last_batch is not None:
            raise ValueError("batch_size, shuffle, sampler and last_batch must " \
                             "not be specified if batch_sampler is specified.")

        self._batch_sampler = batch_sampler
        self._num_workers = num_workers if num_workers >= 0 else 0
        if batchify_fn is None:
            if num_workers > 0:
                self._batchify_fn = default_mp_batchify_fn
            else:
                self._batchify_fn = default_batchify_fn
        else:
            self._batchify_fn = batchify_fn

    def __iter__(self):
        if self._num_workers == 0:
            return _SameProcessIter(self._dataset, self._batchify_fn,
                                    self._batch_sampler, self._pin_memory)

        # multi-worker
        return _MultiWorkerIter(self._num_workers, self._dataset,
                                self._batchify_fn, self._batch_sampler, self._pin_memory)

    def __len__(self):
        return len(self._batch_sampler)
