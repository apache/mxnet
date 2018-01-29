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

from __future__ import print_function
from ctypes import c_bool
import multiprocessing as mp
try:
    from queue import Full as QFullExcept
    from queue import Empty as QEmptyExcept
except ImportError:
    from Queue import Full as QFullExcept
    from Queue import Empty as QEmptyExcept

import numpy as np


class MPData(object):
    """
    Handles multi-process data generation.

    Operation:
        - call start() to start the data generation
        - call get() (blocking) to read one sample
        - call reset() to stop data generation
    """
    def __init__(self, num_processes, max_queue_size, fn):
        """

        Parameters
        ----------
        num_processes: int
            Number of processes to spawn
        max_queue_size: int
            Maximum samples in the queue before processes wait
        fn: function
            function that generates samples, executed on separate processes.
        """
        self.queue = mp.Queue(maxsize=int(max_queue_size))
        self.alive = mp.Value(c_bool, False, lock=False)
        self.num_proc = num_processes
        self.proc = list()
        self.fn = fn

    def start(self):
        """
        Starts the processes
        Parameters
        ----------
        fn: function

        """
        """
        Starts the processes
        """
        self._init_proc()

    @staticmethod
    def _proc_loop(proc_id, alive, queue, fn):
        """
        Thread loop for generating data

        Parameters
        ----------
        proc_id: int
            Process id
        alive: multiprocessing.Value
            variable for signaling whether process should continue or not
        queue: multiprocessing.Queue
            queue for passing data back
        fn: function
            function object that returns a sample to be pushed into the queue
        """
        print("proc {} started".format(proc_id))
        try:
            while alive.value:
                data = fn()
                put_success = False
                while alive.value and not put_success:
                    try:
                        queue.put(data, timeout=0.5)
                        put_success = True
                    except QFullExcept:
                        # print("Queue Full")
                        pass
        except KeyboardInterrupt:
            print("W: interrupt received, stopping process {} ...".format(proc_id))
        print("Closing process {}".format(proc_id))
        queue.close()

    def _init_proc(self):
        """
        Start processes if not already started
        """
        if not self.proc:
            self.proc = [
                mp.Process(target=self._proc_loop, args=(i, self.alive, self.queue, self.fn))
                for i in range(self.num_proc)
            ]
            self.alive.value = True
            for p in self.proc:
                p.start()

    def get(self):
        """
        Get a datum from the queue

        Returns
        -------
        np.ndarray
            A captcha image, normalized to [0, 1]
        """
        self._init_proc()
        return self.queue.get()

    def reset(self):
        """
        Resets the generator by stopping all processes
        """
        self.alive.value = False
        qsize = 0
        try:
            while True:
                self.queue.get(timeout=0.1)
                qsize += 1
        except QEmptyExcept:
            pass
        print("Queue size on reset: {}".format(qsize))
        for i, p in enumerate(self.proc):
            p.join()
        self.proc.clear()
