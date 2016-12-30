# coding: utf-8
# pylint: disable=protected-access, logging-format-interpolation, invalid-name, no-member, too-many-branches
"""Monitor outputs, weights, and gradients for debugging."""
from __future__ import absolute_import

import re
import ctypes
import logging
from math import sqrt

from .ndarray import NDArray
from .base import NDArrayHandle, py_str
from . import ndarray


class Monitor(object):
    """Monitor outputs, weights, and gradients for debugging.

    Parameters
    ----------
    interval : int
        Number of batches between printing.
    stat_func : function
        a function that computes statistics of tensors.
        Takes a NDArray and returns a NDArray. defaults to mean
        absolute value |x|/size(x).
    pattern : str
        A regular expression specifying which tensors to monitor.
        Only tensors with names that match name_pattern will be included.
        For example, '.*weight|.*output' will print all weights and outputs;
        '.*backward.*' will print all gradients.
    """
    def __init__(self, interval, stat_func=None, pattern='.*', sort=False):
        if stat_func is None:
            def asum_stat(x):
                """returns |x|/size(x), async execution."""
                return ndarray.norm(x)/sqrt(x.size)
            stat_func = asum_stat
        self.stat_func = stat_func
        self.interval = interval
        self.activated = False
        self.queue = []
        self.step = 0
        self.exes = []
        self.re_prog = re.compile(pattern)
        self.sort = sort
        def stat_helper(name, array):
            """wrapper for executor callback"""
            array = ctypes.cast(array, NDArrayHandle)
            array = NDArray(array, writable=False)
            if not self.activated or not self.re_prog.match(py_str(name)):
                return
            self.queue.append((self.step, py_str(name), self.stat_func(array)))
        self.stat_helper = stat_helper

    def install(self, exe):
        """install callback to executor.
        Supports installing to multiple exes

        Parameters
        ----------
        exe : mx.executor.Executor
            the Executor (returned by symbol.bind) to install to.
        """
        exe.set_monitor_callback(self.stat_helper)
        self.exes.append(exe)

    def tic(self):
        """start collecting stats for current batch.
        Call before forward"""
        if self.step % self.interval == 0:
            for exe in self.exes:
                for array in exe.arg_arrays:
                    array.wait_to_read()
                for array in exe.aux_arrays:
                    array.wait_to_read()
            self.queue = []
            self.activated = True
        self.step += 1


    def toc(self):
        """End collecting for current batch and return results.
        Call after computation of current batch.

        Returns
        -------
        res : list of """
        if not self.activated:
            return []
        for exe in self.exes:
            for array in exe.arg_arrays:
                array.wait_to_read()
            for array in exe.aux_arrays:
                array.wait_to_read()
        for exe in self.exes:
            for name, array in zip(exe._symbol.list_arguments(), exe.arg_arrays):
                if self.re_prog.match(name):
                    self.queue.append((self.step, name, self.stat_func(array)))
            for name, array in zip(exe._symbol.list_auxiliary_states(), exe.aux_arrays):
                if self.re_prog.match(name):
                    self.queue.append((self.step, name, self.stat_func(array)))
        self.activated = False
        res = []
        if self.sort:
            self.queue.sort(key=lambda x: x[1])
        for n, k, v_list in self.queue:
            if isinstance(v_list, NDArray):
                v_list = [v_list]
            assert isinstance(v_list, list)
            s = ''
            for v in v_list:
                assert isinstance(v, NDArray)
                if v.shape == (1,):
                    s += str(v.asscalar()) + '\t'
                else:
                    s += str(v.asnumpy()) + '\t'
            res.append((n, k, s))
        self.queue = []
        return res

    def toc_print(self):
        """End collecting and print results"""
        res = self.toc()
        for n, k, v in res:
            logging.info('Batch: {:7d} {:30s} {:s}'.format(n, k, v))
