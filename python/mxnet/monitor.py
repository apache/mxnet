# coding: utf-8
# pylint: disable=protected-access, logging-format-interpolation, invalid-name, no-member
"""Monitor outputs, weights, and gradients for debugging."""
import ctypes
from .ndarray import NDArray
from .base import NDArrayHandle
from . import ndarray
import logging
from math import sqrt


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
    """
    def __init__(self, interval, stat_func=None):
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
        def stat_helper(name, array):
            """wrapper for executor callback"""
            if not self.activated:
                return
            array = ctypes.cast(array, NDArrayHandle)
            array = NDArray(array, writable=False)
            self.queue.append((self.step, name, self.stat_func(array)))
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
            self.queue = []
            self.activated = True
        self.step += 1


    def toc(self):
        """End collecting for current batch and return results.
        Call after computation of current batch.

        Returns
        -------
        res : list of """
        if self.activated:
            for exe in self.exes:
                for array in exe.arg_arrays:
                    array.wait_to_read()
            for exe in self.exes:
                for name, array in zip(exe._symbol.list_arguments(), exe.arg_arrays):
                    self.queue.append((self.step, name, self.stat_func(array)))
        else:
            return []
        self.activated = False
        res = []
        for n, k, v in self.queue:
            assert isinstance(v, NDArray)
            if v.shape == (1,):
                res.append((n, k, str(v.asscalar())))
            else:
                res.append((n, k, str(v.asnumpy())))
        self.queue = []
        return res

    def toc_print(self):
        """End collecting and print results"""
        res = self.toc()
        for n, k, v in res:
            logging.info('Batch: {:7d} {:30s} {:s}'.format(n, k, v))




