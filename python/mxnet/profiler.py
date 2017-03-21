# coding: utf-8
# pylint: disable=fixme, invalid-name, too-many-arguments, too-many-locals, too-many-lines
# pylint: disable=too-many-branches, too-many-statements
"""profiler setting methods."""
from __future__ import absolute_import

import ctypes
from .base import _LIB, check_call, c_str

def profiler_set_config(mode='symbolic', filename='profile.json'):
    """Set up the configure of profiler.

    Parameters
    ----------
    mode : string, optional
        Indicting whether to enable the profiler, can
        be 'symbolic' or 'all'. Default is `symbolic`.
    filename : string, optional
        The name of output trace file. Default is
        'profile.json'.
    """
    mode2int = {'symbolic': 0, 'all': 1}
    check_call(_LIB.MXSetProfilerConfig(
        ctypes.c_int(mode2int[mode]),
        c_str(filename)))

def profiler_set_state(state='stop'):
    """Set up the profiler state to record operator.

    Parameters
    ----------
    state : string, optional
        Indicting whether to run the profiler, can
        be 'stop' or 'run'. Default is `stop`.
    """
    state2int = {'stop': 0, 'run': 1}
    check_call(_LIB.MXSetProfilerState(ctypes.c_int(state2int[state])))

def dump_profile():
    """Dump profile and stop profiler. Use this to save profile
    in advance in case your program cannot exit normally"""
    check_call(_LIB.MXDumpProfile())
