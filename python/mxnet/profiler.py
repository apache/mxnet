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
# pylint: disable=fixme, invalid-name, too-many-arguments, too-many-locals, too-many-lines
# pylint: disable=too-many-branches, too-many-statements
"""Profiler setting methods."""
from __future__ import absolute_import
import ctypes
import warnings
from .base import _LIB, check_call, c_str, ProfileHandle, c_str_array, py_str, KVStoreHandle

profiler_kvstore_handle = KVStoreHandle()

def set_kvstore_handle(handle):
    global profiler_kvstore_handle
    profiler_kvstore_handle = handle

def set_config(**kwargs):
    """Set up the configure of profiler (only accepts keyword arguments).

    Parameters
    ----------
    filename : string,
        output file for profile data
    profile_all : boolean,
        all profile types enabled
    profile_symbolic : boolean,
        whether to profile symbolic operators
    profile_imperative : boolean,
        whether to profile imperative operators
    profile_memory : boolean,
        whether to profile memory usage
    profile_api : boolean,
        whether to profile the C API
    contiguous_dump : boolean,
        whether to periodically dump profiling data to file
    dump_period : float,
        seconds between profile data dumps
    aggregate_stats : boolean,
        whether to maintain aggregate stats in memory for console
        dump.  Has some negative performance impact.
    profile_process : string
        whether to profile kvstore `server` or `worker`.
        server can only be profiled when kvstore is of type dist.
        if this is not passed, defaults to `worker`
    """
    kk = kwargs.keys()
    vv = kwargs.values()
    check_call(_LIB.MXSetProcessProfilerConfig(len(kwargs),
                                               c_str_array([key for key in kk]),
                                               c_str_array([str(val) for val in vv]),
                                               profiler_kvstore_handle))


def profiler_set_config(mode='symbolic', filename='profile.json'):
    """Set up the configure of profiler (Deprecated).

    Parameters
    ----------
    mode : string, optional
        Indicates whether to enable the profiler, can
        be 'symbolic', or 'all'. Defaults to `symbolic`.
    filename : string, optional
        The name of output trace file. Defaults to 'profile.json'.
    """
    warnings.warn('profiler.profiler_set_config() is deprecated. '
                  'Please use profiler.set_config() instead')
    keys = c_str_array([key for key in ["profile_" + mode, "filename"]])
    values = c_str_array([str(val) for val in [True, filename]])
    assert len(keys) == len(values)
    check_call(_LIB.MXSetProcessProfilerConfig(len(keys), keys, values, profiler_kvstore_handle))


def set_state(state='stop', profile_process='worker'):
    """Set up the profiler state to 'run' or 'stop'.

    Parameters
    ----------
    state : string, optional
        Indicates whether to run the profiler, can
        be 'stop' or 'run'. Default is `stop`.
    profile_process : string
        whether to profile kvstore `server` or `worker`.
        server can only be profiled when kvstore is of type dist.
        if this is not passed, defaults to `worker`
    """
    state2int = {'stop': 0, 'run': 1}
    profile_process2int = {'worker': 0, 'server': 1}
    check_call(_LIB.MXSetProcessProfilerState(ctypes.c_int(state2int[state]),
                                              profile_process2int[profile_process],
                                              profiler_kvstore_handle))


def profiler_set_state(state='stop'):
    """Set up the profiler state to 'run' or 'stop' (Deprecated).

    Parameters
    ----------
    state : string, optional
        Indicates whether to run the profiler, can
        be 'stop' or 'run'. Default is `stop`.
    """
    warnings.warn('profiler.profiler_set_state() is deprecated. '
                  'Please use profiler.set_state() instead')
    set_state(state)

def dump(finished=True, profile_process='worker'):
    """Dump profile and stop profiler. Use this to save profile
    in advance in case your program cannot exit normally.

    Parameters
    ----------
    finished : boolean
        Indicates whether to stop statistic output (dumping) after this dump.
        Default is True
    profile_process : string
        whether to profile kvstore `server` or `worker`.
        server can only be profiled when kvstore is of type dist.
        if this is not passed, defaults to `worker`
    """
    fin = 1 if finished is True else 0
    profile_process2int = {'worker': 0, 'server': 1}
    check_call(_LIB.MXDumpProcessProfile(fin,
                                         profile_process2int[profile_process],
                                         profiler_kvstore_handle))


def dump_profile():
    """Dump profile and stop profiler. Use this to save profile
    in advance in case your program cannot exit normally."""
    warnings.warn('profiler.dump_profile() is deprecated. '
                  'Please use profiler.dump() instead')
    dump(True)


def dumps(reset=False):
    """Return a printable string of aggregate profile stats.

    Parameters
    ----------
    reset: boolean
        Indicates whether to clean aggeregate statistical data collected up to this point
    """
    debug_str = ctypes.c_char_p()
    do_reset = 1 if reset is True else 0
    check_call(_LIB.MXAggregateProfileStatsPrint(ctypes.byref(debug_str), int(do_reset)))
    return py_str(debug_str.value)


def pause(profile_process='worker'):
    """Pause profiling.

    Parameters
    ----------
    profile_process : string
        whether to profile kvstore `server` or `worker`.
        server can only be profiled when kvstore is of type dist.
        if this is not passed, defaults to `worker`
    """
    profile_process2int = {'worker': 0, 'server': 1}
    check_call(_LIB.MXProcessProfilePause(int(1),
                                          profile_process2int[profile_process],
                                          profiler_kvstore_handle))


def resume(profile_process='worker'):
    """
    Resume paused profiling.

    Parameters
    ----------
    profile_process : string
        whether to profile kvstore `server` or `worker`.
        server can only be profiled when kvstore is of type dist.
        if this is not passed, defaults to `worker`
    """
    profile_process2int = {'worker': 0, 'server': 1}
    check_call(_LIB.MXProcessProfilePause(int(0),
                                          profile_process2int[profile_process],
                                          profiler_kvstore_handle))


class Domain(object):
    """Profiling domain, used to group sub-objects like tasks, counters, etc into categories
    Serves as part of 'categories' for chrome://tracing

    Note: Domain handles are never destroyed.

    Parameters
    ----------
    name : string
        Name of the domain
    """
    def __init__(self, name):
        self.name = name
        self.handle = ProfileHandle()
        check_call(_LIB.MXProfileCreateDomain(c_str(self.name), ctypes.byref(self.handle)))

    def __str__(self):
        return self.name

    def new_task(self, name):
        """Create new Task object owned by this domain

        Parameters
        ----------
        name : string
            Name of the task
        """
        return Task(self, name)

    def new_frame(self, name):
        """Create new Frame object owned by this domain

        Parameters
        ----------
        name : string
            Name of the frame
        """
        return Frame(self, name)

    def new_counter(self, name, value=None):
        """Create new Counter object owned by this domain

        Parameters
        ----------
        name : string
            Name of the counter
        """
        return Counter(self, name, value)

    def new_marker(self, name):
        """Create new Marker object owned by this domain

        Parameters
        ----------
        name : string
            Name of the marker
        """
        return Marker(self, name)

class Task(object):
    """Profiling Task class.

    A task is a logical unit of work performed by a particular thread.
    Tasks can nest; thus, tasks typically correspond to functions, scopes, or a case block
    in a switch statement.
    You can use the Task API to assign tasks to threads.

    This is different from Frame in that all profiling statistics for passes
    through the task's begin and endpoints are accumulated together into a single statistical
    analysys, rather than a separate analysis for each pass (as with a Frame)

    Parameters
    ----------
    domain : Domain object
        Domain to which this object belongs
    name : string
        Name of the task
    """
    def __init__(self, domain, name):
        self.name = name
        self.handle = ProfileHandle()
        check_call(_LIB.MXProfileCreateTask(domain.handle,
                                            c_str(self.name),
                                            ctypes.byref(self.handle)))

    def __del__(self):
        if self.handle is not None:
            check_call(_LIB.MXProfileDestroyHandle(self.handle))

    def start(self):
        """Start timing scope for this object"""
        check_call(_LIB.MXProfileDurationStart(self.handle))

    def stop(self):
        """Stop timing scope for this object"""
        check_call(_LIB.MXProfileDurationStop(self.handle))

    def __str__(self):
        return self.name


class Frame(object):
    """Profiling Frame class.

    Use the frame API to insert calls to the desired places in your code and analyze
    performance per frame, where frame is the time period between frame begin and end points.
    When frames are displayed in Intel VTune Amplifier, they are displayed in a
    separate track, so they provide a way to visually separate this data from normal task data.

    This is different from Task in that each 'Frame' duration will be a discretely-numbered
    event in the VTune output, as well as its rate (frame-rate) shown.  This is analogous to
    profiling each frame of some visual output, such as rendering a video game frame.

    Parameters
    ----------
    domain : Domain object
        Domain to which this object belongs
    name : string
        Name of the frame
    """
    def __init__(self, domain, name):
        self.name = name
        self.handle = ProfileHandle()
        check_call(_LIB.MXProfileCreateFrame(domain.handle,
                                             c_str(self.name),
                                             ctypes.byref(self.handle)))

    def __del__(self):
        if self.handle is not None:
            check_call(_LIB.MXProfileDestroyHandle(self.handle))

    def start(self):
        """Start timing scope for this object"""
        check_call(_LIB.MXProfileDurationStart(self.handle))

    def stop(self):
        """Stop timing scope for this object"""
        check_call(_LIB.MXProfileDurationStop(self.handle))

    def __str__(self):
        return self.name


class Event(object):
    """Profiling Event class.

    The event API is used to observe when demarcated events occur in your application, or to
    identify how long it takes to execute demarcated regions of code. Set annotations in the
    application to demarcate areas where events of interest occur.
    After running analysis, you can see the events marked in the Timeline pane.
    Event API is a per-thread function that works in resumed state.
    This function does not work in paused state.

    Parameters
    ----------
    name : string
        Name of the event
    """
    def __init__(self, name):
        self.name = name
        self.handle = ProfileHandle()
        check_call(_LIB.MXProfileCreateEvent(c_str(self.name), ctypes.byref(self.handle)))

    def __del__(self):
        if self.handle is not None:
            check_call(_LIB.MXProfileDestroyHandle(self.handle))

    def start(self):
        """Start timing scope for this object"""
        check_call(_LIB.MXProfileDurationStart(self.handle))

    def stop(self):
        """Stop timing scope for this object"""
        check_call(_LIB.MXProfileDurationStop(self.handle))

    def __str__(self):
        return self.name


class Counter(object):
    """Profiling Counter class.

    The counter event can track a value as it changes over time.

    Parameters
    ----------
    domain : Domain object
        Domain to which this object belongs
    name : string
        Name of the counter
    value: integer, optional
        Initial value of the counter
    """
    def __init__(self, domain, name, value=None):
        self.name = name
        self.handle = ProfileHandle()
        check_call(_LIB.MXProfileCreateCounter(domain.handle,
                                               c_str(name),
                                               ctypes.byref(self.handle)))
        if value is not None:
            self.set_value(value)

    def __del__(self):
        if self.handle is not None:
            check_call(_LIB.MXProfileDestroyHandle(self.handle))


    def set_value(self, value):
        """Set counter value.

        Parameters
        ----------
        value : int
            Value for the counter
        """
        check_call(_LIB.MXProfileSetCounter(self.handle, int(value)))

    def increment(self, delta=1):
        """Increment counter value.

        Parameters
        ----------
        value_change : int
            Amount by which to add to the counter
        """
        check_call(_LIB.MXProfileAdjustCounter(self.handle, int(delta)))

    def decrement(self, delta=1):
        """Decrement counter value.

        Parameters
        ----------
        value_change : int
            Amount by which to subtract from the counter
        """
        check_call(_LIB.MXProfileAdjustCounter(self.handle, -int(delta)))

    def __iadd__(self, delta):
        self.increment(delta)
        return self

    def __isub__(self, delta):
        self.decrement(delta)
        return self

    def __str__(self):
        return self.name


class Marker(object):
    """Set marker for an instant in time.

    The marker event marks a particular instant in time across some scope boundaries.

    Parameters
    ----------
    domain : Domain object
        Domain to which this object belongs
    name : string
        Name of the marker
    """
    def __init__(self, domain, name):
        self.name = name
        self.domain = domain

    def mark(self, scope='process'):
        """Set up the profiler state to record operator.

        Parameters
        ----------
        scope : string, optional
            Indicates what scope the marker should refer to.
            Can be 'global', 'process', thread', task', and 'marker'
            Default is `process`.
        """
        check_call(_LIB.MXProfileSetMarker(self.domain.handle, c_str(self.name), c_str(scope)))
