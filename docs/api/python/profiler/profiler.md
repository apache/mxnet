# Profiler API

## Overview

MXNet has a built-in profiler which is compatibule with both Intel® VTune™ Amplifier as well as Chrome's chrome://tracing visualization engine.  When built witht he USE_VTUNE=1 flag, MXNet makes actual VTune API calls to define Domains, Frames, Tasks, Events Counters, and Markers.  For a detailed explanation of these, see [Instrumentation and Tracing Technology API Reference ](https://software.intel.com/en-us/vtune-amplifier-help-instrumentation-and-tracing-technology-api-reference)

```eval_rst
.. autosummary::
    :nosignatures:

    mxnet.profiler
```

## API Reference

```eval_rst
    .. currentmodule:: mxnet
```

### Profiling system control


```eval_rst
.. autosummary::
    :nosignatures:

    profiler.set_config
    profiler.set_state
    profiler.pause
    profiler.resume
    profiler.dump
    profiler.dumps
```

### Profiling Objects

These profiling objects can be created and accessed from python in order to resord performance information of the python code paths 

```eval_rst
.. autosummary::
    :nosignatures:

    profiler.Domain
    profiler.Task
    profiler.Frame
    profiler.Event
    profiler.Counter
    profiler.Marker
```

### Example usage
```python
    profiler.set_config(profile_all=True,
                        filename='chrome_tracing_profile.json',  # File used for chrome://tracing visualization
                        continuous_dump=True,
                        aggregate_stats=True)  # Stats printed by dumps() call
                        
    profiler.set_state('run')  # Start profiling engine
    #
    # Profile this section of code
    #
    profiler.pause()  # Pause profiling
    #
    # Don't profile this section
    #
    profiler.resume()  # Resume profiling
    #
    # Profile this section of code 
    #
    profiler.set_state('stop')  # Stop profiling engine (optional)
    print(profiler.dumps())  # Print aggregate statistics if aggregate_stats was set to True
```

## API Reference

<script type="text/javascript" src='../../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.profiler
    :members:
```
<script>auto_index("api-reference");</script>
