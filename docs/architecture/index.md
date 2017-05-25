# MXNet Architecture

Building a high-performance deep learning library
requires many systems-level design decisions.
In this design note, we share the rationale
for the specific choices made when designing _MXNet_.
We imagine that these insights may be useful
to both deep learning practitioners
and builders of other deep learning systems.

## Deep Learning System Design Concepts

The following pages address general design concepts for deep learning systems.
Mainly, they focus on the following 3 areas:
abstraction, optimization, and trade-offs between efficiency and flexibility.
Additionally, we provide an overview of the complete MXNet system.

* [MXNet System Overview](http://mxnet.io/architecture/overview.html)
* [Deep Learning Programming Style: Symbolic vs Imperative](http://mxnet.io/architecture/program_model.html)
* [Dependency Engine for Deep Learning](http://mxnet.io/architecture/note_engine.html)
* [Optimizing the Memory Consumption in Deep Learning](http://mxnet.io/architecture/note_memory.html)
* [Efficient Data Loading Module for Deep Learning](http://mxnet.io/architecture/note_data_loading.html)
