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

```eval_rst
.. toctree::
   :maxdepth: 1

   overview.md
   program_model.md
   note_engine.md
   note_memory.md
   note_data_loading.md
   exception_handling.md
   rnn_interface.md
```
