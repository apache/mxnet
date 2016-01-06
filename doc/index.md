MXNet Documentation
===================
[MXNet](https://github.com/dmlc/mxnet) is a deep learning framework designed for both *efficiency* and *flexibility*.
It allows you to mix the flavours of deep learning programs together to maximize the efficiency and your productivity.

User Guide
----------
* [Build and Installation](build.md)
* [Python Package Document](python/index.md)
* [R Package Document](R-package/index.md)
* [MXNet.jl Julia Package](https://github.com/dmlc/MXNet.jl)
* [Pretrained Model Gallery](pretrained.md)
* [Distributed Training](distributed_training.md)
* [Frequently Asked Questions](faq.md)
* [MXNet Overview in Chinese](overview_zh.md)

Developer Guide
---------------
* [Developer Documents](developer-guide/index.md)
* [Environment Variables for MXNet](env_var.md)
* [Contributor Guideline](contribute.md)
* [Doxygen Version of C++ API](https://mxnet.readthedocs.org/en/latest/doxygen)


Open Source Design Notes
------------------------
This section contains the design document and notes we made for mxnet system design and deep learning
libraries in general. We believe that open sourcing the system design note can letgeneral audiences understand the  motivations and the benefits drawbacks of design choices, and help those who uses deep learning and who builds deep learning systems.

This section will be updated with self-contained design notes on various aspect of deep learning systems,
in terms of abstraction, optimization and trade-offs.

* [Programming Models for Deep Learning](program_model.md)
* [Dependency Engine for Deep Learning](developer-guide/note_engine.md)
* [Squeeze the Memory Consumption of Deep Learning](developer-guide/note_memory.md)
* [Efficient Data Loading Module for Deep Learning](developer-guide/note_data_loading.md)

Tutorial
--------
* [Training Deep Net on 14 Million Images on A Single Machine](tutorial/imagenet_full.md)
* [How to Create New Operations (Layers)](tutorial/new_op_howto.md)
* [Deep Learning in a Single File for Smart Devices](tutorial/smart_device.md)

Indices and tables
------------------

```eval_rst
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```
