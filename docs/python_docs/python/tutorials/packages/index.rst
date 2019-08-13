Packages
========

The documents in this unit dive into the details how each MXNet module works.

High Level APIs
---------------

.. container:: cards

   .. card::
      :title: Gluon
      :link: gluon/index.html

      MXNet's imperative interface for Python. If you're new to MXNet, start here!


Shared APIs
-----------

.. container:: cards

   .. card::
      :title: NDArray API
      :link: ndarray/index.html

      How to use the NDArray API to manipulate data.
      A useful set of tutorials for beginners.

   .. card::
      :title: Symbol API
      :link: symbol/index.html

      How to use MXNet's Symbol API.

   .. card::
      :title: Autograd API
      :link: autograd/autograd.html

      How to use Automatic Differentiation with the Autograd API.

   .. card::
      :title: Learning Rate
      :link: https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/learning_rate_schedules.html

      How to use the Learning Rate Scheduler.

   .. card::
      :title: KVStore API
      :link: kvstore/index.html

      How to use the KVStore API for distributed training.

   .. card::
      :title: Data APIs
      :link: data/index.html

      How to use MXNet's data APIs.

   .. card::
      :title: Visualizations
      :link: viz/index.html

      How to use MXNet's visualization features.

   .. card::
      :title: ONNX
      :link: onnx/index.html

      How to use Open Neural Network Exchange (ONNX) with MXNet.

   ..
      .. card::
         :title: Optimizer
         :link: optimizer.html

         How to use optimizer.
   ..

Old APIs
--------
Currently supported, but not recommended APIs.

.. container:: cards

   .. card::
      :title: Module
      :link: module/index.html

      MXNet's symbolic interface for Python.


.. toctree::
   :hidden:

   gluon/index
   ndarray/index
   symbol/index
   autograd/autograd
   onnx/index

..
   Basic
   -----

   .. toctree::
      :maxdepth: 1

      mxboard
      gpus

   Advanced
   --------


   .. toctree::
      :maxdepth: 1

      symbol
      record-io
      sparse
      control-flow
      distributed-training
