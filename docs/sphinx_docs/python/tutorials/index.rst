Python Tutorials
=====

Getting started
---------------

.. container:: cards

   .. card::
      :title: A 60-minute Gluon crash course
      :link: getting-started/crash-course/index.html

      A quick overview of the core concepts of MXNet using the Gluon API.

   .. card::
      :title: Moving from other frameworks
      :link: getting-started/to-mxnet/index.html

      Guides that ease your transition to MXNet from other framework.


Packages & Modules
------------------

.. container:: cards

   .. card::
      :title: Gluon
      :link: packages/gluon/index.html

      MXNet's imperative interface for Python. If you're new to MXNet, start here!

   .. card::
      :title: NDArray API
      :link: packages/ndarray/index.html

      How to use the NDArray API to manipulate data.
      A useful set of tutorials for beginners.

   .. card::
      :title: Symbol API
      :link: packages/symbol/index.html

      How to use MXNet's Symbol API.

   .. card::
      :title: Autograd API
      :link: packages/autograd/autograd.html

      How to use Automatic Differentiation with the Autograd API.

   .. card::
      :title: Learning Rate
      :link: https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/learning_rate_schedules.html

      How to use the Learning Rate Scheduler.


Performance
-----------
.. container:: cards

   .. card::
      :title: Improving Performance
      :link: performance/perf.html

      How to get the best performance from MXNet.

   .. card::
      :title: Profiler
      :link: performance/profiler.html

      How to profile MXNet models.

   .. card::
      :title: Tuning Numpy Operations
      :link: performance/numpy.html

      Gotchas using NumPy in MXNet.

   .. card::
      :title: Compression: float16
      :link: performance/float16.html

      How to use float16 in your model to boost training speed.

   .. card::
      :title: Compression: int8
      :link: performance/index.html

      How to use int8 in your model to boost training speed.

   .. card::
      :title: Gradient Compression
      :link: performance/gradient_compression.html

      How to use gradient compression to reduce communication bandwidth and increase speed.

   .. card::
      :title: MKL-DNN
      :link: performance/mkl-dnn.html

      How to get the most from your CPU by using Intel's MKL-DNN.

   .. card::
      :title: TensorRT
      :link: performance/index.html

      How to use NVIDIA's TensorRT to boost inference performance.

   .. card::
      :title: TVM
      :link: performance/tvm.html

      How to use TVM to boost performance.


Deployment
----------
.. container:: cards

   .. card::
      :title: MXNet on EC2
      :link: deploy/run-on-aws/use_ec2.html

      How to deploy MXNet on an Amazon EC2 instance.

   .. card::
      :title: MXNet on SageMaker
      :link: deploy/run-on-aws/use_sagemaker.html

      How to run MXNet using Amazon SageMaker.

   .. card::
      :title: Training with Data from S3
      :link: deploy/run-on-aws/use_s3.html

      How to train with data from Amazon S3 buckets.

   .. card::
      :title: ONNX Models
      :link: deploy/onnx.html

      How to export an MXNet model to the ONNX model format.

      ..
         PLACEHOLDER
         .. card::
            :title: Export
            :link: deploy/export.html

            How to export MXNet models.

         .. card::
            :title: C++
            :link: deploy/cpp.html

            How to use MXNet models in a C++ environment.

         .. card::
            :title: Scala and Java
            :link: deploy/scala.html

            How to use MXNet models in a Scala or Java environment.

         .. card::
            :title: C++
            :link: deploy/cpp.html

            How to use MXNet models in a C++ environment.
         PLACEHOLDER
      ..


Customization
-------------
.. container:: cards

   .. card::
      :title: Custom Layers for Gluon
      :link: extend/custom_layer.html

      How to add new layer functionality to MXNet's imperative interface.

   .. card::
      :title: Custom Operators Using Numpy
      :link: extend/custom_op.html

      How to use Numpy to create custom MXNet operators.

   .. card::
      :title: New Layer Creation
      :link: extend/new_op.html

      How to create new MXNet operators.


Next steps
----------

- To learn more about using MXNet to implement various deep learning algorithms
  from scratch, we recommend the `Dive into Deep Learning
  <https://diveintodeeplearning.org>`_ book.

- If you are interested in building your projects based on state-of-the-art deep
  learning algorithms and/or pre-trained models, please refer to the toolkits
  in the `MXNet ecosystem <../index.html#ecosystem>`_.

- Check out the `API Reference docs <../api/index.html>`_.

.. raw:: html

   <style> h1 {display: none;} </style>
   <style>.localtoc { display: none; }</style>


.. toctree::
   :hidden:
   :maxdepth: 1

   getting-started/index
   packages/index
   performance/index
   deploy/index
   extend/index