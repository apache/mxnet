Deployment
==========

The following tutorials will help you learn how to deploy MXNet on various
platforms and in different language environments.

Export_
------
The following tutorials will help you learn export MXNet models.

.. container:: cards

   .. card::
      :title: Export ONNX Models
      :link: onnx.html

      How to export an MXNet model to the ONNX model format.

   .. card::
      :title: Export with GluonCV
      :link: https://gluon-cv.mxnet.io/build/examples_deployment/export_network.html

      How to export models trained with MXNet GluonCV.

Inference_
---------
The following tutorials will help you learn how to deploy MXNet models for inference applications.

.. container:: cards

   .. card::
      :title: GluonCV Models in a C++ Inference Application
      :link: https://gluon-cv.mxnet.io/build/examples_deployment/cpp_inference.html

      An example application that works with an exported MXNet GluonCV YOLO model.

   .. card::
      :title: Inference with Quantized Models
      :link: https://gluon-cv.mxnet.io/build/examples_deployment/int8_inference.html

      How to use quantized GluonCV models for inference on Intel Xeon Processors to gain higher performance.

Cloud_
-----
The following tutorials will show you how to use MXNet on AWS.

.. container:: cards

   .. card::
      :title: MXNet on EC2
      :link: run-on-aws/use_ec2.html

      How to deploy MXNet on an Amazon EC2 instance.

   .. card::
      :title: MXNet on SageMaker
      :link: run-on-aws/use_sagemaker.html

      How to run MXNet using Amazon SageMaker.

   .. card::
      :title: Training with Data from S3
      :link: https://mxnet.incubator.apache.org/versions/master/faq/s3_integration.html

      How to train with data from Amazon S3 buckets.

Security
--------

.. container:: cards

   .. card::
      :title: Securing MXNet
      :link: https://mxnet.incubator.apache.org/versions/master/faq/security.html

      Best practices and deployment considerations.


.. toctree::
   :hidden:
   :maxdepth: 0

   export/index
   inference/index
   run-on-aws/index

.. _Export: export/index.html
.. _Inference: inference/index.html
.. _Cloud: run-on-aws/index.html
