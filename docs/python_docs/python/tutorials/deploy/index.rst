.. Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on an
   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.  See the License for the
   specific language governing permissions and limitations
   under the License.

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

      COMING SOON

   .. card::
      :title: Export with GluonCV
      :link: https://gluon-cv.mxnet.io/build/examples_deployment/export_network.html

      How to export models trained with MXNet GluonCV.

Inference_
---------
The following tutorials will help you learn how to deploy MXNet models for inference applications.

.. container:: cards

   .. card::
      :title: CPP Inference
      :link: inference/cpp.html

      How to deploy MXNet C++ Models

   .. card::
      :title: Scala Inference
      :link: inference/scala.html

      How to run Scala inference

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
      :title: MXNet on the cloud
      :link: run-on-aws/cloud.html

      How to run MXNet on the cloud

   .. card::
      :title: Training with Data from S3
      :link: https://mxnet.apache.org/api/faq/s3_integration

      How to train with data from Amazon S3 buckets.

Security
--------

.. container:: cards

   .. card::
      :title: Securing MXNet
      :link: https://mxnet.apache.org/api/faq/security

      Best practices and deployment considerations.


.. toctree::
   :hidden:
   :maxdepth: 1

   export/index
   inference/index
   run-on-aws/index

.. _Export: export/index.html
.. _Inference: inference/index.html
.. _Cloud: run-on-aws/index.html
