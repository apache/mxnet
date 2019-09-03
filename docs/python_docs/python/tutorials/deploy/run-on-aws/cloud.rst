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

MXNet on the Cloud
==================

Deep learning can require extremely powerful hardware, often for
unpredictable durations of time. Moreover, *MXNet* can benefit from both
multiple GPUs and multiple machines. Accordingly, cloud computing, as
offered by AWS and others, is especially well suited to training deep
learning models. Using AWS, we can rapidly fire up multiple machines
with multiple GPUs each at will and maintain the resources for precisely
the amount of time needed.

Set Up an AWS GPU Cluster from Scratch
--------------------------------------

In this document, we provide a step-by-step guide that will teach you
how to set up an AWS cluster with *MXNet*. We show how to:

-  ``Use Amazon S3 to host data``\ \_
-  ``Set up an EC2 GPU instance with all dependencies installed``\ \_
-  ``Build and run MXNet on a single computer``\ \_
-  ``Set up an EC2 GPU cluster for distributed training``\ \_

Use Amazon S3 to Host Data
:sub:`:sub:`:sub:`:sub:`:sub:`:sub:`:sub:`:sub:`:sub:`:sub:`~`````````\ ~`\ ~~

Amazon S3 provides distributed data storage which proves especially
convenient for hosting large datasets. To use S3, you need
``AWS credentials``\ \_, including an ``ACCESS_KEY_ID`` and a
``SECRET_ACCESS_KEY``.

To use *MXNet* with S3, set the environment variables
``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY`` by adding the
following two lines in ``~/.bashrc`` (replacing the strings with the
correct ones):

.. code:: bash

export AWS\_ACCESS\_KEY\_ID=AKIAIOSFODNN7EXAMPLE export
AWS\_SECRET\_ACCESS\_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

There are several ways to upload data to S3. One simple way is to use
``s3cmd``\ \_. For example:

.. code:: bash

wget http://data.mxnet.io/mxnet/data/mnist.zip unzip mnist.zip && s3cmd
put t\*-ubyte s3://dmlc/mnist/

Use Pre-installed EC2 GPU Instance
:sub:`:sub:`~`\ :sub:`:sub:`:sub:`:sub:`:sub:`:sub:`:sub:`:sub:`:sub:`:sub:`:sub:`:sub:`~`````````````\ ~~

The ``Deep Learning AMI``\ \_ is an Amazon Linux image supported and
maintained by Amazon Web Services for use on Amazon Elastic Compute
Cloud (Amazon EC2). It contains ``MXNet-v0.9.3 tag``\ \_ and the
necessary components to get going with deep learning, including Nvidia
drivers, CUDA, cuDNN, Anaconda, Python2 and Python3. The AMI IDs are the
following:

-  us-east-1: ami-e7c96af1
-  us-west-2: ami-dfb13ebf
-  eu-west-1: ami-6e5d6808

Now you can launch *MXNet* directly on an EC2 GPU instance. You can also
use ``Jupyter``\ \_ notebook on EC2 machine. Here is a
``good tutorial``\ \_ on how to connect to a Jupyter notebook running on
an EC2 instance.

Set Up an EC2 GPU Instance from Scratch
:sub:`:sub:`:sub:`:sub:`:sub:`:sub:`:sub:`~``````\ :sub:`:sub:`:sub:`:sub:`:sub:`:sub:`:sub:`~```````\ :sub:`:sub:`~```

*MXNet* requires the following libraries:

-  C++ compiler with C++11 support, such as ``gcc >= 4.8``
-  ``CUDA`` (``CUDNN`` in optional) for GPU linear algebra
-  ``BLAS`` (cblas, open-blas, atblas, mkl, or others)

.. \_Use Amazon S3 to host data: #use-amazon-s3-to-host-data .. \_Set up
an EC2 GPU instance with all dependencies installed:
#set-up-an-ec2-gpu-instance .. \_Build and run MXNet on a single
computer: #build-and-run-mxnet-on-a-gpu-instance .. \_Set up an EC2 GPU
cluster for distributed training:
#set-up-an-ec2-gpu-cluster-for-distributed-training .. \_AWS
credentials:
http://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSGettingStartedGuide/AWSCredentials.html
.. \_s3cmd: http://s3tools.org/s3cmd .. *Deep Learning AMI:
https://aws.amazon.com/marketplace/pp/B01M0AXXQB?qid=1475211685369&sr=0-1&ref*\ =srh\_res\_product\_title
.. \_MXNet-v0.9.3 tag: https://github.com/dmlc/mxnet .. \_Jupyter:
http://jupyter.org .. \_good tutorial:
https://github.com/dmlc/mxnet-notebooks
