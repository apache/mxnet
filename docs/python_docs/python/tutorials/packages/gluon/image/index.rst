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

Image Tutorials
===============

These tutorials will help you learn how to create and use models that work with
images and other computer vision tasks.
Most of these tutorials use the `MXNet GluonCV toolkit <https://gluon-cv.mxnet.io/>`__.

Basic Image Tutorials
---------------------

.. container:: cards

   .. card::
      :title: MNIST
      :link: mnist.html

      How to create a convolutional neural network for handwritten digit recognition.

   .. card::
      :title: Image Recognition with Pretrained Models
      :link: pre-trained_models.html

      How to use pretrained models to recognize what is in an image.


GluonCV Toolkit Tutorials
-------------------------

These tutorials link to the MXNet GluonCV Toolkit website.

.. container:: cards

   .. card::
      :title: Prepare Datasets
      :link: https://gluon-cv.mxnet.io/build/examples_datasets/index.html

      How to use built-in MXNet GluonCV features for loading and preparing both common & custom datasets.

   .. card::
      :title: Image Classification
      :link: https://gluon-cv.mxnet.io/build/examples_classification/index.html

      Pretrained models for inference, fine-tune models, train your own model
      on ImageNet, and more.

   .. card::
      :title: Object Detection
      :link: https://gluon-cv.mxnet.io/build/examples_detection/index.html

      Learn how to use Single shot detector (SSD), RCNN, and YOLO models.

   .. card::
      :title: Semantic Segmentation
      :link: https://gluon-cv.mxnet.io/build/examples_segmentation/index.html

      Learn how to use and train models that can identify and segment objects in an image.

   .. card::
      :title: Instance Segmentation
      :link: https://gluon-cv.mxnet.io/build/examples_instance/index.html

      Learn how to use and train models the perform a variation of semantic
      segmentation that also classifies similar objects into discrete entities.

   .. card::
      :title: Pose Estimation
      :link: https://gluon-cv.mxnet.io/build/examples_pose/index.html

      Learn how to use a simple Pose network that predicts the heatmap for each
      joint then map it to the coordinates on the original image.


.. toctree::
   :hidden:
   :maxdepth: 1
   :glob:

   *
