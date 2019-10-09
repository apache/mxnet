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

Export
======

The following tutorials will help you learn export MXNet models.
Models are by default exported as a couple of `params` and `json` files,
but you also have the option to export most models to the ONNX format.

.. container:: cards

   .. card::
      :title: Export ONNX Models
      :link: onnx.html

      Export your MXNet model to the Open Neural Exchange Format

   .. card::
      :title: Save / Load Parameters
      :link: ../../packages/gluon/blocks/save_load_params.html

      Save and Load your model parameters with MXnet


   .. card::
      :title: Export with GluonCV
      :link: https://gluon-cv.mxnet.io/build/examples_deployment/export_network.html

      How to export models trained with MXNet GluonCV.

.. toctree::
   :hidden:
   :maxdepth: 1
   :glob:

   *
   Export Gluon CV Models <https://gluon-cv.mxnet.io/build/examples_deployment/export_network.html>
   Save / Load Parameters <https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/blocks/save_load_params.html>
