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

mxnet.gluon
============

The Gluon library in Apache MXNet provides a clear, concise, and simple API for deep learning.
It makes it easy to prototype, build, and train deep learning models without sacrificing training speed.

Example
-------

The following example shows how you might create a simple neural network with three layers:
one input layer, one hidden layer, and one output layer.

.. code-block:: python

   net = gluon.nn.Sequential()
   # When instantiated, Sequential stores a chain of neural network layers.
   # Once presented with data, Sequential executes each layer in turn, using
   # the output of one layer as the input for the next
   with net.name_scope():
       net.add(gluon.nn.Dense(256, activation="relu")) # 1st layer (256 nodes)
       net.add(gluon.nn.Dense(256, activation="relu")) # 2nd hidden layer
       net.add(gluon.nn.Dense(num_outputs))


.. automodule:: mxnet.gluon


Tutorials
---------

.. container:: cards

   .. card::
      :title: Gluon Guide
      :link: ../../guide/packages/gluon/

      The Gluon guide. Start here!

   .. card::
      :title: Gluon-CV Toolkit
      :link: https://gluon-cv.mxnet.io/

      A Gluon add-on module for computer vision.

   .. card::
      :title: Gluon-NLP Toolkit
      :link: https://gluon-nlp.mxnet.io/

      A Gluon add-on module for natural language processing.


APIs and Packages
-----------------

Core Modules
~~~~~~~~~~~~

.. container:: cards

   .. card::
      :title: gluon.nn
      :link: nn/index.html

      Neural network components.

   .. card::
      :title: gluon.rnn
      :link: rnn/index.html

      Recurrent neural network components.

Training
~~~~~~~~

.. container:: cards

   .. card::
      :title: gluon.loss
      :link: loss/index.html

      Loss functions for training neural networks.

   .. card::
      :title: gluon.Parameter
      :link: mxnet.gluon.Parameter.html

      Parameter getting and setting functions.

   .. card::
      :title: gluon.Trainer
      :link: mxnet.gluon.Trainer.html

      Functions for applying an optimizer on a set of parameters.

Data
~~~~

.. container:: cards

   .. card::
      :title: gluon.data
      :link: data/index.html

      Dataset utilities.

   .. card::
      :title: gluon.data.vision
      :link: data/vision/index.html

      Image dataset utilities.

   .. card::
      :title: gluon.model_zoo.vision
      :link: model_zoo/index.vision.html

      A module for loading pre-trained neural network models.


Utilities
~~~~~~~~~

.. container:: cards

   .. card::
      :title: gluon.utils
      :link: utils/index.html

      A variety of utilities for training.

.. toctree::
   :hidden:
   :maxdepth: 2
   :glob:

   block
   hybrid_block
   symbol_block
   constant
   parameter
   parameter_dict
   trainer
   */index