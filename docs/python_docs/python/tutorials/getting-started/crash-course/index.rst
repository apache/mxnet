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

Crash Course
============

This crash course will give you a quick overview of the core concept of NDArray
(manipulating multiple dimensional arrays) and Gluon (create and train neural
networks). This is a good place to start if you are already familiar with
machine learning or other deep learning frameworks.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   1-ndarray
   2-nn
   3-autograd
   4-train
   5-predict
   6-use_gpus


..
   # add back the videos until apis are updated.
   You can also watch the video tutorials for this crash course. Note that two APIs
   described in vidoes have changes:

   - ``with name_scope`` is not necessary any more.
   - use ``save_parameters/load_parameters`` instead of ``save_params/load_params``

   .. raw:: html

      <style> iframe {width: 448px; height: 252px; margin: 1em 0;} </style>

      <iframe src="https://www.youtube.com/embed/r4-Ynxw0X5w" frameborder="0"
      allow="accelerometer; autoplay; encrypted-media; gyroscope;
      picture-in-picture" allowfullscreen></iframe>
