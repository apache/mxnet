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

.. _routines.array-creation:

Array creation routines
=======================

.. currentmodule:: mxnet.np

Ones and zeros
--------------
.. autosummary::
   :toctree: generated/

   eye
   empty
   full
   identity
   ones
   ones_like
   zeros
   zeros_like

.. code::

   full_like
   empty_like

From existing data
------------------
.. autosummary::
   :toctree: generated/

   array
   copy

.. code::

   frombuffer
   fromfunction
   fromiter
   fromstring
   loadtxt

.. _routines.array-creation.rec:

Creating record arrays (:mod:`np.rec`)
-----------------------------------------

.. note:: :mod:`np.rec` is the preferred alias for
   :mod:`np.core.records`.

.. autosummary::
   :toctree: generated/

.. code::

   core.records.array
   core.records.fromarrays
   core.records.fromrecords
   core.records.fromstring
   core.records.fromfile

.. _routines.array-creation.char:

Creating character arrays (:mod:`np.char`)
---------------------------------------------

.. note:: :mod:`np.char` is the preferred alias for
   :mod:`np.core.defchararray`.

.. autosummary::
   :toctree: generated/

.. code::

   core.defchararray.array
   core.defchararray.asarray

Numerical ranges
----------------
.. autosummary::
   :toctree: generated/

   arange
   linspace
   logspace
   meshgrid

.. code::

   geomspace
   mgrid
   ogrid

Building matrices
-----------------
.. autosummary::
   :toctree: generated/

   tril

.. code::

   diag
   diagflat
   tri
   triu
   vander
