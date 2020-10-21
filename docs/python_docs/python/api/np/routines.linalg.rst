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

.. _routines.linalg:

.. module:: mxnet.np.linalg

Linear algebra (:mod:`numpy.linalg`)
************************************

The NumPy linear algebra functions rely on BLAS and LAPACK to provide efficient
low level implementations of standard linear algebra algorithms. Those
libraries may be provided by NumPy itself using C versions of a subset of their
reference implementations but, when possible, highly optimized libraries that
take advantage of specialized processor functionality are preferred. Examples
of such libraries are OpenBLAS_, MKL (TM), and ATLAS. Because those libraries
are multithreaded and processor dependent, environmental variables and external
packages such as threadpoolctl_ may be needed to control the number of threads
or specify the processor architecture.

.. _OpenBLAS: https://www.openblas.net/
.. _threadpoolctl: https://github.com/joblib/threadpoolctl

.. currentmodule:: mxnet.np

Matrix and vector products
--------------------------
.. autosummary::
   :toctree: generated/

   dot
   vdot
   inner
   outer
   tensordot
   einsum
   linalg.multi_dot
   matmul
   linalg.matrix_power
   kron

Decompositions
--------------
.. autosummary::
   :toctree: generated/

   linalg.svd
   linalg.cholesky
   linalg.qr

Matrix eigenvalues
------------------
.. autosummary::
   :toctree: generated/

   linalg.eig
   linalg.eigh
   linalg.eigvals
   linalg.eigvalsh

Norms and other numbers
-----------------------
.. autosummary::
   :toctree: generated/

   linalg.norm
   trace
   linalg.cond
   linalg.det
   linalg.matrix_rank
   linalg.slogdet

Solving equations and inverting matrices
----------------------------------------
.. autosummary::
   :toctree: generated/

   linalg.solve
   linalg.tensorsolve
   linalg.lstsq
   linalg.inv
   linalg.pinv
   linalg.tensorinv
