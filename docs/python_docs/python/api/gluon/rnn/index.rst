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

gluon.rnn
=========

Build-in recurrent neural network layers are provided in the following two modules:


.. autosummary::
    :nosignatures:

    mxnet.gluon.rnn
    mxnet.gluon.contrib.rnn

.. currentmodule:: mxnet.gluon

Recurrent Cells
---------------

.. autosummary::
    :nosignatures:

    rnn.LSTMCell
    rnn.GRUCell
    rnn.RecurrentCell
    rnn.LSTMPCell
    rnn.SequentialRNNCell
    rnn.BidirectionalCell
    rnn.DropoutCell
    rnn.VariationalDropoutCell
    rnn.ZoneoutCell
    rnn.ResidualCell

Convolutional Recurrent Cells
-----------------------------

.. autosummary::
    :nosignatures:

    rnn.Conv1DLSTMCell
    rnn.Conv2DLSTMCell
    rnn.Conv3DLSTMCell
    rnn.Conv1DGRUCell
    rnn.Conv2DGRUCell
    rnn.Conv3DGRUCell
    rnn.Conv1DRNNCell
    rnn.Conv2DRNNCell
    rnn.Conv3DRNNCell

Recurrent Layers
----------------

.. autosummary::
    :nosignatures:

    rnn.RNN
    rnn.LSTM
    rnn.GRU

API Reference
-------------
.. automodule:: mxnet.gluon.rnn
    :members:
    :imported-members:
    :autosummary:
