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

rnn and contrib.rnn
============================

Build-in recurrent neural network layers are provided in the following two modules:


.. autosummary::
    :nosignatures:

    mxnet.gluon.rnn
    mxnet.gluon.contrib.rnn

.. currentmodule:: mxnet.gluon

Recurrent Cells
----------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    rnn.LSTMCell
    rnn.GRUCell
    rnn.RecurrentCell
    rnn.SequentialRNNCell
    rnn.BidirectionalCell
    rnn.DropoutCell
    rnn.ZoneoutCell
    rnn.ResidualCell
    contrib.rnn.Conv1DRNNCell
    contrib.rnn.Conv2DRNNCell
    contrib.rnn.Conv3DRNNCell
    contrib.rnn.Conv1DLSTMCell
    contrib.rnn.Conv2DLSTMCell
    contrib.rnn.Conv3DLSTMCell
    contrib.rnn.Conv1DGRUCell
    contrib.rnn.Conv2DGRUCell
    contrib.rnn.Conv3DGRUCell
    contrib.rnn.VariationalDropoutCell
    contrib.rnn.LSTMPCell

Recurrent Layers
----------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    rnn.RNN
    rnn.LSTM
    rnn.GRU
