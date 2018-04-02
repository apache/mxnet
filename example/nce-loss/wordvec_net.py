# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=missing-docstring
from __future__ import print_function

import mxnet as mx
from nce import nce_loss, nce_loss_subwords


def get_word_net(vocab_size, num_input):
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    label_weight = mx.sym.Variable('label_weight')
    embed_weight = mx.sym.Variable('embed_weight')
    data_embed = mx.sym.Embedding(data=data, input_dim=vocab_size,
                                  weight=embed_weight,
                                  output_dim=100, name='data_embed')
    datavec = mx.sym.SliceChannel(data=data_embed,
                                  num_outputs=num_input,
                                  squeeze_axis=1, name='data_slice')
    pred = datavec[0]
    for i in range(1, num_input):
        pred = pred + datavec[i]
    return nce_loss(data=pred,
                    label=label,
                    label_weight=label_weight,
                    embed_weight=embed_weight,
                    vocab_size=vocab_size,
                    num_hidden=100)


def get_subword_net(vocab_size, num_input, embedding_size):
    data = mx.sym.Variable('data')
    mask = mx.sym.Variable('mask')  # use mask to handle variable-length input.
    label = mx.sym.Variable('label')
    label_mask = mx.sym.Variable('label_mask')
    label_weight = mx.sym.Variable('label_weight')
    embed_weight = mx.sym.Variable('embed_weight')

    # Get embedding for one-hot input.
    # get sub-word units input.
    unit_embed = mx.sym.Embedding(data=data, input_dim=vocab_size,
                                  weight=embed_weight,
                                  output_dim=embedding_size)

    # mask embedding_output to get summation of sub-word units'embedding.
    unit_embed = mx.sym.broadcast_mul(lhs=unit_embed, rhs=mask, name='data_units_embed')

    # sum over all these words then you get word-embedding.
    data_embed = mx.sym.sum(unit_embed, axis=2)

    # Slice input equally along specified axis.
    datavec = mx.sym.SliceChannel(data=data_embed,
                                  num_outputs=num_input,
                                  squeeze_axis=1, name='data_slice')
    pred = datavec[0]
    for i in range(1, num_input):
        pred = pred + datavec[i]

    return nce_loss_subwords(data=pred,
                             label=label,
                             label_mask=label_mask,
                             label_weight=label_weight,
                             embed_weight=embed_weight,
                             vocab_size=vocab_size,
                             num_hidden=embedding_size)
