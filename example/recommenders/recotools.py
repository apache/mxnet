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

import mxnet as mx

from negativesample import NegativeSamplingDataIter
import randomproj
import crossentropy

def CosineLoss(a, b, label):
    a = mx.symbol.L2Normalization(a)
    b = mx.symbol.L2Normalization(b)
    dot = a * b
    dot = mx.symbol.sum_axis(dot, axis=1)
    dot = mx.symbol.Flatten(dot)
    cosine = 1 - dot
    return mx.symbol.MAERegressionOutput(data=cosine, label=label)

def SparseRandomProjection(indexes, values, input_dim, output_dim, ngram=1):
    return mx.symbol.Custom(indexes=indexes, values=values, vocab_size=input_dim,
                            output_dim=output_dim, op_type='SparseRandomProjection')

def SparseBagOfWordProjection(data, vocab_size, output_dim, ngram=1):
    return mx.symbol.Custom(indexes=data, vocab_size=vocab_size,
                            output_dim=output_dim, op_type='SparseBOWProj')

def CrossEntropyLoss(data, label):
    return mx.symbol.Custom(data=data, label=label,
                            op_type='CrossEntropyLoss')

