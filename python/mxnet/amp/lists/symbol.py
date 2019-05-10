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

# coding: utf-8
"""Lists of functions whitelisted/blacklisted for automatic mixed precision in symbol API."""

# Functions that should be cast to lower precision
FP16_FUNCS = [
    'Convolution',
    'Deconvolution',
    'FullyConnected',
    'RNN',
    'NormalizedConvolution',
    ]

# Functions that should not be casted, either because
# they are irrelevant (not used in the network itself
# like image transformations or optimizers) or they
# are dtype neutral (can work in both fp16 and fp32)
FP16_FP32_FUNCS = [
    'BNStatsFinalize',
    'BatchNorm',
    'BatchNormAddRelu',
    'BatchNorm_v1',
    'BilinearSampler',
    'BlockGrad',
    'Cast',
    'Crop',
    'CuDNNBatchNorm',
    'Dropout',
    'Embedding',
    'Flatten',
    'FusedOp',
    'GridGenerator',
    'Pad',

    ]

# Functions that have to be cast to FP32 due to possible
# overflows
FP32_FUNCS = [
    'Convolution_v1',
    'IdentityAttachKLSparseReg',
    'arccos',
    'arcsin',
    'cosh',
    'erfinv',
    'sinh',
    'tan',
    'arctanh',

    # Exponents
    'exp',
    'expm1',
    'log',
    'log10',
    'log2',
    'log1p',

    # Powers
    'broadcast_pow',
    'broadcast_power',
    'square',
    'reciprocal',
    'rsqrt',
    'rcbrt',
    '__pow__',
    'pow',
    'linalg_sumlogdiag',
    'hypot',
    'broadcast_hypot',

    # Reductions
    'sum',
    'nansum',
    'prod',
    'nanprod',
    'mean',
    'norm',
    'softmin',

    # Misc
    'gamma',
    'gammaln',
    'linalg_syrk',
    'linalg_potrf',
    'linalg_gemm2',
    'linalg_gelqf',
    'linalg_trmm',
    'linalg_trsm',
    'quantize',
    'quantize_v2',

    # Neural network
    'SoftmaxOutput',
    'softmax',
    'log_softmax',
    'InstanceNorm',
    'LayerNorm',
    'L2Normalization',
    'LRN',
    'SoftmaxActivation',
    'LinearRegressionOutput',
    'LogisticRegressionOutput',
    'MAERegressionOutput',
    'SVMOutput',
    'softmax_cross_entropy',
    'smooth_l1',
    'MakeLoss',
    'make_loss',
    'Custom',
    'CTCLoss',
    'ctc_loss',
    'DeformableConvolution'
    'DeformablePSROIPooling',
    'SyncBatchNorm',
    ]

# Functions that have to be cast to FP32 only for
# some values of their parameters
CONDITIONAL_FP32_FUNCS = [
    ('Activation', 'act_type', ['softrelu']),
    ('LeakyReLU', 'act_type', ['elu', 'selu']),
    ]

# Functions with multiple inputs, that need the same
# type of all their inputs
WIDEST_TYPE_CASTS = [
    '__add__',
    '__sub__',
    '__rsub__',
    '__mul__',
    '__div__',
    '__rdiv__',
    '__mod__',
    '__rmod__',
    '__ne__',
    '__eq__',
    '__gt__',
    '__ge__',
    '__lt__',
    '__le__',
    'Concat',
    'Correlation',
    'ElementWiseSum',
    'add_n',
    'batch_dot',
    'broadcast_add',
    'broadcast_plus',
    'broadcast_div',
    'broadcast_equal',
    'broadcast_greater',
    'broadcast_greater_equal',
    'broadcast_lesser',
    'broadcast_lesser_equal',
    'broadcast_logical_and',
    'broadcast_logical_or',
    'broadcast_logical_xor',
    'broadcast_maximum',
    'broadcast_minimum',
    'broadcast_minus',
    'broadcast_mod',
    'broadcast_mul',
    'broadcast_not_equal',
    'broadcast_sub',
    'dot',
    'elemwise_add',
    'elemwise_div',
    'elemwise_mul',
    'elemwise_sub',
    'stack',
    'maximum',
    'minimum',
    'MultiBoxDetection',
    'MultiBoxTarget',
    'MultiProposal',
    'PSROIPooling',
    'Proposal',
    'ROIAlign',
    'boolean_mask',
    'box_iou',
    'count_sketch',
    'dgl_csr_neighbor_non_uniform_sample',
    'dgl_csr_neighbor_uniform_sample',
    'dgl_graph_compact',
    'dgl_subgraph',
    'edge_id',
    'where',
    '_rnn_concat_param',
    ]

LOSS_OUTPUT_FUNCTIONS = [
    'SoftmaxOutput',
    'LinearRegressionOutput',
    'LogisticRegressionOutput',
    'MAERegressionOutput',
    ]
