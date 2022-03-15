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
BF16_FUNCS = [
    'Convolution',
    'FullyConnected',
    ]

# Functions that should not be casted, either because
# they are irrelevant (not used in the network itself
# like image transformations or optimizers) or they
# are dtype neutral (can work in both bf16 and fp32)
BF16_FP32_FUNCS = [
    'abs',
    '_add',
    'BatchNorm',
    'BatchNormWithReLU',
    'clip',
    'Concat',
    'concat',
    'LRN',
    'Pooling',
    'relu',
    'shuffle',
    '_shuffle',
    'sqrt',
    'square',
    'tanh',
    ]

# Functions that when running with Bfloat16, the params that still need float32.
BF16_USE_FP32_PARAMS = {
    'BatchNormWithReLU': ["", "gamma", "beta", "moving_mean", "moving_var"],
    'BatchNorm': ["", "gamma", "beta", "moving_mean", "moving_var"],
}

# Functions that have to be cast to FP32 due to possible
# overflows
FP32_FUNCS = [
    'Deconvolution',
    'RNN',
    'BilinearSampler',
    'BlockGrad',
    'Cast',
    'cast',
    'cast_storage',
    'Crop',
    'Dropout',
    'Embedding',
    '_sparse_Embedding',
    '_sparse_FullyConnected',
    'Flatten',
    'GridGenerator',
    'Pad',
    'Pooling_v1',
    'ROIPooling',
    'Reshape',
    'SequenceLast',
    'SequenceMask',
    'SequenceReverse',
    'SliceChannel',
    'SpatialTransformer',
    'SwapAxis',
    'UpSampling',
    '_CachedOp',
    '_CrossDeviceCopy',
    '_CustomFunction',
    '_DivScalar',
    '_EqualScalar',
    '_GreaterScalar',
    '_GreaterEqualScalar',
    '_LesserScalar',
    '_LesserEqualScalar',
    '_LogicalAndScalar',
    '_LogicalOrScalar',
    '_LogicalXorScalar',
    '_MaximumScalar',
    '_MinimumScalar',
    '_MinusScalar',
    '_ModScalar',
    '_MulScalar',
    '_NoGradient',
    '_NotEqualScalar',
    '_PlusScalar',
    '_RMinusScalar',
    '_RModScalar',
    '_adamw_update',
    '_arange',
    '_broadcast_backward',
    '_cond',
    '_contrib_AdaptiveAvgPooling2D',
    '_contrib_BilinearResize2D',
    '_contrib_bipartite_matching',
    '_contrib_dequantize',
    '_contrib_div_sqrt_dim',
    '_contrib_boolean_mask',
    '_contrib_getnnz',
    '_contrib_gradientmultiplier',
    '_contrib_group_adagrad_update',
    '_contrib_index_array',
    '_contrib_index_copy',
    '_contrib_quadratic',
    '_contrib_quantize',
    '_contrib_quantize_asym',
    '_contrib_quantize_v2',
    '_contrib_quantized_concat',
    '_contrib_quantized_conv',
    '_contrib_quantized_flatten',
    '_contrib_quantized_fully_connected',
    '_contrib_quantized_pooling',
    '_contrib_quantized_elemwise_add',
    '_contrib_quantized_act',
    '_contrib_quantized_rnn',
    '_image_crop',
    '_linspace',
    '_contrib_requantize',
    '_copy',
    '_copyto',
    '_crop_assign',
    '_crop_assign_scalar',
    '_cvcopyMakeBorder',
    '_cvimdecode',
    '_cvimread',
    '_cvimresize',
    '_div_scalar',
    '_equal_scalar',
    '_eye',
    '_foreach',
    '_while_loop',
    '_full',
    '_grad_add',
    '_greater_scalar',
    '_greater_equal_scalar',
    '_histogram',
    '_identity_with_attr_like_rhs',
    '_image_adjust_lighting',
    '_image_flip_left_right',
    '_image_flip_top_bottom',
    '_image_normalize',
    '_image_random_brightness',
    '_image_random_color_jitter',
    '_image_random_contrast',
    '_image_random_flip_left_right',
    '_image_random_flip_top_bottom',
    '_image_random_hue',
    '_image_random_lighting',
    '_image_random_saturation',
    '_image_resize',
    '_image_to_tensor',
    '_imdecode',
    '_lesser_scalar',
    '_lesser_equal_scalar',
    '_logical_and_scalar',
    '_logical_or_scalar',
    '_logical_xor_scalar',
    '_maximum_scalar',
    '_minimum_scalar',
    '_minus_scalar',
    '_mod_scalar',
    '_mp_adamw_update',
    '_mul_scalar',
    '_not_equal_scalar',
    '_onehot_encode',
    '_ones',
    '_plus_scalar',
    '_random_exponential',
    '_random_exponential_like',
    '_random_gamma',
    '_random_gamma_like',
    '_random_generalized_negative_binomial',
    '_random_generalized_negative_binomial_like',
    '_random_negative_binomial',
    '_random_negative_binomial_like',
    '_random_normal',
    '_random_normal_like',
    '_random_poisson',
    '_random_poisson_like',
    '_random_randint',
    '_random_uniform',
    '_random_uniform_like',
    '_ravel_multi_index',
    '_rminus_scalar',
    '_rmod_scalar',
    '_rnn_param_concat',
    '_sample_exponential',
    '_sample_gamma',
    '_sample_generalized_negative_binomial',
    '_sample_multinomial',
    '_sample_negative_binomial',
    '_sample_normal',
    '_sample_poisson',
    '_sample_uniform',
    '_sample_unique_zipfian',
    '_scatter_minus_scalar',
    '_scatter_plus_scalar',
    '_scatter_set_nd',
    '_set_value',
    '_slice_assign',
    '_slice_assign_scalar',
    '_sparse_abs',
    '_sparse_adagrad_update',
    '_sparse_adam_update',
    '_sparse_arccosh',
    '_sparse_arcsinh',
    '_sparse_arctan',
    '_sparse_cast_storage',
    '_sparse_cbrt',
    '_sparse_ceil',
    '_sparse_clip',
    '_sparse_concat',
    '_sparse_cos',
    '_sparse_degrees',
    '_sparse_fix',
    '_sparse_floor',
    '_sparse_ftrl_update',
    '_sparse_negative',
    '_sparse_radians',
    '_sparse_relu',
    '_sparse_retain',
    '_sparse_rint',
    '_sparse_round',
    '_sparse_sgd_mom_update',
    '_sparse_sgd_update',
    '_sparse_sigmoid',
    '_sparse_sign',
    '_sparse_sin',
    '_sparse_sinh',
    '_sparse_slice',
    '_sparse_sqrt',
    '_sparse_stop_gradient',
    '_sparse_tanh',
    '_sparse_trunc',
    '_sparse_zeros_like',
    '_split_v2',
    '_split_v2_backward',
    '_unravel_index',
    '_zeros',
    '_zeros_without_dtype',
    'adam_update',
    'all_finite',
    # 'amp_cast',
    # 'amp_multicast',
    'arccosh',
    'arcsinh',
    'arctan',
    'argmax',
    'argmax_channel',
    'argmin',
    'batch_take',
    'broadcast_axes',
    'broadcast_axis',
    'broadcast_like',
    'broadcast_to',
    'cbrt',
    'ceil',
    'choose_element_0index',
    'cos',
    'crop',
    'degrees',
    'depth_to_space',
    'diag',
    'erf',
    'expand_dims',
    'fill_element_0index',
    'fix',
    'flatten',
    'flip',
    'floor',
    'ftml_update',
    'ftrl_update',
    'gather_nd',
    'hard_sigmoid',
    'identity',
    'logical_not',
    'log_sigmoid'
    'max_axis',
    'max',
    'min',
    'min_axis',
    'mish',
    'mp_sgd_mom_update',
    'mp_sgd_update',
    'multi_all_finite',
    'multi_mp_sgd_mom_update',
    'multi_mp_sgd_update',
    'multi_sgd_mom_update',
    'multi_sgd_update',
    'negative',
    'normal',
    'one_hot',
    'ones_like',
    'pad',
    'pick',
    'radians',
    'random_exponential',
    'random_gamma',
    'random_generalized_negative_binomial',
    'random_negative_binomial',
    'random_normal',
    'random_poisson',
    'random_randint',
    'random_uniform',
    'ravel_multi_index',
    'repeat',
    'reshape',
    'reshape_like',
    'reverse',
    'rint',
    'rmsprop_update',
    'rmspropalex_update',
    'round',
    'sample_exponential',
    'sample_gamma',
    'sample_generalized_negative_binomial',
    'sample_multinomial',
    'sample_negative_binomial',
    'sample_normal',
    'sample_poisson',
    'sample_uniform',
    'scatter_nd',
    'sgd_mom_update',
    'sgd_update',
    'shape_array',
    'sigmoid',
    'sign',
    'signsgd_update',
    'signum_update',
    'sin',
    'size_array',
    'slice',
    'slice_axis',
    'slice_like',
    'softsign',
    'sort',
    'space_to_depth',
    'split',
    'squeeze',
    'stop_gradient',
    'swapaxes',
    'take',
    'tile',
    'transpose',
    'trunc',
    'uniform',
    'unravel_index',
    'zeros_like',
    '_sg_onednn_conv',
    '_sg_onednn_fully_connected',
    '_sg_onednn_batch_dot',
    'broadcast_mul',
    'Convolution_v1',
    'IdentityAttachKLSparseReg',
    'arccos',
    '_sparse_arccos',
    'arcsin',
    'cosh',
    '_sparse_cosh',
    'erfinv',
    'sinh',
    'tan',
    '_sparse_tan',
    'arctanh',
    '_sparse_arcsin',
    '_sparse_arctanh',

    # Exponents
    'exp',
    'expm1',
    '_sparse_exp',
    '_sparse_expm1',
    'log',
    'log10',
    'log2',
    'log1p',

    # Powers
    'broadcast_power',
    '_sparse_square',
    'reciprocal',
    '_RDivScalar',
    '_rdiv_scalar',
    'rsqrt',
    'rcbrt',
    '_Power',
    '_PowerScalar',
    '_power',
    '_power_scalar',
    '_RPowerScalar',
    '_rpower_scalar',
    'linalg_sumlogdiag',
    '_Hypot',
    '_HypotScalar',
    '_hypot',
    '_hypot_scalar',
    'broadcast_hypot',
    '_square_sum',
    '_contrib_hawkesll',

    # Reductions
    'sum',
    'sum_axis',
    'nansum',
    'prod',
    'nanprod',
    'mean',
    'norm',
    'softmin',
    'khatri_rao',
    'moments',

    # Misc
    'gamma',
    'gammaln',
    '_linalg_gelqf',
    '_linalg_gemm',
    '_linalg_gemm2',
    '_linalg_potrf',
    '_linalg_potri',
    '_linalg_sumlogdiag',
    '_linalg_syevd',
    '_linalg_syrk',
    '_linalg_trmm',
    '_linalg_trsm',
    '_linalg_makediag',
    '_linalg_extractdiag',
    '_linalg_maketrian',
    '_linalg_extracttrian',
    '_linalg_inverse',
    '_linalg_det',
    '_linalg_slogdet',
    'linalg_syrk',
    'linalg_potrf',
    'linalg_potri',
    'linalg_gemm2',
    'linalg_gemm',
    'linalg_gelqf',
    'linalg_trmm',
    'linalg_trsm',
    'linalg_makediag',
    'linalg_extractdiag',
    'linalg_maketrian',
    'linalg_extracttrian',
    'linalg_inverse',
    'linalg_det',
    'linalg_slogdet',
    '_NDArray',
    '_Native',
    '_contrib_count_sketch',
    '_contrib_SyncBatchNorm',
    '_contrib_fft',
    '_sparse_gamma',
    '_sparse_gammaln',
    '_sparse_log',
    '_sparse_log10',
    '_sparse_log1p',
    '_sparse_log2',
    '_sparse_make_loss',
    '_sparse_mean',
    '_sparse_norm',
    '_sparse_rsqrt',
    'argsort',
    'topk',

    # Neural network
    'softmax',
    'Softmax',
    'log_softmax',
    'masked_softmax',
    'masked_log_softmax',
    'InstanceNorm',
    'LayerNorm',
    'GroupNorm',
    'L2Normalization',
    'SoftmaxActivation',
    'softmax_cross_entropy',
    'smooth_l1',
    'MakeLoss',
    'make_loss',
    'Custom',
    'CTCLoss',
    '_contrib_CTCLoss',
    '_contrib_ctc_loss',
    'ctc_loss',
    '_npx_deformable_convolution',
    '_contrib_DeformablePSROIPooling',
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
    '_Plus',
    '_plus',
    '_Minus',
    '_sub',
    '_Mul',
    '_Div',
    '_div',
    '_scatter_elemwise_div',
    '_Mod',
    '_Not_Equal',
    '_Equal',
    '_equal',
    '_Greater',
    '_greater',
    '_Greater_Equal',
    '_greater_equal',
    '_Lesser',
    '_Lesser_Equal',
    '_lesser',
    '_lesser_equal',
    '_Logical_And',
    '_Logical_Or',
    '_Logical_Xor',
    '_logical_and',
    '_logical_or',
    '_logical_xor',
    '_maximum',
    '_minimum',
    '_minus',
    '_mod',
    '_mul',
    '_not_equal',
    'Correlation',
    'ElementWiseSum',
    '_sparse_ElementWiseSum',
    'add_n',
    '_sparse_add_n',
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
    'broadcast_not_equal',
    'broadcast_sub',
    'dot',
    'elemwise_add',
    'elemwise_div',
    'elemwise_mul',
    'elemwise_sub',
    'stack',
    '_Maximum',
    '_Minimum',
    '_contrib_MultiBoxDetection',
    '_contrib_MultiBoxPrior',
    '_contrib_MultiBoxTarget',
    '_contrib_MultiProposal',
    '_contrib_PSROIPooling',
    '_contrib_Proposal',
    '_contrib_ROIAlign',
    '_contrib_box_iou',
    '_contrib_box_nms',
    '_contrib_box_non_maximum_suppression',
    '_contrib_dgl_adjacency',
    '_contrib_dgl_csr_neighbor_non_uniform_sample',
    '_contrib_dgl_csr_neighbor_uniform_sample',
    '_contrib_dgl_graph_compact',
    '_contrib_dgl_subgraph',
    '_contrib_edge_id',
    'where',
    '_sparse_where',
    '_sparse_broadcast_add',
    '_sparse_broadcast_div',
    '_sparse_broadcast_minus',
    '_sparse_broadcast_mul',
    '_sparse_broadcast_plus',
    '_sparse_broadcast_sub',
    '_sparse_dot',
    '_sparse_elemwise_add',
    '_sparse_elemwise_div',
    '_sparse_elemwise_mul',
    '_sparse_elemwise_sub',
    '_sparse_sum',

    'random_pdf_gamma',
    'random_pdf_exponential',
    'random_pdf_uniform',
    'random_pdf_negative_binomial',
    'random_pdf_generalized_negative_binomial',
    'random_pdf_dirichlet',
    'random_pdf_normal',
    'random_pdf_poisson',
    '_random_pdf_gamma',
    '_random_pdf_exponential',
    '_random_pdf_uniform',
    '_random_pdf_negative_binomial',
    '_random_pdf_generalized_negative_binomial',
    '_random_pdf_dirichlet',
    '_random_pdf_normal',
    '_random_pdf_poisson',
    ]

LOSS_OUTPUT_FUNCTIONS = [
    ]
