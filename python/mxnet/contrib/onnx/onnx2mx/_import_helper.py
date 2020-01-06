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

# coding: utf-8_
# pylint: disable=invalid-name
"""Operator attributes conversion"""
from ._op_translations import identity, random_uniform, random_normal, sample_multinomial
from ._op_translations import add, subtract, multiply, divide, absolute, negative, add_n
from ._op_translations import tanh, arccos, arcsin, arctan, _cos, _sin, _tan
from ._op_translations import softplus, shape, gather, lp_pooling, size
from ._op_translations import ceil, floor, hardsigmoid, global_lppooling
from ._op_translations import concat, hardmax, topk
from ._op_translations import leaky_relu, _elu, _prelu, _selu, softmax, fully_connected
from ._op_translations import global_avgpooling, global_maxpooling, linalg_gemm
from ._op_translations import sigmoid, pad, relu, matrix_multiplication, batch_norm
from ._op_translations import dropout, local_response_norm, conv, deconv
from ._op_translations import reshape, cast, split, _slice, transpose, squeeze, flatten
from ._op_translations import reciprocal, squareroot, power, exponent, _log, unsqueeze
from ._op_translations import reduce_max, reduce_mean, reduce_min, reduce_sum
from ._op_translations import reduce_prod, avg_pooling, max_pooling, instance_norm
from ._op_translations import argmax, argmin, maximum, minimum
from ._op_translations import clip, reduce_log_sum, reduce_log_sum_exp
from ._op_translations import reduce_sum_square, reduce_l1, reduce_l2, max_roi_pooling
from ._op_translations import log_softmax, softsign, lesser, greater, equal
from ._op_translations import logical_and, logical_or, logical_xor, logical_not
from ._op_translations import mean, depthtospace, spacetodepth, lpnormalization

# convert_map defines maps of ONNX operator names to converter functor(callable)
# defined in the op_translations module.
_convert_map = {
    # Generator Functions
    'Constant'          : identity,
    'RandomUniform'     : random_uniform,
    'RandomNormal'      : random_normal,
    'RandomUniformLike' : random_uniform,
    'RandomNormalLike'  : random_normal,
    'Multinomial'       : sample_multinomial,
    # Arithmetic Operators
    'Add'               : add,
    'Sub'               : subtract,
    'Mul'               : multiply,
    'Div'               : divide,
    'Abs'               : absolute,
    'Neg'               : negative,
    'Sum'               : add_n, #elemwise sum
    #Hyperbolic functions
    'Tanh'              : tanh,
    # Rounding
    'Ceil'              : ceil,
    'Floor'             : floor,
    # Joining and spliting
    'Concat'            : concat,
    # Basic neural network functions
    'Sigmoid'           : sigmoid,
    'Relu'              : relu,
    'Pad'               : pad,
    'MatMul'            : matrix_multiplication, #linalg_gemm2
    'Conv'              : conv,
    'ConvTranspose'     : deconv,
    'BatchNormalization': batch_norm,
    'SpatialBN'         : batch_norm,
    'LeakyRelu'         : leaky_relu,
    'Elu'               : _elu,
    'PRelu'             : _prelu,
    'Selu'              : _selu,
    'Softmax'           : softmax,
    'FC'                : fully_connected,
    'GlobalAveragePool' : global_avgpooling,
    'GlobalMaxPool'     : global_maxpooling,
    'GlobalLpPool'      : global_lppooling,
    'Gemm'              : linalg_gemm,
    'LRN'               : local_response_norm,
    'Dropout'           : dropout,
    # Changing shape and type.
    'Reshape'           : reshape,
    'Cast'              : cast,
    'Split'             : split,
    'Slice'             : _slice,
    'Transpose'         : transpose,
    'Squeeze'           : squeeze,
    'Unsqueeze'         : unsqueeze,
    'Flatten'           : flatten,
    'Identity'          : identity,
    #Powers
    'Reciprocal'        : reciprocal,
    'Sqrt'              : squareroot,
    'Pow'               : power,
    'Exp'               : exponent,
    'Log'               : _log,
    # Reduce Functions
    'ReduceMax'         : reduce_max,
    'ReduceMean'        : reduce_mean,
    'ReduceMin'         : reduce_min,
    'ReduceSum'         : reduce_sum,
    'ReduceProd'        : reduce_prod,
    'AveragePool'       : avg_pooling,
    'MaxPool'           : max_pooling,
    # Sorting and Searching
    'ArgMax'            : argmax,
    'ArgMin'            : argmin,
    'Max'               : maximum,
    'Min'               : minimum,
    'Clip'              : clip,
    'ReduceLogSum'      : reduce_log_sum,
    'ReduceLogSumExp'   : reduce_log_sum_exp,
    'ReduceSumSquare'   : reduce_sum_square,
    'ReduceL1'          : reduce_l1,
    'ReduceL2'          : reduce_l2,
    'MaxRoiPool'        : max_roi_pooling,
    'InstanceNormalization' : instance_norm,
    'LogSoftmax'        : log_softmax,
    'Softsign'          : softsign,
    'Less'              : lesser,
    'Greater'           : greater,
    'Equal'             : equal,
    'And'               : logical_and,
    'Xor'               : logical_xor,
    'Not'               : logical_not,
    'Or'                : logical_or,
    'Mean'              : mean,
    'Acos'              : arccos,
    'Asin'              : arcsin,
    'Atan'              : arctan,
    'Cos'               : _cos,
    'Sin'               : _sin,
    'Softplus'          : softplus,
    'Tan'               : _tan,
    'Shape'             : shape,
    'Size'              : size,
    'Gather'            : gather,
    'HardSigmoid'       : hardsigmoid,
    'LpPool'            : lp_pooling,
    'DepthToSpace'      : depthtospace,
    'SpaceToDepth'      : spacetodepth,
    'Hardmax'           : hardmax,
    'LpNormalization'   : lpnormalization,
    'TopK'              : topk
}
