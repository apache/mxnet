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
import sys

# We will use all operators inside NDArray Module
# If you want to run benchmark for all operators in different namespace,
# for example mxnet.numpy.op, update here. All operators for benchmarks
# will be picked up from this module
MX_OP_MODULE = sys.modules["mxnet.ndarray.op"]


"""Default Input Tensor shapes to use for benchmarking"""

# For operators like concat, ElementWiseSum, squeeze, stack
# argument data is passed as variable arg (*args)
DEFAULT_ARGS = [(1024, 1024)]

# For Unary operators like abs, arccos, arcsin etc..
DEFAULT_DATA = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_DTYPE = ['float32', 'int32', 'float32']  # required parameter for amp_cast, cast
DEFAULT_DTYPE_INT = ['int32', 'int64', 'int32']  # randint works for int* types only
DEFAULT_DTYPE_FLOAT = ['float16', 'float32', 'float64']  # random_exp works for float* types only

# For Binary miscellaneous operators like choose_element0_index
# argument data must be indexed via an NDArray.
# NOTE: Data used is DEFAULT_DATA
DEFAULT_INDEX = [(1, 1024), (1, 1), (1, 100)]

# For Binary broadcast operators like - broadcast_add/sub/mod/logical_and etc..
DEFAULT_LHS = [(1024, 1024), (10000, 10), (10000, 1)]
DEFAULT_RHS = [(1024, 1024), (10000, 10), (10000, 1)]

# For operators like - random_uniform, random_normal etc..
DEFAULT_SHAPE = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_SAMPLE = [(2,)]
DEFAULT_LOW = [0]
DEFAULT_HIGH = [5]
DEFAULT_K = [1]
DEFAULT_P = [1]

# For operators like - sample_uniform, sample_normal etc..
# NOTE: There are many overlapping operators in random_* and sample_*,
# Ex: random_uniform, sample_uniform. Parameter names are same, but, for
# random_* operators they are float/int and for sample_* operators they are NDArray.
# Hence, below we append ND to mark the difference.
DEFAULT_LOW_ND = [[0.0, 2.5]]
DEFAULT_HIGH_ND = [[1.0, 3.7]]
DEFAULT_MU_ND = [[2.0, 2.5]]
DEFAULT_SIGMA = [[1.0, 3.7]]
DEFAULT_ALPHA_ND = [[0.0, 2.5]]
DEFAULT_BETA_ND = [[1.0, 0.7]]
DEFAULT_LAM = [[1.0, 8.5]]
DEFAULT_K_ND = [[20, 49]]
DEFAULT_P_ND = [[0.4, 0.77]]
DEFAULT_GRID = [(32, 2, 256, 256)]
DEFAULT_DATA_BILINEAR = [(32, 2, 256, 256)]
DEFAULT_TRANSFORM_TYPE = ['warp', 'affine']
DEFAULT_DATA_GRIDGEN = [(32, 2, 256, 256), (256, 6)]
DEFAULT_TARGET_SHAPE = [(256, 6)]
DEFAULT_DATA_SM = [(32, 32), (64, 64)]

# For reduction operators
# NOTE: Data used is DEFAULT_DATA
DEFAULT_AXIS_SHAPE = [(), 0, (0, 1)]

# For sorting and searching operators
# NOTE: Data used is DEFAULT_DATA
DEFAULT_AXIS = [0]

# For NN basic operators
# General
DEFAULT_DATA_NN_BASIC = [(32, 3, 256, 256), (32, 3, 10000, 10)]
DEFAULT_NUM_HIDDEN = [64]
DEFAULT_BIAS = [(64,)]
DEFAULT_FLATTEN = [True, False]
DEFAULT_GAMMA = [(3,)]
DEFAULT_BETA = [(3,)]
DEFAULT_MOVING_MEAN = [(3,)]
DEFAULT_MOVING_VAR = [(3,)]
DEFAULT_LABEL_REG = [(32, 3, 256, 256), (32, 3, 10000, 10)]
DEFAULT_GRAD_SCALE = [.5]
DEFAULT_NORMALIZATION = ["batch"]
DEFAULT_MARGIN = [.5]
DEFAULT_REG_COEFF = [.5]
DEFAULT_INPUT_DIM = [3, 16]
DEFAULT_OUTPUT_DIM = [4, 9]
DEFAULT_SPARSE_GRAD = [False]
DEFAULT_KERNEL_SIZE = [3]
DEFAULT_MAX_DISPLACEMENT = [2]
DEFAULT_STRIDE_1 = [2]
DEFAULT_STRIDE_2 = [2]
DEFAULT_ALPHA = [.001]
DEFAULT_NSIZE = [3]
DEFAULT_PARAMETERS = [(7,), (104,)]
DEFAULT_STATE = [(1, 4, 1), (2, 10000, 4)]
DEFAULT_MODE = ["rnn_relu", "rnn_tanh"]
DEFAULT_STATE_SIZE = [1, 4]
DEFAULT_NUM_LAYERS = [1, 2]
DEFAULT_NUM_GROUPS = [1, 10]
DEFAULT_TRANSFORM = ["affine"]
DEFAULT_SAMPLER = ["bilinear"]
DEFAULT_DILATE = [(1,), (1, 1)]
DEFAULT_PAD = [(1,), (1, 1)]
DEFAULT_OUTPUT_SIZE = [(64, 16, 1), (32, 8, 1)]
DEFAULT_KERNEL = [(1, 1, 1), (1, 1, 1)]
DEFAULT_STRIDE = [(2, 2, 2), (1, 1, 1)]

# BatchNorm
DEFAULT_AXIS_BN = [1]

# LayerNorm
DEFAULT_GAMMA_LN = [(32,), (32,)]
DEFAULT_BETA_LN = [(32,), (32,)]

# L2Normalization
DEFAULT_MODE_L2 = ['channel', 'instance', 'spatial']

# SVMOutput
DEFAULT_LABEL_SVM = [(32, 3, 256), (32, 3, 10000)]

# SoftmaxOutput
DEFAULT_LABEL_SM = [(32, 3, 256), (32, 3, 10000)]

# FullyConnected
DEFAULT_WEIGHT_FC = [(64, 3 * 256 * 256), (64, 10)]

# Embedding
DEFAULT_WEIGHT_EMBEDDING = [(3, 4), (16, 9)]

# GroupNorm
DEFAULT_DATA_GN = [(32, 3, 256, 256), (32, 10, 10000, 10)]
DEFAULT_BETA_GAMMA_GN = [(1,), (10,)]

# Dropout
DEFAULT_DATA_DROPOUT = [(32, 3, 256, 256), (10000, 10)]
DEFAULT_MODE_DROPOUT = ["always"]

# SpatialTransformer
DEFAULT_DATA_ST = [(32, 3, 256, 6), (256, 3, 10000, 6)]
DEFAULT_LOC_TAR_ST = [(32, 6), (256, 6)]

# im2col
DEFAULT_KERNEL_I2C = [(3,), (3, 3)]
DEFAULT_STRIDE_I2C = [(1,), (1, 1)]

# col2im
DEFAULT_DATA_C2I = [(32, 64, 256), (32, 64, 256)]

# RNN
DEFAULT_DATA_RNN = [(32, 4, 4), (512, 10000, 10)]
DEFAULT_P_RNN = [.5]

# LRN
DEFAULT_BETA_LRN = [.2]

# For optimizer operators
DEFAULT_WEIGHT = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_GRAD = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_MOM = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_MEAN = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_VAR = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_N = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_D = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_V = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_Z = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_G = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_R1 = [(1, 1024), (1, 1), (1, 100)]
DEFAULT_R2 = [(1, 1024), (1, 1), (1, 100)]
DEFAULT_DELTA = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_LRS = [(0.1, 0.1)]
DEFAULT_LR = [0.1, 0.5, 0.9]
DEFAULT_GAMMA_1 = [0.1, 0.5, 0.9]
DEFAULT_GAMMA_2 = [0.1, 0.5, 0.9]
DEFAULT_EPSILON = [1e-08]
DEFAULT_BETA_1 = [0.1, 0.5, 0.9]
DEFAULT_BETA_2 = [0.1, 0.5, 0.9]
DEFAULT_T = [1, 5]
DEFAULT_RESCALE_GRAD = [0.4, 0.77]
DEFAULT_CLIP_GRADIENT = [-1.0, 0.8]
DEFAULT_CLIP_WEIGHTS = [-1.0, 0.8]
DEFAULT_LAZY_UPDATE = [0, 1]

# For rearrange operators
# NOTE: Data needs to be a 4D tensor for  operators like space_to_depth and depth_to_space
# Hence below we append 4d to mark the difference.
# For depth_to_space, dimension 3 needs to be a multiple of 'block' and 1 should be a multiple of `block^2`
DEFAULT_DATA_4d = [(1, 4, 2, 4), (10, 25, 10, 100)]
DEFAULT_BLOCK_SIZE = [2, 5]

# For miscellaneous operators
DEFAULT_DATA_SQUEEZE = [(1, 1024, 1024), (32, 1, 256, 256)]
DEFAULT_AXIS_SQUEEZE = [0, 1]
DEFAULT_A_MIN = [0.1]
DEFAULT_A_MAX = [0.9]
DEFAULT_LRS = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_WSS = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_GSS = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_WDS = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_ETA = [.5]
DEFAULT_STYPE = ['default', 'csr', 'row_sparse']
DEFAULT_A = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_LHS_FEI = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_MHS = [(1024,), (10000,), (10000,)]
DEFAULT_RHS_FEI = [(1024,), (10000,), (10000,)]

# For swapaxis operator
DEFAULT_DIM_1 = [0]
DEFAULT_DIM_2 = [1]

# For indexing routines
DEFAULT_INDEX = [(1,1024), (1,1), (1,100)]
DEFAULT_INDICES = [(1, 1)]
DEFAULT_BEGIN = [0] # slice_axis expects int, slice can have tuple/int
DEFAULT_END =[1] # same as above
DEFAULT_SHAPE_LIKE = [(100, 100), (10, 1), (100, 10)]
DEFAULT_X = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_Y = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_COND = [(1024,), (10000,), (10000,)]
DEFAULT_DEPTH = [0]
# For ravel_multi_index op, ndim(shape) = 2; hence data NDArray's first dim = 2
# First dimension of input of ravel operator should match shape parameter dimension
# DEFAULT_SHAPE is reused for ravel_multi_index op
RAVEL_DATA = [(2, 1024)]

# For loss operators
DEFAULT_DATA_3d = [(1024, 100, 100)]
DEFAULT_LABEL = [(100,100)]
DEFAULT_DATA_SMCE = [(1024, 1024)]
DEFAULT_LABEL_SMCE = [(1024,)]
# For NN operators
DEFAULT_ACT_TYPE_LR = ['leaky', 'elu', 'selu', 'gelu']
DEFAULT_ACT_TYPE_ACTIVATION = ['relu', 'sigmoid', 'softrelu', 'softsign', 'tanh']
DEFAULT_LABEL_SOFTMAX = [(1024, 1024), (10000, 1), (10000, 100)]

# For linalg operators
DEFAULT_A = [(1024, 1024)]
DEFAULT_B = [(1024, 1024)]
DEFAULT_C = [(1024, 1024)]
DEFAULT_A_MT = [(1024, 1035)]
DEFAULT_AXES = [[0, 1]]

# Default Inputs. MXNet Op Param Name to Default Input mapping
DEFAULTS_INPUTS = {"data": DEFAULT_DATA,
                   "dtype": DEFAULT_DTYPE,
                   "dtype_int": DEFAULT_DTYPE_INT,
                   "dtype_float": DEFAULT_DTYPE_FLOAT,
                   "sample": DEFAULT_SAMPLE,
                   "lhs": DEFAULT_LHS,
                   "rhs": DEFAULT_RHS,
                   "shape": DEFAULT_SHAPE,
                   "low": DEFAULT_LOW,
                   "high": DEFAULT_HIGH,
                   "low_nd": DEFAULT_LOW_ND,
                   "high_nd": DEFAULT_HIGH_ND,
                   "mu_nd": DEFAULT_MU_ND,
                   "sigma": DEFAULT_SIGMA,
                   "alpha_nd": DEFAULT_ALPHA_ND,
                   "beta_nd": DEFAULT_BETA_ND,
                   "lam_nd": DEFAULT_LAM,
                   "k": DEFAULT_K,
                   "p": DEFAULT_P,
                   "k_nd": DEFAULT_K_ND,
                   "p_nd": DEFAULT_P_ND,
                   "axis_shape": DEFAULT_AXIS_SHAPE,
                   "axis": DEFAULT_AXIS,
                   "weight": DEFAULT_WEIGHT,
                   "weight32": DEFAULT_WEIGHT,
                   "grad": DEFAULT_GRAD,
                   "mean": DEFAULT_MEAN,
                   "var": DEFAULT_VAR,
                   "mom": DEFAULT_MOM,
                   "r1": DEFAULT_R1,
                   "r2": DEFAULT_R2,
                   "n": DEFAULT_N,
                   "d": DEFAULT_D,
                   "v": DEFAULT_V,
                   "z": DEFAULT_Z,
                   "g": DEFAULT_G,
                   "delta": DEFAULT_DELTA,
                   "lr": DEFAULT_LR,
                   "lrs": DEFAULT_LRS,
                   "wds": DEFAULT_LRS,
                   "wd": DEFAULT_LR,
                   "gamma1": DEFAULT_GAMMA_1,
                   "gamma2": DEFAULT_GAMMA_2,
                   "epsilon": DEFAULT_EPSILON,
                   "beta1": DEFAULT_BETA_1,
                   "beta2": DEFAULT_BETA_2,
                   "t": DEFAULT_T,
                   "rescale_grad": DEFAULT_RESCALE_GRAD,
                   "clip_grad": DEFAULT_CLIP_GRADIENT,
                   "lazy_update": DEFAULT_LAZY_UPDATE,
                   "data_4d": DEFAULT_DATA_4d,
                   "dim1": DEFAULT_DIM_1,
                   "dim2": DEFAULT_DIM_2,
                   "block_size": DEFAULT_BLOCK_SIZE,
                   "args": DEFAULT_ARGS,
                   "a": DEFAULT_DATA,
                   "index": DEFAULT_INDEX,
                   "indices": DEFAULT_INDICES,
                   "begin": DEFAULT_BEGIN,
                   "end": DEFAULT_END,
                   "shape_like": DEFAULT_SHAPE_LIKE,
                   "x": DEFAULT_X,
                   "y": DEFAULT_Y,
                   "condition": DEFAULT_COND,
                   "depth": DEFAULT_DEPTH,
                   "ravel_data": RAVEL_DATA,
                   "data_smce": DEFAULT_DATA_SMCE,
                   "data_3d": DEFAULT_DATA_3d,
                   "label_smce": DEFAULT_LABEL_SMCE,
                   "label": DEFAULT_LABEL,
                   "index": DEFAULT_INDEX,
                   "grid": DEFAULT_GRID,
                   "data_bilinearsampler": DEFAULT_DATA_BILINEAR,
                   "transform_type": DEFAULT_TRANSFORM_TYPE,
                   "data_gridgenerator": DEFAULT_DATA_GRIDGEN,
                   "target_shape_gridgenerator": DEFAULT_TARGET_SHAPE,
                   "data_sample_multinomial": DEFAULT_DATA_SM,
                   "A": DEFAULT_A,
                   "B": DEFAULT_B,
                   "C": DEFAULT_C,
                   "A_linalg_maketrian": DEFAULT_A_MT,
                   "axes": DEFAULT_AXES,
                   "act_type_leakyrelu": DEFAULT_ACT_TYPE_LR,
                   "label_softmax": DEFAULT_LABEL_SOFTMAX,
                   "act_type_activation": DEFAULT_ACT_TYPE_ACTIVATION,
                   "data_squeeze": DEFAULT_DATA_SQUEEZE,
                   "axis_squeeze": DEFAULT_AXIS_SQUEEZE,
                   "a_min": DEFAULT_A_MIN,
                   "a_max": DEFAULT_A_MAX,
                   "lrs": DEFAULT_LRS,
                   "weights_sum_sq": DEFAULT_WSS,
                   "grads_sum_sq": DEFAULT_GSS,
                   "wds": DEFAULT_WDS,
                   "eta": DEFAULT_ETA,
                   "eps": DEFAULT_EPSILON,
                   "stype": DEFAULT_STYPE,
                   "a": DEFAULT_A,
                   "lhs_fill_element_0index": DEFAULT_LHS_FEI,
                   "rhs_fill_element_0index": DEFAULT_RHS_FEI,
                   "mhs": DEFAULT_MHS,
                   "data_spatialtransformer": DEFAULT_DATA_ST,
                   "loc_spatialtransformer": DEFAULT_LOC_TAR_ST,
                   "target_shape": DEFAULT_LOC_TAR_ST,
                   "transform_type_spatialtransformer": DEFAULT_TRANSFORM,
                   "sampler_type": DEFAULT_SAMPLER,
                   "data_col2im": DEFAULT_DATA_C2I,
                   "output_size": DEFAULT_OUTPUT_SIZE,
                   "kernel_col2im": DEFAULT_KERNEL,
                   "stride_col2im": DEFAULT_STRIDE,
                   "data_rnn": DEFAULT_DATA_RNN,
                   "p_rnn": DEFAULT_P_RNN,
                   "parameters": DEFAULT_PARAMETERS,
                   "state": DEFAULT_STATE,
                   "state_size": DEFAULT_STATE_SIZE,
                   "num_layers": DEFAULT_NUM_LAYERS,
                   "mode_rnn": DEFAULT_MODE,
                   "data_groupnorm": DEFAULT_DATA_GN,
                   "gamma_groupnorm": DEFAULT_BETA_GAMMA_GN,
                   "beta_groupnorm": DEFAULT_BETA_GAMMA_GN,
                   "num_groups": DEFAULT_NUM_GROUPS,
                   "eps": DEFAULT_EPSILON,
                   "data_dropout": DEFAULT_DATA_DROPOUT,
                   "mode_dropout": DEFAULT_MODE_DROPOUT,
                   "p_dropout": DEFAULT_P,
                   "data_nn_basic": DEFAULT_DATA_NN_BASIC,
                   "num_hidden": DEFAULT_NUM_HIDDEN,
                   "data_fullyconnected": DEFAULT_DATA_NN_BASIC,
                   "weight_fullyconnected": DEFAULT_WEIGHT_FC,
                   "weight_embedding": DEFAULT_WEIGHT_EMBEDDING,
                   "bias": DEFAULT_BIAS,
                   "flatten": DEFAULT_FLATTEN,
                   "data_batchnorm": DEFAULT_DATA_NN_BASIC,
                   "gamma_batchnorm": DEFAULT_GAMMA,
                   "beta_batchnorm": DEFAULT_BETA,
                   "moving_mean_batchnorm": DEFAULT_MOVING_MEAN,
                   "moving_var_batchnorm": DEFAULT_MOVING_VAR,
                   "axis_batchnorm": DEFAULT_AXIS_BN,
                   "data_softmaxoutput": DEFAULT_DATA_NN_BASIC,
                   "label_softmaxoutput": DEFAULT_LABEL_SM,
                   "data_maeregressionoutput": DEFAULT_DATA_NN_BASIC,
                   "label_maeregressionoutput": DEFAULT_LABEL_REG,
                   "data_logisticregressionoutput": DEFAULT_DATA_NN_BASIC,
                   "label_logisticregressionoutput": DEFAULT_LABEL_REG,
                   "data_linearregressionoutput": DEFAULT_DATA_NN_BASIC,
                   "label_linearregressionoutput": DEFAULT_LABEL_REG,
                   "data_svmoutput": DEFAULT_DATA_NN_BASIC,
                   "label_svmoutput": DEFAULT_LABEL_SVM,
                   "grad_scale": DEFAULT_GRAD_SCALE,
                   "normalization": DEFAULT_NORMALIZATION,
                   "margin": DEFAULT_MARGIN,
                   "regularization_coefficient": DEFAULT_REG_COEFF,
                   "data_l2normalization": DEFAULT_DATA_NN_BASIC,
                   "mode_l2normalization": DEFAULT_MODE_L2,
                   "gamma_layernorm": DEFAULT_GAMMA_LN,
                   "beta_layernorm": DEFAULT_BETA_LN,
                   "data_instancenorm": DEFAULT_DATA_NN_BASIC,
                   "gamma_instancenorm": DEFAULT_GAMMA,
                   "beta_instancenorm": DEFAULT_BETA,
                   "input_dim": DEFAULT_INPUT_DIM,
                   "output_dim": DEFAULT_OUTPUT_DIM,
                   "sparse_grad": DEFAULT_SPARSE_GRAD,
                   "data1": DEFAULT_DATA_NN_BASIC,
                   "data2": DEFAULT_DATA_NN_BASIC,
                   "kernel_size": DEFAULT_KERNEL_SIZE,
                   "max_displacement": DEFAULT_MAX_DISPLACEMENT,
                   "stride1": DEFAULT_STRIDE_1,
                   "stride2": DEFAULT_STRIDE_2,
                   "data_im2col": DEFAULT_DATA_NN_BASIC,
                   "kernel_im2col": DEFAULT_KERNEL_I2C,
                   "stride_im2col": DEFAULT_STRIDE_I2C,
                   "dilate_im2col": DEFAULT_DILATE,
                   "pad_im2col": DEFAULT_PAD,
                   "data_lrn": DEFAULT_DATA_NN_BASIC,
                   "alpha_lrn": DEFAULT_ALPHA,
                   "beta_lrn": DEFAULT_BETA_LRN,
                   "nsize": DEFAULT_NSIZE,
                   "data_layernorm": DEFAULT_DATA_NN_BASIC,
                   "axis_layernorm": DEFAULT_AXIS}


# These are names of MXNet operator parameters that is of type NDArray.
# We maintain this list to automatically recognize these parameters are to be
# given as NDArray and translate users inputs such as a shape tuple, Numpy Array or
# a list to MXNet NDArray. This is just a convenience added so benchmark utility users
# can just say shape of the tensor, and we automatically create Tensors.
PARAMS_OF_TYPE_NDARRAY = ["lhs", "rhs", "data", "base", "exp", "sample",
                          "mu", "sigma", "lam", "alpha", "beta", "gamma", "k", "p",
                          "low", "high", "weight", "bias", "moving_mean", "moving_var",
                          "weight", "weight32", "grad", "mean", "var", "mom", "n", "d",
                          "v", "z", "g", "delta", "args", "indices", "shape_like", "y",
                          "x", "condition", "a", "index", "raveL_data", "label", "grid",
                          "A", "B", "C", "r1", "r2", "rois", "lrs", "wds", "weights_sum_sq",
                          "grads_sum_sq", "mhs", "data1", "data2", "loc", "parameters", "state"]
