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

# For reduction operators
# NOTE: Data used is DEFAULT_DATA
DEFAULT_AXIS_SHAPE = [(), 0, (0, 1)]

# For sorting and searching operators
# NOTE: Data used is DEFAULT_DATA
DEFAULT_AXIS = [0]

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
DEFAULT_DELTA = [(1024, 1024), (10000, 1), (10000, 100)]
DEFAULT_LRS = [(0.1,0.1)]
DEFAULT_LR = [0.1,0.5,0.9]
DEFAULT_GAMMA_1 = [0.1,0.5,0.9]
DEFAULT_GAMMA_2 = [0.1,0.5,0.9]
DEFAULT_EPSILON = [1e-08]
DEFAULT_BETA_1 = [0.1,0.5,0.9]
DEFAULT_BETA_2 = [0.1,0.5,0.9]
DEFAULT_T = [1,5]
DEFAULT_RESCALE_GRAD = [0.4, 0.77]
DEFAULT_CLIP_GRADIENT = [-1.0,0.8]
DEFAULT_CLIP_WEIGHTS = [-1.0,0.8]
DEFAULT_LAZY_UPDATE = [0,1]

# For rearrange operators
# NOTE: Data needs to be a 4D tensor for  operators like space_to_depth and depth_to_space
# Hence below we append 4d to mark the difference.
# For depth_to_space, dimension 3 needs to be a multiple of 'block' and 1 should be a multiple of `block^2`
DEFAULT_DATA_4d = [(1, 4, 2, 4), (10,25,10,100)]
DEFAULT_BLOCK_SIZE = [2, 5]

# For swapaxis operator
DEFAULT_DIM_1 = [0]
DEFAULT_DIM_2 = [1]

# Default Inputs. MXNet Op Param Name to Default Input mapping
DEFAULTS_INPUTS = {"data": DEFAULT_DATA,
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
                   "weight" : DEFAULT_WEIGHT,
                   "weight32" : DEFAULT_WEIGHT,
                   "grad" : DEFAULT_GRAD,
                   "mean" : DEFAULT_MEAN,
                   "var" : DEFAULT_VAR,
                   "mom" : DEFAULT_MOM,
                   "n" : DEFAULT_N,
                   "d" : DEFAULT_D,
                   "v" : DEFAULT_V,
                   "z" : DEFAULT_Z,
                   "g" : DEFAULT_G,
                   "delta" : DEFAULT_DELTA,
                   "lr" : DEFAULT_LR,
                   "lrs" : DEFAULT_LRS,
                   "wds" : DEFAULT_LRS,
                   "gamma1" : DEFAULT_GAMMA_1,
                   "gamma2" : DEFAULT_GAMMA_2,
                   "epsilon" : DEFAULT_EPSILON,
                   "beta1" : DEFAULT_BETA_1,
                   "beta2" : DEFAULT_BETA_2,
                   "t" : DEFAULT_T,
                   "rescale_grad" : DEFAULT_RESCALE_GRAD,
                   "clip_grad" : DEFAULT_CLIP_GRADIENT,
                   "lazy_update" : DEFAULT_LAZY_UPDATE,
                   "data_4d": DEFAULT_DATA_4d,
                   "dim1": DEFAULT_DIM_1,
                   "dim2": DEFAULT_DIM_2,
                   "block_size": DEFAULT_BLOCK_SIZE,
                   "args": DEFAULT_ARGS}


# These are names of MXNet operator parameters that is of type NDArray.
# We maintain this list to automatically recognize these parameters are to be
# given as NDArray and translate users inputs such as a shape tuple, Numpy Array or
# a list to MXNet NDArray. This is just a convenience added so benchmark utility users
# can just say shape of the tensor, and we automatically create Tensors.
PARAMS_OF_TYPE_NDARRAY = ["lhs", "rhs", "data", "base", "exp", "sample",
                          "mu", "sigma", "lam", "alpha", "beta", "gamma", "k", "p",
                          "low", "high", "weight", "bias", "moving_mean", "moving_var",
                          "weight", "weight32", "grad", "mean", "var", "mom", "n", "d",
                          "v", "z", "g", "delta", "args"]
