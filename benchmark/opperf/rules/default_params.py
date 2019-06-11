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

"""Default Input Tensor shapes to use for benchmarking"""

""""""

# For Unary operators like abs, arccos, arcsin etc..
DEFAULT_DATA = [(1024, 1024), (10000, 1), (10000, 100)]

# For Binary broadcast operators like - broadcast_add/sub/mode/logical_and etc..
DEFAULT_LHS = [[(1024, 1024), (10000, 10), (10000, 1)]]
DEFAULT_RHS = [[(1024, 1024), (10000, 10), (10000, 1)]]

# For operators like - random_uniform, random_normal etc..
DEFAULT_SHAPE = [(1024, 1024), (10000, 1), (10000, 100)]
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
DEFAULT_AXIS = [(), 0, (0, 1)]

# Default Inputs. MXNet Op Param Name to Default Input mapping
DEFAULTS_INPUTS = {"data": DEFAULT_DATA,
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
                   "axis": DEFAULT_AXIS}

# These are names of MXNet operator parameters that is of type NDArray.
# We maintain this list to automatically recognize these parameters are to be
# given as NDArray and translate users inputs such as a shape tuple, Numpy Array or
# a list to MXNet NDArray. This is just a convenience added so benchmark utility users
# can just say shape of the tensor, and we automatically create Tensors.
PARAMS_OF_TYPE_NDARRAY = ["lhs", "rhs", "data", "base", "exp",
                          "mu", "sigma", "lam", "alpha", "beta", "gamma", "k", "p",
                          "low", "high", "weight", "bias", "moving_mean", "moving_var"]
