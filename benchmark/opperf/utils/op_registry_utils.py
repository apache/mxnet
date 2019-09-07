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

"""Utilities to interact with MXNet operator registry."""
from operator import itemgetter
from mxnet import runtime
import mxnet as mx

from benchmark.opperf.rules.default_params import DEFAULTS_INPUTS, MX_OP_MODULE

# Operators where parameter have special criteria that cannot be cleanly automated.
# Example: sample_multinomial operator has a parameter 'data'. It expects values to sum up to 1.
unique_ops = ("sample_multinomial",)


def _select_ops(operator_names, filters=("_contrib", "_"), merge_op_forward_backward=True):
    """From a given list of operators, filter out all operator names starting with given filters and prepares
    a dictionary of operator with attributes - 'has_backward' and 'nd_op_handle = mxnet.ndarray.op'

    By default, merge forward and backward operators for a given op into one operator and sets the attribute
    'has_backward' for the operator.

    By default, filter out all Contrib operators that starts with '_contrib' and internal operators that
    starts with '_'.

    Note - All deprecated operators are filtered out as well.

    Parameters
    ----------
    operator_names: List[str]
        List of operator names.
    filters: Tuple(str)
        Tuple of filters to apply on operator names.
    merge_op_forward_backward: Boolean, Default - True
        Merge forward and backward operators for a given op in to one op.

    Returns
    -------
    {"operator_name": {"has_backward", "nd_op_handle"}}
    """
    mx_operators = {}
    operators_with_backward = []

    # Filter out deprecated operators
    filters += ("normal", "uniform", "BatchNorm_v1", "Flatten", "contrib_CTCLoss", "Pad", "Cast",
                "Pooling_v1", "Concat", "Reshape", "Convolution_v1", "SliceChannel", "Crop",
                "crop", "onehot_encode")

    if merge_op_forward_backward:
        filters += ("_backward",)

    for cur_op_name in operator_names:
        if not cur_op_name.startswith(filters):
            mx_operators[cur_op_name] = {"has_backward": False,
                                         "nd_op_handle": getattr(MX_OP_MODULE, cur_op_name)}

        if cur_op_name.startswith("_backward_"):
            operators_with_backward.append(cur_op_name)

    if merge_op_forward_backward:
        # Identify all operators that can run backward.
        for op_with_backward in operators_with_backward:
            op_name = op_with_backward.split("_backward_")[1]
            if op_name in mx_operators:
                mx_operators[op_name]["has_backward"] = True

    return mx_operators


def _set_op_arguments(mx_operators):
    """Fetch and set operator arguments - nargs, arg_names, arg_types
    """
    for op_name in mx_operators:
        operator_arguments = mx.operator.get_operator_arguments(op_name)
        mx_operators[op_name]["params"] = {"narg": operator_arguments.narg,
                                           "arg_names": operator_arguments.names,
                                           "arg_types": operator_arguments.types}


def _get_all_mxnet_operators():
    # Step 1 - Get all registered op names and filter it
    operator_names = mx.operator.get_all_registered_operators()
    mx_operators = _select_ops(operator_names)

    # Step 2 - Get all parameters for the operators
    _set_op_arguments(mx_operators)
    return mx_operators


def prepare_op_inputs(arg_params, arg_values):
    inputs = []

    for arg_value in arg_values:
        inp = {}
        for arg_name in arg_params["params"]["arg_names"]:
            if arg_name in arg_value:
                inp[arg_name] = arg_value[arg_name]
        inputs.append(inp)
    return inputs


def prepare_op_inputs(op, arg_params):
    inputs = []

    # 4d tensor is needed only by following two ops
    ops_4d = ['depth_to_space','space_to_depth']

    # Prepare op to default input mapping
    arg_values = {}
    for arg_name, arg_type in zip(arg_params["params"]["arg_names"],
                                  arg_params["params"]["arg_types"]):
        if "NDArray" in arg_type and arg_name + "_nd" in DEFAULTS_INPUTS:
            arg_values[arg_name] = DEFAULTS_INPUTS[arg_name + "_nd"]
        elif "NDArray" in arg_type and op in ops_4d and arg_name + "_4d" in DEFAULTS_INPUTS:
            arg_values[arg_name] = DEFAULTS_INPUTS[arg_name + "_4d"]
        elif arg_name in DEFAULTS_INPUTS:
            arg_values[arg_name] = DEFAULTS_INPUTS[arg_name]
        elif "float" in arg_type and arg_name + "_float" in DEFAULTS_INPUTS:
            arg_values[arg_name] = DEFAULTS_INPUTS[arg_name + "_float"]
        elif "Shape" in arg_type and arg_name + "_shape" in DEFAULTS_INPUTS:
            # This is for cases where in some ops 'axis' is Int in some ops a shape tuple.
            # Ex: axis in sum is shape, axis in sort is int.
            arg_values[arg_name] = DEFAULTS_INPUTS[arg_name + "_shape"]

    # Number of different inputs we want to use to test
    # the operator
    num_input_combinations = max([len(value) for value in arg_values.values()])

    # Prepare key/value args for param to input value
    for idx in range(num_input_combinations):
        inp = {}
        for arg_name in arg_params["params"]["arg_names"]:
            if arg_name in arg_values:
                if len(arg_values[arg_name]) == num_input_combinations:
                    inp[arg_name] = arg_values[arg_name][idx]
                else:
                    # This is required when we want to use a param same across all
                    # input combination. Example: keeping low and high same for random sampling
                    # operator for all different types of Tensor shape.
                    inp[arg_name] = arg_values[arg_name][0]

        inputs.append(inp)
    return inputs


def get_all_unary_operators():
    """Gets all Unary operators registered with MXNet.

    Returns
    -------
    {"operator_name": {"has_backward", "nd_op_handle", "params"}}
    """
    # Get all mxnet operators
    mx_operators = _get_all_mxnet_operators()

    # Filter for unary broadcast operators
    unary_broadcast_mx_operators = {}
    for op_name, op_params in mx_operators.items():
        if op_params["params"]["narg"] == 1 and \
                "data" in op_params["params"]["arg_names"]:
            unary_broadcast_mx_operators[op_name] = mx_operators[op_name]
    return unary_broadcast_mx_operators


def get_all_broadcast_binary_operators():
    """Gets all binary broadcast operators registered with MXNet.

    Returns
    -------
    {"operator_name": {"has_backward", "nd_op_handle", "params"}}
    """
    # Get all mxnet operators
    mx_operators = _get_all_mxnet_operators()

    # Filter for binary broadcast operators
    binary_broadcast_mx_operators = {}
    for op_name, op_params in mx_operators.items():
        if op_name.startswith("broadcast_") and op_params["params"]["narg"] == 2 and \
                "lhs" in op_params["params"]["arg_names"] and \
                "rhs" in op_params["params"]["arg_names"]:
            binary_broadcast_mx_operators[op_name] = mx_operators[op_name]
    return binary_broadcast_mx_operators


def get_all_elemen_wise_binary_operators():
    """Gets all binary elemen_wise operators registered with MXNet.

    Returns
    -------
    {"operator_name": {"has_backward", "nd_op_handle", "params"}}
    """
    # Get all mxnet operators
    mx_operators = _get_all_mxnet_operators()

    # Filter for binary elemen_wise operators
    binary_elemen_wise_mx_operators = {}
    for op_name, op_params in mx_operators.items():
        if op_name.startswith("elemwise_") and op_params["params"]["narg"] == 2 and \
                "lhs" in op_params["params"]["arg_names"] and \
                "rhs" in op_params["params"]["arg_names"]:
            binary_elemen_wise_mx_operators[op_name] = mx_operators[op_name]
    return binary_elemen_wise_mx_operators


def get_all_random_sampling_operators():
    """Gets all Random Sampling operators registered with MXNet.

    Returns
    -------
    {"operator_name": {"has_backward", "nd_op_handle", "params"}}
    """
    # Get all mxnet operators
    mx_operators = _get_all_mxnet_operators()

    # Filter for Random Sampling operators
    random_sampling_mx_operators = {}
    for op_name, op_params in mx_operators.items():
        if op_name.startswith(("random_", "sample_")) and op_name not in unique_ops:
            random_sampling_mx_operators[op_name] = mx_operators[op_name]
    return random_sampling_mx_operators


def get_all_reduction_operators():
    """Gets all Reduction operators registered with MXNet.

    Returns
    -------
    {"operator_name": {"has_backward", "nd_op_handle", "params"}}
    """
    # Get all mxnet operators
    mx_operators = _get_all_mxnet_operators()

    # Filter for Reduction operators
    reduction_mx_operators = {}
    for op_name, op_params in mx_operators.items():
        if op_params["params"]["narg"] == 4 and \
                set(["data", "axis", "exclude", "keepdims"]).issubset(set(op_params["params"]["arg_names"])) \
                and op_name not in unique_ops:
            reduction_mx_operators[op_name] = mx_operators[op_name]
    return reduction_mx_operators


def get_all_optimizer_operators():
    """Gets all Optimizer operators registered with MXNet.

     Returns
     -------
     {"operator_name": {"has_backward", "nd_op_handle", "params"}}
     """
    optimizer_ops = ['mp_sgd_update', 'signum_update', 'rmspropalex_update', 'ftml_update', 'rmsprop_update',
                     'sgd_mom_update', 'signsgd_update', 'mp_sgd_mom_update', 'ftrl_update', 'sgd_update',
                     'adam_update']

    # Get all mxnet operators
    mx_operators = _get_all_mxnet_operators()

    # Filter for Optimizer operators
    optimizer_mx_operators = {}
    for op_name, op_params in mx_operators.items():
         if op_name in optimizer_ops and op_name not in unique_ops:
             optimizer_mx_operators[op_name] = mx_operators[op_name]
    return optimizer_mx_operators

def get_all_sorting_searching_operators():
    """Gets all Sorting and Searching operators registered with MXNet.

    Returns
    -------
    {"operator_name": {"has_backward", "nd_op_handle", "params"}}
    """
    sort_search_ops = ['sort', 'argsort', 'argmax', 'argmin', 'topk']

    # Get all mxnet operators
    mx_operators = _get_all_mxnet_operators()

    # Filter for Sort and search operators
    sort_search_mx_operators = {}
    for op_name, op_params in mx_operators.items():
        if op_name in sort_search_ops and op_name not in unique_ops:
            sort_search_mx_operators[op_name] = mx_operators[op_name]
    return sort_search_mx_operators


def get_all_rearrange_operators():
    """Gets all array rearrange operators registered with MXNet.

    Returns
    -------
    {"operator_name": {"has_backward", "nd_op_handle", "params"}}
    """
    rearrange_ops = ['transpose','swapaxes','flip','depth_to_space','space_to_depth']

    # Get all mxnet operators
    mx_operators = _get_all_mxnet_operators()

    # Filter for Array Rearrange operators
    rearrange_mx_operators = {}
    for op_name, op_params in mx_operators.items():
        if op_name in rearrange_ops and op_name not in unique_ops:
            rearrange_mx_operators[op_name] = mx_operators[op_name]
    return rearrange_mx_operators
    

def get_operators_with_no_benchmark(operators_with_benchmark):
    """Gets all MXNet operators with not benchmark.

    Retrieve all operators registered with MXNet and prepares a list of operators that are not part of given
    operators with benchmark list.

    Parameters
    ----------
    operators_with_benchmark: list[Str]
        List of operator names that has benchmarks

    Returns
    -------
    list[Str]
        List of operator names that is registered with MXNet but has no benchmarks.
    """
    all_mxnet_operators = _get_all_mxnet_operators().keys()
    return list(set(all_mxnet_operators) - set(operators_with_benchmark))


def get_current_runtime_features():
    """Get all current runtime time flags/configuration for MXNet.

    Returns
    -------
    Map of current runtime features such as compile flags used by MXNet.
        Example: {'runtime_features': {'OPENCV' : '✔ OPENCV', 'CUDA': '✖ CUDA'}}
    """
    features = runtime.Features()
    runtime_features = {}
    for feature, config in sorted(features.items(), key=itemgetter(0)):
        runtime_features[feature] = config

    return {'runtime_features': runtime_features}
