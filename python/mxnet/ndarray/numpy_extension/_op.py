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

"""Namespace for the operators not belonging to the official numpy package
used in Gluon dispatched by F=ndarray module."""

__all__ = ['cond']

def cond(pred, then_func, else_func):
    """Run an if-then-else using user-defined condition and computation

    This operator simulates a if-like branch which chooses to do one of
    the two customized computations according to the specified condition.

    `pred` is a scalar MXNet NDArray,
    indicating which branch of computation should be used.

    `then_func` is a user-defined function, used as computation of the then branch.
    It produces `outputs`, which is a list of NDArrays.
    The signature of `then_func` should be
    `then_func() => NDArray or nested List[NDArray]`.

    `else_func` is a user-defined function, used as computation of the else branch.
    It produces `outputs`, which is a list of NDArrays.
    The signature of `else_func` should be
    `else_func() => NDArray or nested List[NDArray]`.

    The `outputs` produces by `then_func` and `else_func` should have the same number
    of elements, all of which should be in the same shape, of the same dtype and stype.

    This function returns a list of symbols, representing the computation result.

    Parameters
    ----------
    pred: a MXNet numpy NDArray representing a scalar.
        The branch condition.
    then_func: a Python function.
        The computation to be executed if `pred` is true.
    else_func: a Python function.
        The computation to be executed if `pred` is false.

    Returns
    -------
    outputs: an NDArray or nested lists of NDArrays, representing the result of computation.

    Examples
    --------
    >>> a, b = mx.nd.array([1]), mx.nd.array([2])
    >>> pred = a * b < 5
    >>> then_func = lambda: (a + 5) * (b + 5)
    >>> else_func = lambda: (a - 5) * (b - 5)
    >>> outputs = mx.nd.contrib.cond(pred, then_func, else_func)
    >>> outputs[0]
    [42.]
    <NDArray 1 @cpu(0)>
    """
    def _to_python_scalar(inputs, type_, name):
        """Converts "inputs", possibly typed mxnet NDArray, a numpy ndarray, other python types,
        to the given type
        """
        if hasattr(inputs, "asnumpy"):
            inputs = inputs.asnumpy()
        try:
            inputs = type_(inputs)
        except:
            raise ValueError("Cannot convert %s to python %s" % (name, type_.__name__))
        return inputs

    branch = _to_python_scalar(pred, bool, "pred")
    if branch:
        return then_func()
    else:
        return else_func()
