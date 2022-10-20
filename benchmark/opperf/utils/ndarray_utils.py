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

import numpy as np
import mxnet as mx
import mxnet.ndarray as nd


def nd_forward_backward_and_profile(op, runs, **kwargs):
    """Helper function to run a given NDArray operator (op) for 'runs' number of times with
    given args and kwargs. Executes both forward and backward pass.

    NOTE: This is a sync call and waits for all the operations execution to complete.

    Parameters
    ----------
    op: Str
        NDArray operator (Function reference) to execute. Example: mx.nd.add
    runs: int
        Number of times to execute the operation
    kwargs:
        Key value arguments for the NDArray operator (op) being executed.

    Returns
    -------
    any results from NDArray operation execution

    """
    for _ in range(runs):
        with mx.autograd.record():
            args = []
            # need to create a new dictionary because can't update dict while iterating
            kwargs_new = dict()
            for key in kwargs:
                # separate positional args from key-worded args
                if key.startswith("args"):
                    args.append(kwargs[key])
                else:
                    kwargs_new[key]=kwargs[key]
            # check for positional args
            if len(args):
                res = op(*args, **kwargs_new)
            else:
                res = op(**kwargs_new)
        res.backward()
        nd.waitall()
    return res


def nd_forward_and_profile(op, runs, **kwargs):
    """Helper function to run a given NDArray operator (op) for 'runs' number of times with
    given args and kwargs. Executes ONLY forward pass.

    NOTE: This is a sync call and waits for all the operations execution to complete.

    Parameters
    ----------
    op: Str
        NDArray operator (Function reference) to execute. Example: mx.nd.add
    runs: int
        Number of time to execute the operation
    kwargs:
        Key value arguments for the NDArray operator (op) being executed.

    Returns
    -------
    any results from NDArray operation execution
    """
    for _ in range(runs):
        args = []
        # need to create a new dictionary because can't update dict while iterating
        kwargs_new = dict()
        for key in kwargs:
            # separate positional args from key-worded args
            if key.startswith("args"):
                args.append(kwargs[key])
            else:
                kwargs_new[key]=kwargs[key]
        # check for positional args
        if len(args):
            res = op(*args, **kwargs_new)
        else:
            res = op(**kwargs_new)
        nd.waitall()
    return res


def get_mx_ndarray(ctx, in_tensor, dtype, initializer, attach_grad=True):
    """Helper function to prepare a MXNet NDArray tensor in given Context (ctx) of type (dtype) with given
    initializer. You can get a new Tensor by providing only "Shape" or "Numpy NDArray" or another MXNet NDArray as
    "in_tensor".

    NOTE: This is a sync call and waits for the Tensor to be created.

    Parameters
    ----------
    ctx: mx.ctx, default mx.cpu()
        Context of the new MXNet NDArray Tensor.
    in_tensor: Numpy NDArray or MXNet NDArray or Tuple of shape
        Can be a tuple of shape or Numpy NDArray or MXNet NDArray.
    dtype: str
        Precision or Dtype of the expected Tensor. Ex: "float32", "Int64"
    initializer:
        Function reference to the initialize to use. Ex: mx.nd.random.normal, mx.nd.zeros
    attach_grad: Boolean, default True
        To attach a gradient for the Tensor. Default is True.

    Returns
    -------
    MXNet NDArray Tensor.
    """
    if isinstance(in_tensor, int) or isinstance(in_tensor, float):
        return in_tensor

    if isinstance(in_tensor, tuple):
        tensor = initializer(ctx=ctx, shape=in_tensor, dtype=dtype)
    elif isinstance(in_tensor, list):
        tensor = nd.array(in_tensor, ctx=ctx, dtype=dtype)
    elif isinstance(in_tensor, np.ndarray):
        tensor = nd.array(in_tensor)
    elif isinstance(in_tensor, mx.np.ndarray):
        tensor = in_tensor.as_nd_ndarray()
    elif isinstance(in_tensor, nd.NDArray):
        tensor = in_tensor.as_in_context(ctx)
    else:
        raise ValueError("Invalid input type for creating input tensor. Input can be tuple() of shape or Numpy Array or"
                         " MXNet NDArray. Given - ", in_tensor)

    if attach_grad:
        tensor.attach_grad()

    tensor.wait_to_read()
    return tensor
