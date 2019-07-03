/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2019 by Contributors
 * \file np_elemwise_binary_op.cc
 * \brief CPU Implementation for element-wise binary operators.
 */


#include <mxnet/base.h>
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../tensor/elemwise_binary_op.h"
#include "np_elemwise_broadcast_op.cc"

namespace mxnet {
namespace op {

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(ldexp)
.describe(R"code(ldexp(x1, x2)

Returns x1 * 2**x2, element-wise.

The mantissas `x1` and twos exponents `x2` are used to construct
floating point numbers ``x1 * 2**x2``.

Parameters
----------
x1 : array_like
    Array of multipliers.
x2 : array_like, int
    Array of twos exponents.
out : ndarray, None, or tuple of ndarray and None, optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or `None`,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
where : array_like, optional
    Values of True indicate to calculate the ufunc at that position, values
    of False indicate to leave the value in the output alone.
**kwargs
    For other keyword-only arguments, see the
    :ref:`ufunc docs <ufuncs.kwargs>`.

Returns
-------
y : ndarray or scalar
    The result of ``x1 * 2**x2``.
    This is a scalar if both `x1` and `x2` are scalars.

See Also
--------
frexp : Return (y1, y2) from ``x = y1 * 2**y2``, inverse to `ldexp`.

Notes
-----
Complex dtypes are not supported, they will raise a TypeError.

`ldexp` is useful as the inverse of `frexp`, if used by itself it is
more clear to simply use the expression ``x1 * 2**x2``.

Examples
--------
>>> np.ldexp(5, np.arange(4))
array([  5.,  10.,  20.,  40.])

>>> x = np.arange(6)
>>> np.ldexp(*np.frexp(x))
array([ 0.,  1.,  2.,  3.,  4.,  5.])

)code" ADD_FILELINE)
.add_alias("_npi_ldexp")  
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::ldexp>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_ldexp"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(ldexp_scalar)
.add_alias("_npi_ldexp_scalar")  
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::ldexp>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_ldexp_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(rldexp_scalar)
.add_alias("_npi_rldexp_scalar") 
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::ldexp>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_rldexp_scalar"});

NNVM_REGISTER_OP(_backward_ldexp)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseIn<cpu, mshadow_op::ldexp_grad,
                                                              mshadow_op::ldexp_rgrad>);

MXNET_OPERATOR_REGISTER_BINARY(_backward_ldexp_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs *attrs) { attrs->parsed = std::stod(attrs->dict["scalar"]); })
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Backward<cpu, mshadow_op::ldexp_grad>);

MXNET_OPERATOR_REGISTER_BINARY(_backward_rldexp_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs *attrs) { attrs->parsed = std::stod(attrs->dict["scalar"]); })
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Backward<cpu, mshadow_op::ldexp_rgrad>);

}  // namespace op
}  // namespace mxnet
