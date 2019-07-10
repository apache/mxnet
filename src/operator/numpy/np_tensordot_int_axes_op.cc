/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the 
 * icense at
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
 * \file np_tensordot_int_axes_op.cc
 * \brief CPU Implementation of numpy-compatible tensordot
 */

#include <string>

#include "np_tensordot_int_axes_op-inl.h"

namespace mxnet {
namespace op {

bool TensordotIntAxesOpShape(
    const nnvm::NodeAttrs& attrs,
    mxnet::ShapeVector *in_attrs,
    mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& a_shape = in_attrs->at(0);
  const mxnet::TShape& b_shape = in_attrs->at(1);

  if (!ndim_is_known(a_shape) || !ndim_is_known(b_shape)) {
    return false;
  }

  if ((a_shape.ndim() < 1) || (b_shape.ndim() < 1)) {
    return false;
  }

  const TensordotIntAxesParam& param = nnvm::get<TensordotIntAxesParam>(attrs.parsed);
  const int& axes = param.axes;

  Tuple<int> a_axes_summed;
  Tuple<int> b_axes_summed;
  GetSummedAxes(&a_axes_summed, &b_axes_summed, axes, a_shape);

  Tuple<int> a_axes_remained;
  Tuple<int> b_axes_remained;
  Tuple<int> a_axes;
  Tuple<int> b_axes;
  GetReorderedAxes(a_axes_summed, &a_axes_remained, &a_axes, b_axes_summed, &b_axes_remained,
    &b_axes, a_shape, b_shape);

  CHECK_EQ(a_axes_summed.ndim(), b_axes_summed.ndim());

  mxnet::TShape out_shape(a_axes_remained.ndim() + b_axes_remained.ndim(), -1);
  for (int i = 0; i < a_axes_remained.ndim(); i++) {
    out_shape[i] = a_shape[a_axes_remained[i]];
  }
  for (int i = 0; i < b_axes_remained.ndim(); i++) {
    out_shape[a_axes_remained.ndim() + i] = b_shape[b_axes_remained[i]];
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);

  mxnet::TShape tem_shape1(a_axes.ndim(), -1);
  for (int i = 0; i < a_axes_remained.ndim(); i++) {
    tem_shape1[a_axes_remained[i]] = out_shape[i];
  }
  for (int i = 0; i < a_axes_summed.ndim(); i++) {
    tem_shape1[a_axes_summed[i]] = b_shape[b_axes_summed[i]];
  }
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, tem_shape1);

  mxnet::TShape tem_shape2(b_axes.ndim(), -1);
  for (int i = 0; i < b_axes_remained.ndim(); i++) {
    tem_shape2[b_axes_remained[i]] = out_shape[a_axes_remained.ndim() + i];
  }
  for (int i = 0; i < b_axes_summed.ndim(); i++) {
    tem_shape2[b_axes_summed[i]] = a_shape[a_axes_summed[i]];
  }
  SHAPE_ASSIGN_CHECK(*in_attrs, 1, tem_shape2);

  return shape_is_known(*in_attrs) && shape_is_known(*out_attrs);
}

DMLC_REGISTER_PARAMETER(TensordotIntAxesParam);

NNVM_REGISTER_OP(tensordot_int_axes)
.add_alias("_npi_tensordot_int_axes")
.describe(R"code(tensordot(a, b, axes=2)

    Compute tensor dot product along specified axes for arrays >= 1-D.

    Given two tensors (arrays of dimension greater than or equal to one),
    `a` and `b`, and an array_like object containing two array_like
    objects, ``(a_axes, b_axes)``, sum the products of `a`'s and `b`'s
    elements (components) over the axes specified by ``a_axes`` and
    ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N``
    dimensions of `a` and the first ``N`` dimensions of `b` are summed
    over.

    Parameters
    ----------
    a, b : ndarray, len(shape) >= 1
        Tensors to "dot".

    axes : int or (2,) ndarray
        * integer_like
        If an int N, sum over the last N axes of `a` and the first N axes
        of `b` in order. The sizes of the corresponding axes must match.
        * (2,) ndarray
        Or, a list of axes to be summed over, first sequence applying to `a`,
        second to `b`. Both elements ndarray must be of the same length.

    See Also
    --------
    dot, einsum

    Notes
    -----
    Three common use cases are:
        * ``axes = 0`` : tensor product :math:`a\otimes b`
        * ``axes = 1`` : tensor dot product :math:`a\cdot b`
        * ``axes = 2`` : (default) tensor double contraction :math:`a:b`

    When `axes` is integer_like, the sequence for evaluation will be: first
    the -Nth axis in `a` and 0th axis in `b`, and the -1th axis in `a` and
    Nth axis in `b` last.

    When there is more than one axis to sum over - and they are not the last
    (first) axes of `a` (`b`) - the argument `axes` should consist of
    two sequences of the same length, with the first axis to sum over given
    first in both sequences, the second axis second, and so forth.

    Examples
    --------
    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> c = np.tensordot(a,b, axes=([1,0],[0,1]))
    >>> c.shape
    (5, 2)
    >>> c
    array([[ 4400.,  4730.],
           [ 4532.,  4874.],
           [ 4664.,  5018.],
           [ 4796.,  5162.],
           [ 4928.,  5306.]])
)code" ADD_FILELINE)
.set_attr_parser(mxnet::op::ParamParser<TensordotIntAxesParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "b"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", TensordotIntAxesOpShape)
.set_attr<nnvm::FInferType>("FInferType", mxnet::op::ElemwiseType<2, 1>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", TensordotIntAxesOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
    mxnet::op::ElemwiseGradUseIn{"_backward_tensordot_int_axes"})
.add_argument("a", "NDArray-or-Symbol", "First input")
.add_argument("b", "NDArray-or-Symbol", "Second input")
.add_arguments(TensordotIntAxesParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_tensordot_int_axes)
.set_attr_parser(mxnet::op::ParamParser<TensordotIntAxesParam>)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", TensordotIntAxesOpBackward<cpu>);

}  // namespace op
}  // namespace mxnet
