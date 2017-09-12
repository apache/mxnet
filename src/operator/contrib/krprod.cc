/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
n * with the License.  You may obtain a copy of the License at
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
 *  \file krprod.cc
 *  \brief Operator registration for Khatri-Rao product
 *  \author Chris Swierczewski
 */

#include <mshadow/tensor.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/operator_util.h>
#include <vector>
#include <string>
#include <algorithm>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../../ndarray/ndarray_function.h"
#include "krprod.h"

namespace mxnet {
namespace op {

inline bool KhatriRaoShape(
      const nnvm::NodeAttrs& attrs,
      std::vector<TShape> *in_attrs,
      std::vector<TShape> *out_attrs) {
  CHECK_EQ(out_attrs->size(), 1);
  CHECK_GE(in_attrs->size(), 1);

  // all input and output matrices must have the same number of rows/columns
  // (when inputs_transposed is set to true/false)
  int num_columns = static_cast<int>((*in_attrs)[0][1]);
  int num_rows = 1;
  for (const TShape& attr_shape : (*in_attrs)) {
    CHECK_EQ(num_columns, static_cast<int>(attr_shape[1]));
    num_rows *= attr_shape[0];
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, Shape2(num_rows, num_columns));
  return true;
}


struct KhatriRaoParam : public dmlc::Parameter<KhatriRaoParam> {
  int num_args;
  bool row_wise = false;
  DMLC_DECLARE_PARAMETER(KhatriRaoParam) {
    DMLC_DECLARE_FIELD(num_args)
      .set_lower_bound(1)
      .describe("Number of input matrices.");
  }
};
DMLC_REGISTER_PARAMETER(KhatriRaoParam);


NNVM_REGISTER_OP(khatri_rao)
.describe(R"code(Computes the Khatri-Rao product of the input matrices.

Given a collection of :math:`n` input matrices,

.. math::
   A_1 \in \mathbb{R}^{M_1 \times M}, \ldots, A_n \in \mathbb{R}^{M_n \times N},

the (column-wise) Khatri-Rao product is defined as the matrix,

.. math::
   X = A_1 \otimes \cdots \otimes A_n \in \mathbb{R}^{(M_1 \cdots M_n) \times N},

where the :math:`k`th column is equal to the column-wise outer product
:math:`{A_1}_k \otimes \cdots \otimes {A_n}_k` where :math:`{A_i}_k` is the kth
column of the ith matrix.

Example::

  >>> A = mx.nd.array([[1, -1],
  >>>                  [2, -3]])
  >>> B = mx.nd.array([[1, 4],
  >>>                  [2, 5],
  >>>                  [3, 6]])
  >>> C = mx.nd.khatri_rao(A, B)
  >>> print(C.asnumpy())
  [[  1.  -4.]
   [  2.  -5.]
   [  3.  -6.]
   [  2. -12.]
   [  4. -15.]
   [  6. -18.]]

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<KhatriRaoParam>)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    uint32_t ret = dmlc::get<KhatriRaoParam>(attrs.parsed).num_args;
    return ret;
  })
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", KhatriRaoShape)
.set_attr<nnvm::FInferType>("FInferType",
  [](const nnvm::NodeAttrs& attrs,
     std::vector<int> *in_attrs,
     std::vector<int> *out_attrs) {
    return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
      attrs, in_attrs, out_attrs, -1);
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    uint32_t num_args = dmlc::get<KhatriRaoParam>(attrs.parsed).num_args;
    std::vector<std::string> ret;
    for (uint32_t i = 0; i < num_args; ++i)
      ret.push_back(std::string("arg") + std::to_string(i));
    return ret;
  })
.set_attr<FCompute>("FCompute<cpu>", KhatriRaoCompute<cpu>)
.set_attr<std::string>("key_var_num_args", "num_args")
.add_argument("args", "NDArray-or-Symbol[]", "Positional input matrices");

}  // namespace op
}  // namespace mxnet
