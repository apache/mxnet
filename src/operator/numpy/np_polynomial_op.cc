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
 * Copyright (c) 202o by Contributors
 * \file np_polynomial_op.cc
*/

#include <math.h>
#include "np_polynomial_op-inl.h"

namespace mxnet {
namespace op {

template<int req>
struct polyval_backward_x {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType* p_dptr, const DType* x_dptr,
                                  DType* igrad_x_dptr, const DType* ograd_dptr,
                                  const index_t p_size) {
    DType igrad_x = 0;
    index_t j = p_size - 1;
    while (j > 0) {
        igrad_x = igrad_x * x_dptr[i] + p_dptr[p_size - j - 1] * j;
        j--;
    }
    KERNEL_ASSIGN(igrad_x_dptr[i], req, igrad_x * ograd_dptr[i]);
  }
};

template<int req>
struct polyval_backward_p {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType* p_dptr, const DType* x_dptr,
                                  DType* igrad_p_dptr, const DType* ograd_dptr,
                                  const index_t p_size, const index_t x_size) {
    DType igrad_p = 0;
    index_t j = x_size - 1;
    while (j >= 0) {
        igrad_p += pow(x_dptr[j], p_size - i - 1) * ograd_dptr[j];
        j--;
    }
    KERNEL_ASSIGN(igrad_p_dptr[i], req, igrad_p);
  }
};

void NumpyPolyvalBackwardCPU(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_NE(req[0], kWriteInplace);

  if (inputs[1].type_flag_ != inputs[2].type_flag_ ||
      !common::is_float(inputs[1].type_flag_) ||
      !common::is_float(inputs[2].type_flag_)) {
        return;
  }

  mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
  const TBlob& ograd = inputs[0];
  const TBlob& p = inputs[1];
  const TBlob& x = inputs[2];
  const TBlob& igrad_p = outputs[0];
  const TBlob& igrad_x = outputs[1];
  const size_t p_size = p.Size();

  using namespace mxnet_op;
  MSHADOW_REAL_TYPE_SWITCH(ograd.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<polyval_backward_x<req_type>, cpu>::Launch(
        s, ograd.Size(), p.dptr<DType>(), x.dptr<DType>(),
        igrad_x.dptr<DType>(), ograd.dptr<DType>(), p_size);
      Kernel<polyval_backward_p<req_type>, cpu>::Launch(
        s, p_size, p.dptr<DType>(), x.dptr<DType>(),
        igrad_p.dptr<DType>(), ograd.dptr<DType>(), p_size, x.Size());
    });
  });
}

NNVM_REGISTER_OP(_npi_polyval)
.set_num_inputs(2)
.set_num_outputs(1)
.add_argument("p", "NDArray-or-Symbol", "polynomial coefficients")
.add_argument("x", "NDArray-or-Symbol", "variables")
.set_attr<nnvm::FListInputNames>("FListInputNames",
[](const NodeAttrs& attrs) {
    return std::vector<std::string>{"p", "x"};
})
.set_attr<mxnet::FInferShape>("FInferShape", NumpyPolyvalShape)
.set_attr<nnvm::FInferType>("FInferType", mxnet::op::ElemwiseType<2, 1>)
.set_attr<mxnet::FCompute>("FCompute<cpu>", NumpyPolyvalForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_npi_backward_polyval"});

NNVM_REGISTER_OP(_npi_backward_polyval)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<mxnet::FCompute>("FCompute<cpu>", NumpyPolyvalBackwardCPU);

}  // namespace op
}  // namespace mxnet
