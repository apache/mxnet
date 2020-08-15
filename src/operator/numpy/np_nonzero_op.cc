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
 * Copyright (c) 2018 by Contributors
 * \file np_nonzero_op.cc
*/
#include "np_nonzero_op-inl.h"

namespace mxnet {
namespace op {

bool NonzeroType(const nnvm::NodeAttrs& attrs,
                 std::vector<int> *in_attrs,
                 std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  // Output must be int64.
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt64);
  return out_attrs->at(0) != -1;
}

#define MAXDIM 5

bool NonzeroStorageType(const nnvm::NodeAttrs& attrs,
                        const int dev_mask,
                        DispatchMode* dispatch_mode,
                        std::vector<int> *in_attrs,
                        std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  for (int &attr : *in_attrs) {
    CHECK_EQ(attr, kDefaultStorage) << "Only default storage is supported";
  }
  for (int &attr : *out_attrs) {
    attr = kDefaultStorage;
  }
  *dispatch_mode = DispatchMode::kFComputeEx;
  return true;
}

void NonzeroForwardCPU(const nnvm::NodeAttrs& attrs,
                       const OpContext &ctx,
                       const std::vector<NDArray> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<NDArray> &outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const NDArray &in = inputs[0];
  const NDArray &out = outputs[0];
  CHECK_LE(in.shape().ndim(), MAXDIM) << "ndim of input cannot larger than " << MAXDIM;
  // 0-dim
  if (0 == in.shape().ndim()) {
    MSHADOW_TYPE_SWITCH_WITH_BOOL(in.dtype(), DType, {
      DType* in_dptr = in.data().dptr<DType>();
      if (*in_dptr) {
        mxnet::TShape s(2, 1);
        const_cast<NDArray &>(out).Init(s);
        *(out.data().dptr<int64_t>()) = 0;
      } else {
        mxnet::TShape s(2, 1);
        s[0] = 0;
        const_cast<NDArray &>(out).Init(s);
      }
    });
    return;
  }
  size_t in_size = in.shape().Size();
  // 0-shape
  if (0 == in_size) {
    mxnet::TShape s(2, in.shape().ndim());
    s[0] = 0;
    const_cast<NDArray &>(out).Init(s);
    return;
  }
  std::vector<int32_t> prefix_sum(in_size, 0);
  size_t valid_num = 0;
  // Calculate prefix sum
  MSHADOW_TYPE_SWITCH_WITH_BOOL(in.dtype(), DType, {
    DType* in_dptr = in.data().dptr<DType>();
    for (size_t i = 0; i < in_size; i++) {
      prefix_sum[i] = (i == 0) ? 0 : prefix_sum[i - 1];
      prefix_sum[i] += (in_dptr[i]) ? 1 : 0;
    }
  });
  valid_num = prefix_sum[in_size - 1];
  // set the output shape forcefully
  mxnet::TShape s(2, in.shape().ndim());
  s[0] = valid_num;
  const_cast<NDArray &>(out).Init(s);
  // get the shape from the input
  MXNET_NDIM_SWITCH(in.shape().ndim(), ndim, {
    mshadow::Shape<ndim> shape = in.shape().get<ndim>();
    mshadow::Stream<cpu> *stream = ctx.get_stream<cpu>();
    mxnet_op::Kernel<NonzeroForwardKernel, cpu>::Launch(
      stream, in_size, out.data().dptr<int64_t>(), prefix_sum.data(), shape);
  })
}

NNVM_REGISTER_OP(_npx_nonzero)
.add_alias("_npi_nonzero")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"x"};
  })
.set_attr<nnvm::FInferType>("FInferType", NonzeroType)
.set_attr<FComputeEx>("FComputeEx<cpu>", NonzeroForwardCPU)
.set_attr<FInferStorageType>("FInferStorageType", NonzeroStorageType)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("x", "NDArray-or-Symbol", "The input array.");

}  // namespace op
}  // namespace mxnet
