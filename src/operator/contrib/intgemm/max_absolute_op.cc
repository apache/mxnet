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
 * \file max_absolute_op.cc
 * \brief Computes maximum absolute value of a tensor using intgemm
 */

#include <mxnet/operator_util.h>
#include <vector>
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../tensor/init_op.h"

#include "intgemm/intgemm.h"

namespace mxnet {
namespace op {

inline bool MaxAbsoluteOpShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_attrs,
                             mxnet::ShapeVector* out_attrs) {
  // One in, one out.
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(1, 1));
  return shape_is_known(in_attrs->at(0));
}

inline bool MaxAbsoluteOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, mshadow::kFloat32);
  return true;
}

inline bool MaxAbsoluteOpStorageType(const nnvm::NodeAttrs& attrs,
                                   const int dev_mask,
                                   DispatchMode* dispatch_mode,
                                   std::vector<int>* in_attrs,
                                   std::vector<int>* out_attrs) {
  *dispatch_mode = DispatchMode::kFCompute;
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  (*out_attrs)[0] = kDefaultStorage;
  return true;
}

void MaxAbsoluteOpForwardCPU(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const TBlob &in = inputs.front(), &out = outputs.front();
  CHECK_EQ(in.type_flag_, mshadow::kFloat32);
  CHECK_EQ(out.type_flag_, mshadow::kFloat32);
  CHECK(in.CheckContiguous());
  CHECK(out.CheckContiguous());

  const std::size_t size = in.shape_.Size();

  const float *data = in.dptr<float>();
  // To maintain alignment, be a multiple of AVX512 register size.
  const std::size_t kMultiple = 512 / 8;
  CHECK_EQ(reinterpret_cast<intptr_t>(data) % kMultiple, 0)
    << "Data must be aligned to " << kMultiple << " bytes.";

  float result = ::intgemm::MaxAbsolute(data, data + size);
  KERNEL_ASSIGN(*out.dptr<float>(), req[0], result);
}

NNVM_REGISTER_OP(_contrib_intgemm_maxabsolute)
.add_alias("_npx_intgemm_maxabsolute")
.describe(R"code(Compute the maximum absolute value in a tensor of float32 fast on a CPU.  The tensor's total size must be a multiple of 16 and aligned to a multiple of 64 bytes.
mxnet.nd.contrib.intgemm_maxabsolute(arr) == arr.abs().max()
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", MaxAbsoluteOpShape)
.set_attr<nnvm::FInferType>("FInferType", MaxAbsoluteOpType)
.set_attr<FInferStorageType>("FInferStorageType", MaxAbsoluteOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", MaxAbsoluteOpForwardCPU)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Tensor to compute maximum absolute value of");

}  // namespace op
}  // namespace mxnet
