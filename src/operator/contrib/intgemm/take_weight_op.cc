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
 * \file select_weight_op.cc
 * \brief Takes from the all-but-last dimension of a tensor stored in
 * intgemm's weight format.  This is particularly useful for output matrices where
 * some outputs are excluded.
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

inline bool TakeWeightOpShape(const nnvm::NodeAttrs& shape,
                             mxnet::ShapeVector* in_shape,
                             mxnet::ShapeVector* out_shape) {
  // 0 is weight, 1 is indices.
  CHECK_EQ(in_shape->size(), 2U);
  CHECK_EQ(out_shape->size(), 1U);

  mxnet::TShape &weight = (*in_shape)[0];
  mxnet::TShape &indices = (*in_shape)[1];
  mxnet::TShape &out = (*out_shape)[0];

  // weight matrices should be 2-dimensional by now.
  SHAPE_ASSIGN_CHECK(*in_shape, 0, mxnet::TShape(2, -1));
  SHAPE_ASSIGN_CHECK(*out_shape, 0, mxnet::TShape(2, -1));
  // indices are 1-dimensional.
  SHAPE_ASSIGN_CHECK(*in_shape, 1, mxnet::TShape(1, -1));

  SHAPE_ASSIGN_CHECK(*out_shape, 0, mxnet::TShape({indices[0], weight[1]}));
  SHAPE_ASSIGN_CHECK(*in_shape, 0, mxnet::TShape({-1, out[1]}));
  SHAPE_ASSIGN_CHECK(*in_shape, 1, mxnet::TShape({out[0]}));

  return shape_is_known(weight) && shape_is_known(indices) && shape_is_known(out);
}

inline bool TakeWeightOpType(const nnvm::NodeAttrs& attrs,
                             std::vector<int>* in_attrs,
                             std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt8);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, mshadow::kInt8);
  TYPE_ASSIGN_CHECK(*in_attrs, 1, mshadow::kInt32);
  return true;
}

inline bool TakeWeightOpStorageType(const nnvm::NodeAttrs& attrs,
                                    const int dev_mask,
                                    DispatchMode* dispatch_mode,
                                    std::vector<int>* in_attrs,
                                    std::vector<int>* out_attrs) {
  *dispatch_mode = DispatchMode::kFCompute;
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  (*out_attrs)[0] = kDefaultStorage;
  return true;
}

void TakeWeightOpForwardCPU(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteTo) << "TODO request types other than write";
  const TBlob &weight = inputs.front(), &indices = inputs[1], &out = outputs.front();
  CHECK_EQ(weight.type_flag_, mshadow::kInt8);
  CHECK_EQ(indices.type_flag_, mshadow::kInt32);
  CHECK_EQ(out.type_flag_, mshadow::kInt8);
  CHECK(weight.CheckContiguous());
  CHECK(indices.CheckContiguous());
  CHECK(out.CheckContiguous());
  size_t B_cols = indices.shape_[0];
  size_t inner = weight.shape_[weight.shape_.ndim() - 1];
  CHECK_EQ(inner % ::intgemm::Int8::tile_info.b_rows, 0) <<
    "intgemm requires the inner dimension be a multiple of " << ::intgemm::Int8::tile_info.b_rows;
  CHECK_EQ(B_cols % ::intgemm::Int8::tile_info.b_cols, 0) <<
    "For efficiency, intgemm requires there to be a multiple of " <<
    ::intgemm::Int8::tile_info.b_cols << " indices.";
  // mxnet doesn't have a uint32_t type so we'll just pointer cast. But check the sizes are the
  // same.  Ideally this should be static.
  assert(sizeof(int32_t) == sizeof(::intgemm::Index));
  const ::intgemm::Index *index =
    reinterpret_cast<const ::intgemm::Index*>(indices.dptr<int32_t>());

  ::intgemm::Int8::SelectColumnsB(
      weight.dptr<int8_t>(),
      out.dptr<int8_t>(),
      inner,
      index,
      index + B_cols);
}

NNVM_REGISTER_OP(_contrib_intgemm_take_weight)
.add_alias("_npx_intgemm_take_weight")
.describe(R"code(Index a weight matrix stored in intgemm's weight format.
The indices select the outputs of matrix multiplication, not the inner dot product dimension.
)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"weight", "indices"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", TakeWeightOpShape)
.set_attr<nnvm::FInferType>("FInferType", TakeWeightOpType)
.set_attr<FInferStorageType>("FInferStorageType", TakeWeightOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", TakeWeightOpForwardCPU)
.add_argument(
    "weight",
    "NDArray-or-Symbol",
    "Tensor already in intgemm weight format to select from")
.add_argument("indices", "NDArray-or-Symbol", "indices to select on the 0th dimension of weight");

}  // namespace op
}  // namespace mxnet
