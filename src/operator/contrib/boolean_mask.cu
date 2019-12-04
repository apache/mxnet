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
 * \file boolean_mask.cu
*/

#include "./boolean_mask-inl.h"
#include <cub/cub.cuh>

namespace mxnet {
namespace op {

template<>
inline void BooleanMaskForward<gpu>(const nnvm::NodeAttrs& attrs,
                                    const OpContext &ctx,
                                    const std::vector<NDArray> &inputs,
                                    const std::vector<OpReqType> &req,
                                    const std::vector<NDArray> &outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK(req[0] == kWriteTo || req[0] == kWriteInplace);
  const BooleanMaskParam& param = nnvm::get<BooleanMaskParam>(attrs.parsed);
  const int axis = param.axis;
  const NDArray &data = inputs[0];
  const NDArray &idx = inputs[1];
  const NDArray &out = outputs[0];
  CHECK_EQ(axis, 0) << "Not supported yet";
  CHECK_EQ(data.shape()[axis], idx.shape()[0]);
  CHECK_EQ(idx.shape().ndim(), 1U);
  Stream<gpu>* s = ctx.get_stream<gpu>();
  // count the number of 1s in `idx`, so that we could know the output dimension
  size_t idx_size = idx.shape()[0];
  int32_t valid_num = 0;
  int32_t* prefix_sum = nullptr;
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  // Calculate total temporary memory size
  cub::DeviceScan::InclusiveSum(d_temp_storage,
                                temp_storage_bytes,
                                prefix_sum,
                                prefix_sum,
                                idx_size,
                                Stream<gpu>::GetStream(s));
  size_t buffer_size = idx_size * sizeof(int32_t);
  temp_storage_bytes += buffer_size;
  // Allocate memory on GPU and allocate pointer
  Tensor<gpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(temp_storage_bytes), s);
  prefix_sum = reinterpret_cast<int32_t*>(workspace.dptr_);
  d_temp_storage = workspace.dptr_ + buffer_size;
  MSHADOW_TYPE_SWITCH_WITH_BOOL(idx.dtype(), IType, {
    mxnet_op::Kernel<mshadow_op::identity_with_cast, gpu>::Launch(
      s, idx.shape()[0], prefix_sum, idx.data().dptr<IType>());
  });
  // Calculate prefix sum
  cub::DeviceScan::InclusiveSum(d_temp_storage,
                                temp_storage_bytes,
                                prefix_sum,
                                prefix_sum,
                                idx_size,
                                Stream<gpu>::GetStream(s));
  CUDA_CALL(cudaMemcpy(&valid_num, &prefix_sum[idx_size - 1], sizeof(int32_t),
                       cudaMemcpyDeviceToHost));
  // Set the output shape forcefully
  mxnet::TShape data_shape = data.shape();
  data_shape[axis] = valid_num;
  const_cast<NDArray &>(out).Init(data_shape);
  size_t input_size = data.shape().Size();
  size_t col_size = input_size / idx.shape()[0];
  // Do the copy
  MSHADOW_TYPE_SWITCH_WITH_BOOL(out.dtype(), DType, {
    if (valid_num > 0) {
      mxnet_op::Kernel<BooleanMaskForwardKernel, gpu>::Launch(
        s, input_size, out.data().dptr<DType>(), data.data().dptr<DType>(), prefix_sum, col_size);
    }
  });
}

template<>
inline void BooleanMaskBackward<gpu>(const nnvm::NodeAttrs& attrs,
                                     const OpContext &ctx,
                                     const std::vector<NDArray> &inputs,
                                     const std::vector<OpReqType> &req,
                                     const std::vector<NDArray> &outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);
  if (req[0] == kNullOp) return;
  // inputs: {ograd, data, idx}
  // outputs: {igrad_data, igrad_idx}
  const NDArray& ograd = inputs[0];
  const NDArray& idx = inputs[2];
  const NDArray& igrad_data = outputs[0];
  Stream<gpu>* s = ctx.get_stream<gpu>();
  // Count the number of 1s in `idx`, so that we could know the output dimension
  size_t idx_size = idx.shape()[0];
  int32_t* prefix_sum = nullptr;
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  // Calculate total temporary memory size
  cub::DeviceScan::InclusiveSum(d_temp_storage,
                                temp_storage_bytes,
                                prefix_sum,
                                prefix_sum,
                                idx_size,
                                Stream<gpu>::GetStream(s));
  size_t buffer_size = idx_size * sizeof(int32_t);
  temp_storage_bytes += buffer_size;
  // Allocate memory on GPU and allocate pointer
  Tensor<gpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(temp_storage_bytes), s);
  prefix_sum = reinterpret_cast<int32_t*>(workspace.dptr_);
  d_temp_storage = workspace.dptr_ + buffer_size;
  MSHADOW_TYPE_SWITCH_WITH_BOOL(idx.dtype(), IType, {
    mxnet_op::Kernel<mshadow_op::identity_with_cast, gpu>::Launch(
      s, idx.shape()[0], prefix_sum, idx.data().dptr<IType>());
  });
  // Calculate prefix sum
  cub::DeviceScan::InclusiveSum(d_temp_storage,
                                temp_storage_bytes,
                                prefix_sum,
                                prefix_sum,
                                idx_size,
                                Stream<gpu>::GetStream(s));
  size_t input_size = igrad_data.shape().Size();
  size_t col_size = input_size / idx_size;
  // Backward pass
  MSHADOW_TYPE_SWITCH(igrad_data.dtype(), DType, {
    if (input_size > 0) {
      mxnet_op::Kernel<BooleanMaskBackwardKernel, gpu>::Launch(
        s, input_size, igrad_data.data().dptr<DType>(), req[0], ograd.data().dptr<DType>(),
        prefix_sum, col_size);
    }
  });
}

NNVM_REGISTER_OP(_contrib_boolean_mask)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FComputeEx>("FComputeEx<gpu>", BooleanMaskForward<gpu>);

NNVM_REGISTER_OP(_backward_contrib_boolean_mask)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FComputeEx>("FComputeEx<gpu>", BooleanMaskBackward<gpu>);

}  // namespace op
}  // namespace mxnet
