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
 * Copyright (c) 2020 by Contributors
 * \file spatial_parallel_support.cu
 * \brief Support operators for spatial parallelism
 * \author Przemyslaw Tredak
*/

#include "spatial_parallel_support.h"
#include <mxnet/base.h>
#include <mxnet/storage.h>
#include <mutex>
#include <vector>
#include "../operator_common.h"
#include "../../common/utils.h"
#include "../tensor/elemwise_binary_op.h"

namespace mxnet {
namespace op {

void SpatialParallelSplitCompute(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  if (req[0] == OpReqType::kNullOp) return;
  const SpatialParallelParam& param = nnvm::get<SpatialParallelParam>(attrs.parsed);
  if (param.num_gpus == 1 && req[0] == OpReqType::kWriteInplace) return;
  const auto& in = inputs[0];
  const auto& out = outputs[0];
  CHECK(param.num_gpus == 1 || in.shape_[0] == 1) <<
    "SpatialParallelSplit supports only a single sample when number of GPUs is greater than 1.";
  CHECK(in.shape_[1] % param.num_gpus == 0) <<
    "The outermost spatial dimension needs to be divisible by the number of GPUs.";
  const index_t stride = in.shape_.Size() / in.shape_[1];
  const index_t dim_per_gpu = in.shape_[1] / param.num_gpus;
  const index_t type_size = common::mshadow_type_info(in.type_flag_).size;
  const index_t size = dim_per_gpu * stride * type_size;
  const index_t start = param.rank * size;
  if (req[0] != OpReqType::kAddTo) {
    cudaStream_t stream = mshadow::Stream<gpu>::GetStream(ctx.get_stream<gpu>());
    cudaMemcpyAsync(out.dptr_,
                    reinterpret_cast<uint8_t*>(in.dptr_) + start,
                    size,
                    cudaMemcpyDeviceToDevice,
                    stream);
  } else {
    TBlob new_in(reinterpret_cast<uint8_t*>(in.dptr_) + start, out.shape_,
                 in.dev_mask(), in.type_flag_, in.dev_id());
    ElemwiseBinaryRTCCompute {"add"}(attrs, ctx, {new_in, out}, {kWriteInplace}, {out});
  }
}

void SpatialParallelAllgatherCompute(const nnvm::NodeAttrs& attrs,
                                     const OpContext& ctx,
                                     const std::vector<TBlob>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<TBlob>& outputs) {
  const SpatialParallelParam& param = nnvm::get<SpatialParallelParam>(attrs.parsed);
  if (req[0] == OpReqType::kNullOp) return;
  if (param.num_gpus == 1 && req[0] == OpReqType::kWriteInplace) return;

  NCCLCommContainer::Init(param);

  std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
  ncclComm_t comm = *(NCCLCommContainer::comm_map.at(param.num_gpus));
  const index_t size = inputs[0].shape_.Size() *
                       common::mshadow_type_info(inputs[0].type_flag_).size;
  if (req[0] != OpReqType::kAddTo) {
    ncclResult_t result = ncclAllGather(inputs[0].dptr_,
                                        outputs[0].dptr_,
                                        size, ncclInt8,
                                        comm,
                                        mshadow::Stream<gpu>::GetStream(ctx.get_stream<gpu>()));
    CHECK_EQ(result, ncclSuccess) << "NCCL Allgather failed!";
  } else {
    LOG(FATAL) << "kAddTo not supported yet!";
  }
}

NNVM_REGISTER_OP(_contrib_SpatialParallelSplit)
.set_attr<FCompute>("FCompute<gpu>", SpatialParallelSplitCompute);

NNVM_REGISTER_OP(_contrib_SpatialParallelAllgather)
.set_attr<FCompute>("FCompute<gpu>", SpatialParallelAllgatherCompute);

}  // namespace op
}  // namespace mxnet
