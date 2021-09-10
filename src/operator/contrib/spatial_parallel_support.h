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
 * \file spatial_parallel_support.h
 * \brief Declarations needed for spatial parallelism
 * \author Przemyslaw Tredak
*/

#ifndef MXNET_OPERATOR_CONTRIB_SPATIAL_PARALLEL_SUPPORT_H_
#define MXNET_OPERATOR_CONTRIB_SPATIAL_PARALLEL_SUPPORT_H_

#include <mxnet/base.h>

#if MXNET_USE_NCCL
#include <nccl.h>
#include <unordered_map>
#include <memory>

namespace mxnet {
namespace op {

struct SpatialParallelParam : public dmlc::Parameter<SpatialParallelParam> {
  int32_t num_gpus;
  int32_t rank;
  uintptr_t nccl_unique_id;

  DMLC_DECLARE_PARAMETER(SpatialParallelParam) {
    DMLC_DECLARE_FIELD(num_gpus).describe("Number of GPUs per sample.");
    DMLC_DECLARE_FIELD(rank).describe("Rank inside a group");
    DMLC_DECLARE_FIELD(nccl_unique_id).describe("NCCL unique ID");
  }
};

class NCCLCommContainer {
 public:
  static inline std::unordered_map<int, std::unique_ptr<ncclComm_t>> comm_map;

  static void Init(const SpatialParallelParam& param);
};

}  // namespace op
}  // namespace mxnet

#else
static_assert(false, "You need to compile with NCCL support to use spatial parallelism!");
#endif  // MXNET_USE_NCCL

#endif  // MXNET_OPERATOR_CONTRIB_SPATIAL_PARALLEL_SUPPORT_H_
