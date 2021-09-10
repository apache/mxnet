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
 * \file spatial_parallel_support.cc
 * \brief Support operators for spatial parallelism
 * \author Przemyslaw Tredak
*/

#include "spatial_parallel_support.h"
#include <mxnet/base.h>
#include <mxnet/storage.h>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

void NCCLCommContainer::Init(const SpatialParallelParam& param) {
  std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
  if (NCCLCommContainer::comm_map.count(param.num_gpus) == 0) {
    auto [it, inserted] = NCCLCommContainer::comm_map.emplace(param.num_gpus, // NOLINT(*)
        std::make_unique<ncclComm_t>());
    CHECK(inserted) << "Could not insert new NCCL communicator!";
    ncclComm_t* comm = it->second.get();
    ncclUniqueId id = *(reinterpret_cast<ncclUniqueId*>(
          reinterpret_cast<void*>(param.nccl_unique_id)));
    auto result = ncclCommInitRank(comm, param.num_gpus, id, param.rank);
    CHECK_EQ(result, ncclSuccess) << "ncclCommInitRank failed!";
  }
}

DMLC_REGISTER_PARAMETER(SpatialParallelParam);

namespace {

bool SpatialParallelSplitShape(const nnvm::NodeAttrs& attrs,
                               std::vector<mxnet::TShape>* in_attrs,
                               std::vector<mxnet::TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  mxnet::TShape& in = (*in_attrs)[0];
  mxnet::TShape& out = (*out_attrs)[0];

  if (!mxnet::ndim_is_known(in)) return false;

  const SpatialParallelParam& param = nnvm::get<SpatialParallelParam>(attrs.parsed);

  mxnet::TShape desired_shape(in.ndim(), -1);
  desired_shape[0] = 1;
  SHAPE_ASSIGN_CHECK(&desired_shape, 0, in);
  if (desired_shape[1] != -1) {
    CHECK(desired_shape[1] % param.num_gpus == 0) <<
      "The outermost spatial dimension needs to be divisible by the number of GPUs!";
    desired_shape[1] /= param.num_gpus;
  }

  SHAPE_ASSIGN_CHECK(&desired_shape, 0, out);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, desired_shape);

  if (desired_shape[1] >= 0)
    desired_shape[1] *= param.num_gpus;

  SHAPE_ASSIGN_CHECK(*in_attrs, 0, desired_shape);

  return shape_is_known(in) && shape_is_known(out);
}

bool SpatialParallelAllgatherShape(const nnvm::NodeAttrs& attrs,
                                   std::vector<mxnet::TShape>* in_attrs,
                                   std::vector<mxnet::TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  mxnet::TShape& in = (*in_attrs)[0];
  mxnet::TShape& out = (*out_attrs)[0];

  if (!mxnet::ndim_is_known(in)) return false;

  const SpatialParallelParam& param = nnvm::get<SpatialParallelParam>(attrs.parsed);

  mxnet::TShape desired_shape(in.ndim(), -1);
  desired_shape[0] = 1;
  SHAPE_ASSIGN_CHECK(&desired_shape, 0, in);
  if (desired_shape[1] != -1) {
    desired_shape[1] *= param.num_gpus;
  }

  SHAPE_ASSIGN_CHECK(&desired_shape, 0, out);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, desired_shape);

  if (desired_shape[1] >= 0)
    desired_shape[1] /= param.num_gpus;

  SHAPE_ASSIGN_CHECK(*in_attrs, 0, desired_shape);

  return shape_is_known(in) && shape_is_known(out);
}

}  // namespace

NNVM_REGISTER_OP(_contrib_SpatialParallelSplit)
.describe(R"code(Split the input so that each GPU in the
in the group gets an equal part.

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SpatialParallelParam>)
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output"};
})
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs){
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    const SpatialParallelParam& param = nnvm::get<SpatialParallelParam>(attrs.parsed);
    if (param.num_gpus == 1) {
      return std::vector<bool>{true};
    } else {
      return std::vector<bool>{false};
    }
  })
.set_attr<mxnet::FInferShape>("FInferShape", SpatialParallelSplitShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_contrib_SpatialParallelAllgather"})
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kMultiGPUComm};
})
.add_argument("data", "NDArray-or-Symbol", "Data")
.add_arguments(SpatialParallelParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_SpatialParallelAllgather)
.describe(R"code(Gather the input so that each GPU in the
in the group gets the full sample.

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SpatialParallelParam>)
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output"};
})
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs){
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    const SpatialParallelParam& param = nnvm::get<SpatialParallelParam>(attrs.parsed);
    if (param.num_gpus == 1) {
      return std::vector<bool>{true};
    } else {
      return std::vector<bool>{false};
    }
  })
.set_attr<mxnet::FInferShape>("FInferShape", SpatialParallelAllgatherShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_contrib_SpatialParallelSplit"})
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kMultiGPUComm};
})
.add_argument("data", "NDArray-or-Symbol", "Data")
.add_arguments(SpatialParallelParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
