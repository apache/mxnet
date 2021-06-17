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
 * \file spatial_parallel_convolution-inl.h
 * \brief Spatial parallel convolution
 * \author Przemyslaw Tredak
*/
#ifndef MXNET_OPERATOR_CONTRIB_NN_SPATIAL_PARALLEL_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_CONTRIB_NN_SPATIAL_PARALLEL_CONVOLUTION_INL_H_

#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../../operator_common.h"


namespace mxnet {
namespace op {

namespace spconv {
enum ConvolutionOpInputs {kData, kWeight, kBias};
enum ConvolutionOpOutputs {kOut};
enum ConvolutionOpResource {kTempSpace};
enum ConvolutionOpCudnnTune {kOff, kLimited, kFastest};
}

struct SpatialParallelConvolutionParam : public dmlc::Parameter<SpatialParallelConvolutionParam> {
  mxnet::TShape kernel;
  mxnet::TShape stride;
  mxnet::TShape dilate;
  mxnet::TShape pad;
  uint32_t num_filter;
  uint32_t num_group;
  uint64_t workspace;
  bool no_bias;
  dmlc::optional<int> cudnn_tune;
  bool cudnn_off;
  dmlc::optional<int> layout;
  int32_t num_gpus;
  int32_t rank;
  uintptr_t nccl_unique_id;
  DMLC_DECLARE_PARAMETER(SpatialParallelConvolutionParam) {
    DMLC_DECLARE_FIELD(kernel).describe("Convolution kernel size: (w,), (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(stride).set_default(mxnet::TShape(0, 0))
    .describe("Convolution stride: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(dilate).set_default(mxnet::TShape(0, 0))
    .describe("Convolution dilate: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(pad).set_default(mxnet::TShape(0, 0))
    .describe("Zero pad for convolution: (w,), (h, w) or (d, h, w). Defaults to no padding.");
    DMLC_DECLARE_FIELD(num_filter).set_lower_bound(1)
    .describe("Convolution filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
    .describe("Number of group partitions.");
    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_lower_bound(0)
    .describe("Maximum temporary workspace allowed (MB) in convolution."
              "This parameter has two usages. When CUDNN is not used, it determines the "
              "effective batch size of the convolution kernel. When CUDNN is used, it controls "
              "the maximum temporary storage used for tuning the best CUDNN kernel when "
              "`limited_workspace` strategy is used.");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(cudnn_tune)
    .add_enum("off", spconv::kOff)
    .add_enum("limited_workspace", spconv::kLimited)
    .add_enum("fastest", spconv::kFastest)
    .set_default(dmlc::optional<int>())
        .describe("Whether to pick convolution algo by running performance test.");
    DMLC_DECLARE_FIELD(cudnn_off).set_default(false)
    .describe("Turn off cudnn for this layer.");
    DMLC_DECLARE_FIELD(layout)
    .add_enum("NCW", mshadow::kNCW)
    .add_enum("NCHW", mshadow::kNCHW)
    .add_enum("NCDHW", mshadow::kNCDHW)
    .add_enum("NWC", mshadow::kNWC)
    .add_enum("NHWC", mshadow::kNHWC)
    .add_enum("NDHWC", mshadow::kNDHWC)
    .set_default(dmlc::optional<int>())
    .describe("Set layout for input, output and weight. Empty for\n    "
              "default layout: NWC for 1d, NHWC for 2d and NDHWC for 3d.");
    DMLC_DECLARE_FIELD(num_gpus).describe("Number of GPUs per sample.");
    DMLC_DECLARE_FIELD(rank).describe("Rank inside a group");
    DMLC_DECLARE_FIELD(nccl_unique_id).describe("NCCL unique ID");
  }
  // Adjusts kernel size for effects of dilation in the dimension `dim`.
  index_t DilatedKernelSize(int dim) const {
    return 1 + (kernel[dim] - 1) * dilate[dim];
  }

  bool operator==(const SpatialParallelConvolutionParam& other) const {
    return this->kernel == other.kernel &&
           this->stride == other.stride &&
           this->dilate == other.dilate &&
           this->pad == other.pad &&
           this->num_filter == other.num_filter &&
           this->num_group == other.num_group &&
           this->workspace == other.workspace &&
           this->no_bias == other.no_bias &&
           this->cudnn_tune == other.cudnn_tune &&
           this->cudnn_off == other.cudnn_off &&
           this->layout == other.layout &&
           this->num_gpus == other.num_gpus;
  }
};

void SPConvolutionParamParser(nnvm::NodeAttrs* attrs);

using SPConvSignature = ParamOpSign<SpatialParallelConvolutionParam>;
}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::SpatialParallelConvolutionParam> {
  size_t operator()(const mxnet::op::SpatialParallelConvolutionParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.kernel);
    ret = dmlc::HashCombine(ret, val.stride);
    ret = dmlc::HashCombine(ret, val.dilate);
    ret = dmlc::HashCombine(ret, val.pad);
    ret = dmlc::HashCombine(ret, val.num_filter);
    ret = dmlc::HashCombine(ret, val.num_group);
    ret = dmlc::HashCombine(ret, val.workspace);
    ret = dmlc::HashCombine(ret, val.no_bias);
    ret = dmlc::HashCombine(ret, val.cudnn_tune);
    ret = dmlc::HashCombine(ret, val.cudnn_off);
    ret = dmlc::HashCombine(ret, val.layout);
    ret = dmlc::HashCombine(ret, val.num_gpus);
    ret = dmlc::HashCombine(ret, val.rank);

    return ret;
  }
};
}  // namespace std

namespace mxnet {
namespace op {

template<typename xpu>
void SPConvolutionCompute(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx, const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
}

template<typename xpu>
void SPConvolutionGradCompute(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx, const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_NN_SPATIAL_PARALLEL_CONVOLUTION_INL_H_
