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
 * \file  cudnn_ops.cc
 * \brief cuDNN v8 ops
 */

#include "cudnn_ops.h"

#include <mxnet/base.h>
#if MXNET_USE_CUDNN == 1

#include <dmlc/parameter.h>

#include <cstdlib>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

namespace mxnet {
namespace op {

using cudnn_cxx::Descriptor;
using cudnn_cxx::GetAttr;
using cudnn_cxx::GetSomeAttrs;
using cudnn_cxx::IsCompatible;
using cudnn_cxx::MakeAvgSampler;
using cudnn_cxx::MakeFinalized;
using cudnn_cxx::PackedStrides;
using cudnn_cxx::PlanStr;

namespace cudnn {

cudnnDataType_t CudnnType(mshadow::TypeFlag dtype) {
  static std::unordered_map<mshadow::TypeFlag, cudnnDataType_t> type_map {
    {mshadow::kFloat32, CUDNN_DATA_FLOAT}, {mshadow::kFloat64, CUDNN_DATA_DOUBLE},
        {mshadow::kFloat16, CUDNN_DATA_HALF}, {mshadow::kUint8, CUDNN_DATA_UINT8},
        {mshadow::kInt8, CUDNN_DATA_INT8}, {mshadow::kInt32, CUDNN_DATA_INT32},
#if CUDNN_VERSION >= 8100
        {mshadow::kInt64, CUDNN_DATA_INT64},
#endif  // CUDNN_VERSION >= 8100
  };
  auto it = type_map.find(dtype);
  CHECK(it != type_map.end()) << "Unsupported type: " << dtype;
  return it->second;
}

std::vector<size_t> LayoutInfo::Order() const {
  std::vector<size_t> ret(n_space_dims + 2);
  std::iota(ret.begin(), ret.end(), 0);
  if (channel_last)
    std::rotate(ret.begin() + 1, ret.begin() + 2, ret.end());
  return ret;
}

size_t LayoutInfo::ChannelIdx() const {
  return channel_last ? 1 + n_space_dims : 1;
}

LayoutInfo GetLayoutInfo(mshadow::LayoutFlag layout) {
  static std::unordered_map<mshadow::LayoutFlag, LayoutInfo> layout_map{
      {mshadow::kNCW, {1, false}},
      {mshadow::kNWC, {1, true}},
      {mshadow::kNCHW, {2, false}},
      {mshadow::kNHWC, {2, true}},
      {mshadow::kNCDHW, {3, false}},
      {mshadow::kNDHWC, {3, true}},
  };
  auto it = layout_map.find(layout);
  CHECK(it != layout_map.end()) << "Unsupported layout: " << layout;
  return it->second;
}

TShape ExpandChannelDims(mshadow::LayoutFlag layout, int c) {
  auto li = GetLayoutInfo(layout);
  std::vector<int> dims(li.n_space_dims + 2, 1);
  dims[li.ChannelIdx()] = c;
  return TShape(dims.begin(), dims.end());
}

std::vector<size_t> ReverseOrder(const std::vector<size_t>& o) {
  std::vector<size_t> ret(o.size());
  for (size_t i = 0; i < ret.size(); ++i)
    ret[o[i]] = i;
  return ret;
}

std::vector<cudnnBackendNumericalNote_t> RequireNumerics() {
  std::vector<cudnnBackendNumericalNote_t> ret;
  return ret;
}

std::vector<cudnnBackendNumericalNote_t> ExcludeNumerics() {
  std::vector<cudnnBackendNumericalNote_t> ret;
  if (!dmlc::GetEnv("MXNET_CUDA_ALLOW_TENSOR_CORE", true))
    ret.push_back(CUDNN_NUMERICAL_NOTE_TENSOR_CORE);
  if (!dmlc::GetEnv("MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION", false))
    ret.push_back(CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS);
  if (!dmlc::GetEnv("MXNET_CUDNN_ALLOW_REDUCED_PRECISION_REDUCTION", true))
    ret.push_back(CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION);
  if (!dmlc::GetEnv("MXNET_CUDNN_ALLOW_FFT", true))
    ret.push_back(CUDNN_NUMERICAL_NOTE_FFT);
  if (dmlc::GetEnv("MXNET_ENFORCE_DETERMINISM", false))
    ret.push_back(CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC);
  if (!dmlc::GetEnv("MXNET_CUDNN_ALLOW_WINOGRAD", true))
    ret.push_back(CUDNN_NUMERICAL_NOTE_WINOGRAD);
  return ret;
}

Descriptor MakeTensorDesc(int64_t uid,
                          cudnnDataType_t dtype,
                          const std::vector<int64_t>& dims,
                          const std::vector<int64_t>& strides,
                          bool is_virtual) {
  int64_t alignment = 16;  // TODO(vcherepanov): ?
  return MakeFinalized(CUDNN_BACKEND_TENSOR_DESCRIPTOR,
                       CUDNN_ATTR_TENSOR_UNIQUE_ID,
                       uid,
                       CUDNN_ATTR_TENSOR_DATA_TYPE,
                       dtype,
                       CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                       alignment,
                       CUDNN_ATTR_TENSOR_DIMENSIONS,
                       dims,
                       CUDNN_ATTR_TENSOR_STRIDES,
                       strides,
                       CUDNN_ATTR_TENSOR_IS_VIRTUAL,
                       is_virtual);
}

Descriptor MakeTensorDesc(int64_t uid,
                          const TBlob& blob,
                          const LayoutInfo& li,
                          bool expand_1d,
                          bool is_virtual) {
  std::vector<int64_t> dims(blob.shape_.ndim());
  CHECK_EQ(dims.size(), li.n_space_dims + 2);
  auto rev_order = ReverseOrder(li.Order());
  for (size_t i = 0; i < dims.size(); ++i)
    dims[i] = blob.shape_[rev_order[i]];
  auto strides = li.Strides(dims);
  if (expand_1d)
    li.ExpandIf1d(&dims, &strides);
  return MakeTensorDesc(
      uid, CudnnType(static_cast<mshadow::TypeFlag>(blob.type_flag_)), dims, strides, is_virtual);
}

Descriptor MakeCTensorDescExpandDims(int64_t uid,
                                     const TBlob& b,
                                     const LayoutInfo& li,
                                     bool is_virtual) {
  std::vector<int64_t> dims(li.n_space_dims + 2, 1);
  dims[1]    = b.shape_[0];
  auto dtype = CudnnType(static_cast<mshadow::TypeFlag>(b.type_flag_));
  return MakeTensorDesc(uid, dtype, dims, li.Strides(dims), is_virtual);
}

Descriptor MakeConvDesc(const ConvParam& param, mshadow::TypeFlag dtype) {
  int64_t sdims = param.kernel.ndim();
  std::vector<int64_t> stride(param.stride.begin(), param.stride.end());
  std::vector<int64_t> dilate(param.dilate.begin(), param.dilate.end());
  std::vector<int64_t> pad(param.pad.begin(), param.pad.end());

  auto comp_type = CudnnType(dtype);
  if (comp_type == CUDNN_DATA_HALF)
    comp_type = CUDNN_DATA_FLOAT;

  if (sdims == 1) {
    // TODO(vcherepanov): remove this once cuDNN properly supports 1D convolutions.
    // For now, making spacial dims 2D: 1 x W.
    ++sdims;
    stride.insert(stride.begin(), 1);
    dilate.insert(dilate.begin(), 1);
    pad.insert(pad.begin(), 0);
  }
  return MakeFinalized(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR,
                       CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,
                       sdims,
                       CUDNN_ATTR_CONVOLUTION_COMP_TYPE,
                       comp_type,
                       CUDNN_ATTR_CONVOLUTION_CONV_MODE,
                       CUDNN_CROSS_CORRELATION,
                       CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                       stride,
                       CUDNN_ATTR_CONVOLUTION_DILATIONS,
                       dilate,
                       CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                       pad,
                       CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                       pad);
}

Descriptor MakeConvFwdOp(const Descriptor& conv,
                         const Descriptor& x,
                         const Descriptor& w,
                         const Descriptor& y,
                         bool add_to) {
  auto ret = Make(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                  CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                  conv,
                  CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                  x,
                  CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                  w,
                  CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                  y);
  if (GetAttr<cudnnDataType_t>(x, CUDNN_ATTR_TENSOR_DATA_TYPE) == CUDNN_DATA_DOUBLE) {
    SetAttrs(ret,
             CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
             1.0,
             CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
             add_to ? 1.0 : 0.0);
  } else {
    SetAttrs(ret,
             CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
             1.0f,
             CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
             add_to ? 1.0f : 0.0f);
  }
  CUDNN_CALL(cudnnBackendFinalize(ret.get()));
  return ret;
}

Descriptor Conv::MakeConvFwdOp(const OpContext& ctx,
                               const Param& param,
                               const TBlob& x,
                               const TBlob& w,
                               const TBlob& y) {
  auto dtype  = static_cast<mshadow::TypeFlag>(x.type_flag_);
  auto conv   = MakeConvDesc(param, dtype);
  auto li     = GetLayoutInfo(static_cast<mshadow::LayoutFlag>(param.layout.value()));
  auto x_desc = MakeTensorDesc(ID_X, x, li, true, false);
  auto w_desc = MakeTensorDesc(ID_W, w, li, true, false);
  auto y_desc = MakeTensorDesc(ID_Y, y, li, true, false);
  return cudnn::MakeConvFwdOp(conv, x_desc, w_desc, y_desc, param.add_to);
}

Descriptor MakeConvDgradOp(const Descriptor& conv,
                           const Descriptor& w,
                           const Descriptor& dy,
                           const Descriptor& dx,
                           bool add_to) {
  auto ret = Make(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
                  CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
                  conv,
                  CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                  w,
                  CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                  dy,
                  CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                  dx);
  if (GetAttr<cudnnDataType_t>(w, CUDNN_ATTR_TENSOR_DATA_TYPE) == CUDNN_DATA_DOUBLE) {
    SetAttrs(ret,
             CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
             1.0,
             CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
             add_to ? 1.0 : 0.0);
  } else {
    SetAttrs(ret,
             CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
             1.0f,
             CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
             add_to ? 1.0f : 0.0f);
  }
  CUDNN_CALL(cudnnBackendFinalize(ret.get()));
  return ret;
}

Descriptor ConvDgrad::MakeConvDgradOp(const OpContext& ctx,
                                      const Param& param,
                                      const TBlob& w,
                                      const TBlob& dy,
                                      const TBlob& dx) {
  auto dtype   = static_cast<mshadow::TypeFlag>(w.type_flag_);
  auto conv    = MakeConvDesc(param, dtype);
  auto li      = GetLayoutInfo(static_cast<mshadow::LayoutFlag>(param.layout.value()));
  auto w_desc  = MakeTensorDesc(ID_W, w, li, true, false);
  auto dy_desc = MakeTensorDesc(ID_DY, dy, li, true, false);
  auto dx_desc = MakeTensorDesc(ID_DX, dx, li, true, false);
  return cudnn::MakeConvDgradOp(conv, w_desc, dy_desc, dx_desc, param.add_to);
}

Descriptor MakeConvWgradOp(const Descriptor& conv,
                           const Descriptor& x,
                           const Descriptor& dy,
                           const Descriptor& dw,
                           bool add_to) {
  auto ret = Make(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
                  CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
                  conv,
                  CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                  x,
                  CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                  dy,
                  CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                  dw);
  if (GetAttr<cudnnDataType_t>(x, CUDNN_ATTR_TENSOR_DATA_TYPE) == CUDNN_DATA_DOUBLE) {
    SetAttrs(ret,
             CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
             1.0,
             CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
             add_to ? 1.0 : 0.0);
  } else {
    SetAttrs(ret,
             CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
             1.0f,
             CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
             add_to ? 1.0f : 0.0f);
  }
  CUDNN_CALL(cudnnBackendFinalize(ret.get()));
  return ret;
}

Descriptor ConvWgrad::MakeConvWgradOp(const OpContext& ctx,
                                      const Param& param,
                                      const TBlob& x,
                                      const TBlob& dy,
                                      const TBlob& dw) {
  auto dtype   = static_cast<mshadow::TypeFlag>(x.type_flag_);
  auto conv    = MakeConvDesc(param, dtype);
  auto li      = GetLayoutInfo(static_cast<mshadow::LayoutFlag>(param.layout.value()));
  auto x_desc  = MakeTensorDesc(ID_X, x, li, true, false);
  auto dy_desc = MakeTensorDesc(ID_DY, dy, li, true, false);
  auto dw_desc = MakeTensorDesc(ID_DW, dw, li, true, false);
  return cudnn::MakeConvWgradOp(conv, x_desc, dy_desc, dw_desc, param.add_to);
}

Descriptor MakeOpGraph(cudnnHandle_t handle, const std::vector<Descriptor>& ops) {
  return MakeFinalized(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
                       CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
                       handle,
                       CUDNN_ATTR_OPERATIONGRAPH_OPS,
                       ops);
}

Descriptor MakeOpGraph(cudnnHandle_t handle, Descriptor op) {
  std::vector<Descriptor> ops;
  ops.push_back(std::move(op));
  return MakeOpGraph(handle, ops);
}

Descriptor ClonePlan(cudnnHandle_t handle, Descriptor op_graph, const Descriptor& plan) {
  auto cfg =
      GetAttr(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
  auto engine     = GetAttr(cfg, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_BACKEND_ENGINE_DESCRIPTOR);
  auto engine_idx = GetAttr<int64_t>(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX);

  auto choices = GetSomeAttrs(CUDNN_KNOB_TYPE_COUNTS,
                              cfg,
                              CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                              CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);

  auto cloned_engine = MakeFinalized(CUDNN_BACKEND_ENGINE_DESCRIPTOR,
                                     CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
                                     engine_idx,
                                     CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                     op_graph);

  auto cloned_cfg = MakeFinalized(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR,
                                  CUDNN_ATTR_ENGINECFG_ENGINE,
                                  cloned_engine,
                                  CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                  choices);

  auto cloned_plan = cudnn_cxx::Make(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
                                     CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
                                     handle,
                                     CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                     cloned_cfg);
  CUDNN_CALL(cudnnBackendFinalize(cloned_plan.get()));
  return cloned_plan;
}

ConvParam::ConvParam(const ConvolutionParam& p, bool add_to)
    : kernel(p.kernel),
      stride(p.stride),
      dilate(p.dilate),
      pad(p.pad),
      num_filter(p.num_filter),
      num_group(p.num_group),
      workspace(p.workspace),
      cudnn_tune(p.cudnn_tune),
      layout(p.layout),
      add_to(add_to) {}

ConvParam::ConvParam(const DeconvolutionParam& p, bool add_to)
    : kernel(p.kernel),
      stride(p.stride),
      dilate(p.dilate),
      pad(p.pad),
      num_filter(p.num_filter),
      num_group(p.num_group),
      workspace(p.workspace),
      cudnn_tune(p.cudnn_tune),
      layout(p.layout),
      add_to(add_to) {}

void TuneWarnOnce() {
  thread_local bool done = false;
  if (!done) {
    LOG(INFO) << "Auto-tuning cuDNN op, set MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable";
    done = true;
  }
}

std::vector<Descriptor> MakeFallbackPlans(
    const std::vector<int64_t>& ixs,
    cudnnHandle_t handle,
    const Descriptor& op_graph,
    size_t workspace_limit,
    size_t* max_workspace,
    const std::unordered_set<int64_t>& excl_engines,
    const std::vector<cudnnBackendNumericalNote_t>& req_numeric,
    const std::vector<cudnnBackendNumericalNote_t>& excl_numeric
#if CUDNN_VERSION >= 8200
    ,
    const std::vector<cudnnBackendBehaviorNote_t>& req_behavior,
    const std::vector<cudnnBackendBehaviorNote_t>& excl_behavior
#endif  // CUDNN_VERSION >= 8200
) {
  std::vector<Descriptor> plans;
  if (max_workspace)
    *max_workspace = 0;
  for (auto ix : ixs) {
    if (excl_engines.count(ix))
      continue;
    auto engine = Make(CUDNN_BACKEND_ENGINE_DESCRIPTOR,
                       CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
                       op_graph,
                       CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
                       ix);
    auto err    = cudnnBackendFinalize(engine.get());
    if (err == CUDNN_STATUS_NOT_SUPPORTED || err == CUDNN_STATUS_ARCH_MISMATCH)
      continue;
    if (err != CUDNN_STATUS_SUCCESS) {
      LOG(WARNING) << "Unexpected cuDNN status: " << err << ": " << cudnnGetErrorString(err);
      continue;
    }
    auto cfg =
        MakeFinalized(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, CUDNN_ATTR_ENGINECFG_ENGINE, engine);
    auto plan = Make(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
                     CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
                     handle,
                     CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                     cfg);
    err       = cudnnBackendFinalize(plan.get());
    if (err == CUDNN_STATUS_NOT_SUPPORTED || err == CUDNN_STATUS_ARCH_MISMATCH)
      continue;
    if (err != CUDNN_STATUS_SUCCESS) {
      LOG(WARNING) << "Unexpected cuDNN status: " << err << ": " << cudnnGetErrorString(err);
      continue;
    }
    auto workspace = GetAttr<int64_t>(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE);
    if (workspace > workspace_limit)
      continue;
    auto numerical = GetSomeAttrs<cudnnBackendNumericalNote_t>(
        CUDNN_NUMERICAL_NOTE_TYPE_COUNT, engine, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE);
    if (!IsCompatible(numerical, req_numeric, excl_numeric))
      continue;
#if CUDNN_VERSION >= 8200
    auto behavior = GetSomeAttrs<cudnnBackendBehaviorNote_t>(
        CUDNN_BEHAVIOR_NOTE_TYPE_COUNT, engine, CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE);
    if (!IsCompatible(behavior, req_behavior, excl_behavior))
      continue;
#endif  // CUDNN_VERSION >= 8200
    plans.push_back(std::move(plan));
    if (max_workspace)
      *max_workspace = std::max(*max_workspace, static_cast<size_t>(workspace));
  }
  return plans;
}

cudnnBackendHeurMode_t HeurMode() {
#if CUDNN_VERSION >= 8100
  int default_mode = cudnnGetVersion() < 8100 ? CUDNN_HEUR_MODE_INSTANT : CUDNN_HEUR_MODE_B;
#else
  int default_mode = CUDNN_HEUR_MODE_INSTANT;
#endif  // CUDNN_VERSION >= 8100
  return static_cast<cudnnBackendHeurMode_t>(dmlc::GetEnv("MXNET_CUDNN_HEUR_MODE", default_mode));
}

std::string ConvParamStr(const ConvParam& param) {
  std::ostringstream ss;
  ss << mshadow::toString(static_cast<mshadow::LayoutFlag>(param.layout.value()));
  ss << " kernel: " << param.kernel;
  ss << " stride: " << param.stride;
  ss << " dilate: " << param.dilate;
  ss << " pad: " << param.pad;
  ss << " num_filter: " << param.num_filter;
  ss << " num_group: " << param.num_group;
  ss << " workspace: " << param.workspace;
  return ss.str();
}

size_t GetWorkspace(const Descriptor& plan) {
  return GetAttr<int64_t>(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE);
}

Storage::Handle FailsafeAlloc(size_t workspace_size) {
  return Storage::Get()->Alloc(workspace_size, Context::GPU(), true);
}

Storage::Handle AllocWorkspace(std::vector<Descriptor>* plans, size_t* workspace_size) {
  Storage::Handle workspace;
  size_t alloc_size = *workspace_size;
  while ((workspace = FailsafeAlloc(alloc_size)).dptr == nullptr && alloc_size > 0) {
    // Remove any plan whose workspace_size equals the failed allocation size
    auto hasMaxWorkspace = [alloc_size](auto const& plan) {
      return GetWorkspace(plan) == alloc_size;
    };
    plans->erase(std::remove_if(plans->begin(), plans->end(), hasMaxWorkspace), plans->end());
    // Calculate new maximum workspace_size for remaining plans
    alloc_size = 0;
    for (auto& plan : *plans)
      alloc_size = std::max(alloc_size, GetWorkspace(plan));
  }
  *workspace_size = alloc_size;
  return workspace;
}

std::unordered_set<int64_t> ExcludeEngines(const std::string& env_var) {
  std::string engines = dmlc::GetEnv(env_var.c_str(), std::string());
  std::replace(engines.begin(), engines.end(), ',', ' ');
  std::istringstream ss(engines);
  return std::unordered_set<int64_t>(std::istream_iterator<int64_t>(ss),
                                     std::istream_iterator<int64_t>());
}

Descriptor SelectPlan(const OpContext& ctx,
                      const ConvParam& param,
                      Descriptor op,
                      size_t n_fallbacks,
                      const std::function<std::string()>& make_op_str,
                      const std::vector<int64_t>& ids,
                      const std::vector<void*>& tensor_ptrs,
                      int64_t out_size,
                      const std::string& excl_engines_var) {
  auto s = ctx.get_stream<gpu>();
  auto op_graph = MakeOpGraph(s->dnn_handle_, std::move(op));

  int verbose = dmlc::GetEnv("MXNET_CUDNN_ALGO_VERBOSE_LEVEL", 0);
  if (verbose > 0)
    LOG(INFO) << "Selecting plan for " << make_op_str() << ":";

  auto tune = param.cudnn_tune ?
                  param.cudnn_tune.value() :
                  dmlc::GetEnv("MXNET_CUDNN_AUTOTUNE_DEFAULT", static_cast<int>(conv::kLimited));
  size_t workspace_size = 0;
  size_t workspace_limit =
      tune != conv::kFastest ? param.workspace << 20 : std::numeric_limits<size_t>::max();
  auto excl_engines = ExcludeEngines(excl_engines_var);
  auto plans        = GetPlans(HeurMode(),
                        s->dnn_handle_,
                        op_graph,
                        workspace_limit,
                        &workspace_size,
                        excl_engines,
                        RequireNumerics(),
                        ExcludeNumerics(),
#if CUDNN_VERSION >= 8200
                        {},
                        {},
#endif  // CUDNN_VERSION >= 8200
                        verbose > 1);
  Storage::Handle out_space;
  auto ptrs = tensor_ptrs;
  if (tune != conv::kOff && param.add_to) {
    // Cannot trash output tensor while auto-tuning.
    out_space = FailsafeAlloc(out_size);
    if (out_space.dptr)
      ptrs.back() = out_space.dptr;
  }
  // Todo:
  //     - should we be able to ask the tempspace for it's current size, then
  //       alloc the workspace from the tempspace if its current size > workspace_size?
  auto workspace = AllocWorkspace(&plans, &workspace_size);

  if (plans.empty()) {
    std::vector<int64_t> ixs(n_fallbacks);
    std::iota(ixs.begin(), ixs.end(), 0);
#if CUDNN_VERSION >= 8200
    plans = MakeFallbackPlans(ixs,
                              s->dnn_handle_,
                              op_graph,
                              workspace_limit,
                              &workspace_size,
                              excl_engines,
                              RequireNumerics(),
                              ExcludeNumerics(),
                              {},
                              {});
#else
    plans = MakeFallbackPlans(ixs,
                              s->dnn_handle_,
                              op_graph,
                              workspace_limit,
                              &workspace_size,
                              excl_engines,
                              RequireNumerics(),
                              ExcludeNumerics());
#endif  // CUDNN_VERSION >= 8200
    workspace = AllocWorkspace(&plans, &workspace_size);
    CHECK(!plans.empty());
    LOG(WARNING) << "Using fallback engine(s) for " << make_op_str();
  }

  if (tune == conv::kOff || plans.size() == 1 || (param.add_to && !out_space.dptr)) {
    if (verbose > 0)
      LOG(INFO) << " " << PlanStr(plans[0]);
    Storage::Get()->DirectFree(out_space);
    Storage::Get()->DirectFree(workspace);
    return std::move(plans[0]);
  }

  TuneWarnOnce();
  size_t n      = verbose > 1 ? plans.size() : 1;
  auto var_pack = MakeFinalized(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
                                CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                ids,
                                CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                ptrs,
                                CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                workspace.dptr);
  auto top      = FindTopPlans(std::move(plans), n, s->dnn_handle_, var_pack, MakeAvgSampler(3));
  Storage::Get()->DirectFree(out_space);
  Storage::Get()->DirectFree(workspace);
  auto str_time = [](float t) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << t;
    return ss.str();
  };
  for (size_t i = 0; verbose > 0 && i < top.size(); ++i) {
    std::ostringstream ss;
    auto prefix = i == 0 ? " * " : "   ";
    ss << prefix << top[i].heur_i << ") " << str_time(top[i].time) << "ms " << PlanStr(top[i].plan);
    LOG(INFO) << ss.str();
  }
  return std::move(top[0].plan);
}

size_t Size(const TBlob& t) {
  return t.Size() * mshadow::mshadow_sizeof(t.type_flag_);
}

// TODO(vcherepanov): remove these, once fallbacks are received as a heuristics mode in 8.3
enum MaxFallbacks { kMaxConvFallbacks = 58, kMaxDgradFallbacks = 63, kMaxWgradFallbacks = 62 };

cudnn_cxx::Descriptor Conv::Make(const OpContext& ctx,
                                 const Param& param,
                                 const TBlob& x,
                                 const TBlob& w,
                                 const TBlob& y) {
  auto conv_fwd = MakeConvFwdOp(ctx, param, x, w, y);

  auto make_op_str = [&param, &x]() {
    std::ostringstream ss;
    ss << "fprop " << mshadow::dtype_string(x.type_flag_) << " " << ConvParamStr(param);
    return ss.str();
  };

  std::vector<int64_t> ids{ID_X, ID_W, ID_Y};
  std::vector<void*> ptrs{x.dptr_, w.dptr_, y.dptr_};

  return SelectPlan(ctx,
                    param,
                    std::move(conv_fwd),
                    kMaxConvFallbacks,
                    make_op_str,
                    ids,
                    ptrs,
                    Size(y),
                    "MXNET_CUDNN_DISABLED_CONV_FWD_ENGINES");
}

cudnn_cxx::Descriptor Conv::Clone(const cudnn_cxx::Descriptor& plan,
                                  const OpContext& ctx,
                                  const Param& param,
                                  const TBlob& x,
                                  const TBlob& w,
                                  const TBlob& y) {
  auto conv_fwd    = MakeConvFwdOp(ctx, param, x, w, y);
  auto handle      = ctx.get_stream<gpu>()->dnn_handle_;
  auto op_graph    = MakeOpGraph(handle, std::move(conv_fwd));
  auto cloned_plan = ClonePlan(handle, std::move(op_graph), plan);
  return cloned_plan;
}

void Conv::Exec(const cudnn_cxx::Descriptor& plan,
                const OpContext& ctx,
                const TBlob& x,
                const TBlob& w,
                const TBlob& y) {
  auto s              = ctx.get_stream<gpu>();
  auto workspace_size = GetAttr<int64_t>(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE);
  auto workspace      = ctx.requested[0].get_space_internal(workspace_size, "Conv");

  std::vector<int64_t> ids{ID_X, ID_W, ID_Y};
  std::vector<void*> ptrs{x.dptr_, w.dptr_, y.dptr_};
  auto var_pack = MakeFinalized(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
                                CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                ids,
                                CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                ptrs,
                                CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                workspace);
  CUDNN_CALL(cudnnBackendExecute(s->dnn_handle_, plan.get(), var_pack.get()));
}

cudnn_cxx::Descriptor ConvDgrad::Make(const OpContext& ctx,
                                      const Param& param,
                                      const TBlob& w,
                                      const TBlob& dy,
                                      const TBlob& dx) {
  auto conv_dgrad = MakeConvDgradOp(ctx, param, w, dy, dx);

  auto make_op_str = [&param, &dx]() {
    std::ostringstream ss;
    ss << "dgrad " << mshadow::dtype_string(dx.type_flag_) << " " << ConvParamStr(param);
    return ss.str();
  };

  std::vector<int64_t> ids{ID_W, ID_DY, ID_DX};
  std::vector<void*> ptrs{w.dptr_, dy.dptr_, dx.dptr_};

  return SelectPlan(ctx,
                    param,
                    std::move(conv_dgrad),
                    kMaxDgradFallbacks,
                    make_op_str,
                    ids,
                    ptrs,
                    Size(dx),
                    "MXNET_CUDNN_DISABLED_CONV_DGRAD_ENGINES");
}

cudnn_cxx::Descriptor ConvDgrad::Clone(const cudnn_cxx::Descriptor& plan,
                                       const OpContext& ctx,
                                       const Param& param,
                                       const TBlob& w,
                                       const TBlob& dy,
                                       const TBlob& dx) {
  auto conv_dgrad  = MakeConvDgradOp(ctx, param, w, dy, dx);
  auto handle      = ctx.get_stream<gpu>()->dnn_handle_;
  auto op_graph    = MakeOpGraph(handle, std::move(conv_dgrad));
  auto cloned_plan = ClonePlan(handle, std::move(op_graph), plan);
  return cloned_plan;
}

void ConvDgrad::Exec(const cudnn_cxx::Descriptor& plan,
                     const OpContext& ctx,
                     const TBlob& w,
                     const TBlob& dy,
                     const TBlob& dx) {
  auto s              = ctx.get_stream<gpu>();
  auto workspace_size = GetAttr<int64_t>(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE);
  auto workspace      = ctx.requested[0].get_space_internal(workspace_size, "ConvDgrad");

  std::vector<int64_t> ids{ID_W, ID_DY, ID_DX};
  std::vector<void*> ptrs{w.dptr_, dy.dptr_, dx.dptr_};
  auto var_pack = MakeFinalized(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
                                CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                ids,
                                CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                ptrs,
                                CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                workspace);
  CUDNN_CALL(cudnnBackendExecute(s->dnn_handle_, plan.get(), var_pack.get()));
}

cudnn_cxx::Descriptor ConvWgrad::Make(const OpContext& ctx,
                                      const Param& param,
                                      const TBlob& x,
                                      const TBlob& dy,
                                      const TBlob& dw) {
  auto conv_wgrad = MakeConvWgradOp(ctx, param, x, dy, dw);

  auto make_op_str = [&param, &x]() {
    std::ostringstream ss;
    ss << "wgrad " << mshadow::dtype_string(x.type_flag_) << " " << ConvParamStr(param);
    return ss.str();
  };

  std::vector<int64_t> ids{ID_X, ID_DY, ID_DW};
  std::vector<void*> ptrs{x.dptr_, dy.dptr_, dw.dptr_};

  return SelectPlan(ctx,
                    param,
                    std::move(conv_wgrad),
                    kMaxWgradFallbacks,
                    make_op_str,
                    ids,
                    ptrs,
                    Size(dw),
                    "MXNET_CUDNN_DISABLED_CONV_WGRAD_ENGINES");
}

cudnn_cxx::Descriptor ConvWgrad::Clone(const cudnn_cxx::Descriptor& plan,
                                       const OpContext& ctx,
                                       const Param& param,
                                       const TBlob& x,
                                       const TBlob& dy,
                                       const TBlob& dw) {
  auto conv_wgrad  = MakeConvWgradOp(ctx, param, x, dy, dw);
  auto handle      = ctx.get_stream<gpu>()->dnn_handle_;
  auto op_graph    = MakeOpGraph(handle, std::move(conv_wgrad));
  auto cloned_plan = ClonePlan(handle, std::move(op_graph), plan);
  return cloned_plan;
}

void ConvWgrad::Exec(const cudnn_cxx::Descriptor& plan,
                     const OpContext& ctx,
                     const TBlob& x,
                     const TBlob& dy,
                     const TBlob& dw) {
  auto s              = ctx.get_stream<gpu>();
  auto workspace_size = GetAttr<int64_t>(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE);
  auto workspace      = ctx.requested[0].get_space_internal(workspace_size, "ConvWgrad");

  std::vector<int64_t> ids{ID_X, ID_DY, ID_DW};
  std::vector<void*> ptrs{x.dptr_, dy.dptr_, dw.dptr_};
  auto var_pack = MakeFinalized(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
                                CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                ids,
                                CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                ptrs,
                                CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                workspace);
  CUDNN_CALL(cudnnBackendExecute(s->dnn_handle_, plan.get(), var_pack.get()));
}

struct LegacyTensorDestroyer {
  using pointer = cudnnTensorDescriptor_t;

  void operator()(cudnnTensorDescriptor_t desc) {
    CUDNN_CALL_NONFATAL(cudnnDestroyTensorDescriptor(desc));
  }
};

using LegacyTensor = std::unique_ptr<cudnnTensorDescriptor_t, LegacyTensorDestroyer>;

LegacyTensor MakeLegacyTensor() {
  cudnnTensorDescriptor_t desc{};
  CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
  return LegacyTensor(desc);
}

union ScalingParam {
  double d;
  float f;
};

std::pair<ScalingParam, ScalingParam> AlphaBeta(int type_flag, double init_a, double init_b) {
  ScalingParam a, b;
  switch (type_flag) {
    case kFloat64:
      a.d = init_a;
      b.d = init_b;
      break;
    case kFloat32:  // fallthrough
    case kFloat16:
      a.f = init_a;
      b.f = init_b;
      break;
    default:
      LOG(FATAL) << "Unexpected type: " << type_flag;
  }
  return {a, b};
}

void SetLegacyTensor(cudnnTensorDescriptor_t desc, const TBlob& blob, const LayoutInfo& li) {
  std::vector<int> dims(blob.shape_.ndim());
  CHECK_EQ(dims.size(), li.n_space_dims + 2);
  auto rev_order = ReverseOrder(li.Order());
  for (size_t i = 0; i < dims.size(); ++i)
    dims[i] = blob.shape_[rev_order[i]];
  auto strides = li.Strides(dims);
  li.ExpandIf1d(&dims, &strides);
  auto type = static_cast<mshadow::TypeFlag>(blob.type_flag_);
  CUDNN_CALL(cudnnSetTensorNdDescriptor(desc, CudnnType(type), dims.size(), &dims[0], &strides[0]));
}

void SetLegacyCTensorExpandDims(cudnnTensorDescriptor_t desc,
                                const TBlob& blob,
                                const LayoutInfo& li) {
  std::vector<int> dims(li.n_space_dims + 2, 1);
  dims[1] = blob.shape_[0];
  std::vector<int> strides(dims.size(), 1);
  strides[0] = blob.shape_[0];
  li.ExpandIf1d(&dims, &strides);
  auto type = static_cast<mshadow::TypeFlag>(blob.type_flag_);
  CUDNN_CALL(cudnnSetTensorNdDescriptor(desc, CudnnType(type), dims.size(), &dims[0], &strides[0]));
}

bool LegacyAddBias(const OpContext& ctx, const LayoutInfo& li, const TBlob& y, const TBlob& b) {
  thread_local auto y_desc = MakeLegacyTensor();
  thread_local auto b_desc = MakeLegacyTensor();

  auto s             = ctx.get_stream<gpu>();
  auto [alpha, beta] = AlphaBeta(y.type_flag_, 1.0, 1.0);  // NOLINT(whitespace/braces)

  SetLegacyTensor(y_desc.get(), y, li);
  SetLegacyCTensorExpandDims(b_desc.get(), b, li);

  auto err =
      cudnnAddTensor(s->dnn_handle_, &alpha, b_desc.get(), b.dptr_, &beta, y_desc.get(), y.dptr_);
  if (err == CUDNN_STATUS_NOT_SUPPORTED)
    return false;
  CHECK_EQ(err, CUDNN_STATUS_SUCCESS);
  return true;
}

bool LegacyBiasGrad(const OpContext& ctx,
                    const LayoutInfo& li,
                    bool add_to,
                    const TBlob& db,
                    const TBlob& dy) {
  thread_local auto db_desc = MakeLegacyTensor();
  thread_local auto dy_desc = MakeLegacyTensor();

  auto s             = ctx.get_stream<gpu>();
  auto [alpha, beta] = AlphaBeta(dy.type_flag_, 1.0, add_to ? 1.0 : 0.0);  // NOLINT(*)

  SetLegacyCTensorExpandDims(db_desc.get(), db, li);
  SetLegacyTensor(dy_desc.get(), dy, li);

  auto err = cudnnConvolutionBackwardBias(
      s->dnn_handle_, &alpha, dy_desc.get(), dy.dptr_, &beta, db_desc.get(), db.dptr_);
  if (err == CUDNN_STATUS_NOT_SUPPORTED)
    return false;
  CHECK_EQ(err, CUDNN_STATUS_SUCCESS);
  return true;
}

}  // namespace cudnn
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_CUDNN == 1
