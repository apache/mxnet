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
 * \file  cudnn_ops.h
 * \brief cuDNN v8 ops
 */
#ifndef MXNET_OPERATOR_CUDNN_OPS_H_
#define MXNET_OPERATOR_CUDNN_OPS_H_

#include <mxnet/base.h>
#if MXNET_USE_CUDNN == 1

#include <mxnet/op_attr_types.h>

#include <algorithm>
#include <mutex>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nn/convolution-inl.h"
#include "nn/deconvolution-inl.h"

#include "../common/cuda/cudnn_cxx.h"

namespace mxnet {
namespace tuple_util {

template <size_t... Is, typename... Ts>
auto TailImpl(std::index_sequence<Is...>, const std::tuple<Ts...>& t) {
  return std::make_tuple(std::get<Is + 1>(t)...);
}

template <typename... Ts>
auto Tail(const std::tuple<Ts...>& t) {
  return TailImpl(std::make_index_sequence<sizeof...(Ts) - 1>(), t);
}

}  // namespace tuple_util
}  // namespace mxnet

// Enable tuples as keys.
namespace std {

template <>
struct hash<std::tuple<>> {
  size_t operator()(const std::tuple<>&) const {
    return 0;
  }
};

template <typename... Ts>
struct hash<std::tuple<Ts...>> {
  size_t operator()(const std::tuple<Ts...>& t) const {
    size_t ret = 0;
    ret        = dmlc::HashCombine(ret, std::get<0>(t));
    ret        = dmlc::HashCombine(ret, mxnet::tuple_util::Tail(t));
    return ret;
  }
};

}  // namespace std

namespace mxnet {
namespace op {

namespace cudnn {

struct LayoutInfo {
  size_t n_space_dims;
  bool channel_last;

  std::vector<size_t> Order() const;
  size_t ChannelIdx() const;

  template <typename T>
  std::vector<T> Strides(const std::vector<T>& dims) const {
    return cudnn_cxx::PackedStrides(Order(), dims);
  }

  template <typename T>
  void ExpandIf1d(std::vector<T>* dims, std::vector<T>* strides) const {
    if (n_space_dims != 1)
      return;
    dims->insert(dims->begin() + 2, 1);
    std::vector<size_t> order(dims->size());
    std::iota(order.begin(), order.end(), 0);
    if (channel_last)
      std::rotate(order.begin() + 1, order.begin() + 2, order.end());
    *strides = cudnn_cxx::PackedStrides(order, *dims);
  }
};

LayoutInfo GetLayoutInfo(mshadow::LayoutFlag layout);

cudnn_cxx::Descriptor MakeOpGraph(cudnnHandle_t handle, cudnn_cxx::Descriptor op);

// Make a copy of an existing execution plan with a new cuDNN handle.  Op graph re-supplied.
cudnn_cxx::Descriptor ClonePlan(cudnnHandle_t handle,
                                cudnn_cxx::Descriptor op_graph,
                                const cudnn_cxx::Descriptor& plan);

TShape ExpandChannelDims(mshadow::LayoutFlag layout, int c);

void MaybeLogSelectedPlan(const cudnn_cxx::Descriptor& plan);

// To support cached lookup and execution an operation Op must define:
//
// Op::Param - a type, collecting all data, required to create cuDNN descriptor(s), but not needed
//             for execution.
// Op::MakeKey() - a static function, which maps its arguments to a tuple - a key in the op cache.
// Op::Make() - a static function, creating all necessary cuDNN descriptors.
// Op::Clone() - a static function, creating a copy of the op's descriptors with a new cudNN handle.
// Op::Exec() - a static function, calling cudnnBackendExecute() with the prepared descriptor(s) and
//              the passed arguments.
template <typename Op, typename... Args>
bool Exec(const OpContext& ctx, const typename Op::Param& param, Args&&... args) {
  auto key = std::tuple_cat(std::make_tuple(ctx.run_ctx.ctx.dev_id),
                            Op::MakeKey(param, std::forward<Args>(args)...));
  static std::mutex mx;
  std::unique_lock<std::mutex> lk(mx);
  static std::unordered_multimap<decltype(key), const cudnn_cxx::Descriptor> op_map;
  auto match_it = [&]() {
    // Some cuDNN Op implementations require that the thread's cuDNN handle
    // (used in cudnnBackendExecute()) matches the one used in making the plan.
    const bool ignore_handles = false;
    auto range                = op_map.equal_range(key);
    auto handle               = ctx.get_stream<gpu>()->dnn_handle_;
    for (auto it = range.first; it != range.second; ++it) {
      if (ignore_handles || handle == cudnn_cxx::GetAttr<cudnnHandle_t>(
                                          it->second, CUDNN_ATTR_EXECUTION_PLAN_HANDLE)) {
        return it;
      }
    }
    // No Op exists with this handle. Make a new op, cloning from an existing op if possible.
    auto op = (range.first == range.second) ?
                  Op::Make(ctx, param, std::forward<Args>(args)...) :
                  Op::Clone(range.first->second, ctx, param, std::forward<Args>(args)...);
    return op_map.emplace(key, std::move(op));
  }();
  lk.unlock();
  if (!match_it->second)
    return false;
  Op::Exec(match_it->second, ctx, std::forward<Args>(args)...);
  return true;
}

// The subset of ConvolutionParam / DeconvolutionParam fields,
// which unambiguously identify and consturct cuDNN convolution, plus add_to flag.
struct ConvParam {
  mxnet::TShape kernel;
  mxnet::TShape stride;
  mxnet::TShape dilate;
  mxnet::TShape pad;
  uint32_t num_filter;
  uint32_t num_group;
  uint64_t workspace;
  dmlc::optional<int> cudnn_tune;
  dmlc::optional<int> layout;

  bool add_to;

  ConvParam(const ConvolutionParam& p, bool add_to);
  ConvParam(const DeconvolutionParam& p, bool add_to);
};

struct Conv {
  using Param = ConvParam;
  enum UIDs { ID_X = 1, ID_W, ID_Y };

  static auto MakeKey(const Param& p, const TBlob& x, const TBlob& w, const TBlob& y) {
    return std::make_tuple(p.kernel,
                           p.stride,
                           p.dilate,
                           p.pad,
                           p.num_filter,
                           p.num_group,
                           p.workspace,
                           p.layout,
                           p.add_to,
                           x.shape_,
                           x.type_flag_,
                           w.shape_,
                           w.type_flag_,
                           y.shape_);
  }

  static cudnn_cxx::Descriptor Make(const OpContext& ctx,
                                    const Param& param,
                                    const TBlob& x,
                                    const TBlob& w,
                                    const TBlob& y);

  static cudnn_cxx::Descriptor Clone(const cudnn_cxx::Descriptor& plan,
                                     const OpContext& ctx,
                                     const Param& param,
                                     const TBlob& x,
                                     const TBlob& w,
                                     const TBlob& y);

  static void Exec(const cudnn_cxx::Descriptor& plan,
                   const OpContext& ctx,
                   const TBlob& x,
                   const TBlob& w,
                   const TBlob& y);

 private:
  static cudnn_cxx::Descriptor MakeConvFwdOp(const OpContext& ctx,
                                             const Param& param,
                                             const TBlob& x,
                                             const TBlob& w,
                                             const TBlob& y);
};

struct ConvDgrad {
  using Param = ConvParam;
  enum UIDs { ID_W = 1, ID_DY, ID_DX };

  static auto MakeKey(const Param& p, const TBlob& w, const TBlob& dy, const TBlob& dx) {
    return std::make_tuple(p.kernel,
                           p.stride,
                           p.dilate,
                           p.pad,
                           p.num_filter,
                           p.num_group,
                           p.workspace,
                           p.layout,
                           p.add_to,
                           w.shape_,
                           w.type_flag_,
                           dy.shape_,
                           dy.type_flag_,
                           dx.shape_);
  }

  static cudnn_cxx::Descriptor Make(const OpContext& ctx,
                                    const Param& param,
                                    const TBlob& w,
                                    const TBlob& dy,
                                    const TBlob& dx);

  static cudnn_cxx::Descriptor Clone(const cudnn_cxx::Descriptor& plan,
                                     const OpContext& ctx,
                                     const Param& param,
                                     const TBlob& x,
                                     const TBlob& w,
                                     const TBlob& y);

  static void Exec(const cudnn_cxx::Descriptor& plan,
                   const OpContext& ctx,
                   const TBlob& w,
                   const TBlob& dy,
                   const TBlob& dx);

 private:
  static cudnn_cxx::Descriptor MakeConvDgradOp(const OpContext& ctx,
                                               const Param& param,
                                               const TBlob& w,
                                               const TBlob& dy,
                                               const TBlob& dx);
};

struct ConvWgrad {
  using Param = ConvParam;
  enum UIDs { ID_X = 1, ID_DY, ID_DW };

  static auto MakeKey(const Param& p, const TBlob& x, const TBlob& dy, const TBlob& dw) {
    return std::make_tuple(p.kernel,
                           p.stride,
                           p.dilate,
                           p.pad,
                           p.num_filter,
                           p.num_group,
                           p.workspace,
                           p.layout,
                           p.add_to,
                           x.shape_,
                           x.type_flag_,
                           dy.shape_,
                           dy.type_flag_,
                           dw.shape_);
  }

  static cudnn_cxx::Descriptor Make(const OpContext& ctx,
                                    const Param& param,
                                    const TBlob& x,
                                    const TBlob& dy,
                                    const TBlob& dw);

  static cudnn_cxx::Descriptor Clone(const cudnn_cxx::Descriptor& plan,
                                     const OpContext& ctx,
                                     const Param& param,
                                     const TBlob& x,
                                     const TBlob& w,
                                     const TBlob& y);

  static void Exec(const cudnn_cxx::Descriptor& plan,
                   const OpContext& ctx,
                   const TBlob& x,
                   const TBlob& dy,
                   const TBlob& dw);

 private:
  static cudnn_cxx::Descriptor MakeConvWgradOp(const OpContext& ctx,
                                               const Param& param,
                                               const TBlob& x,
                                               const TBlob& dy,
                                               const TBlob& dw);
};

bool LegacyAddBias(const OpContext& ctx, const LayoutInfo& li, const TBlob& y, const TBlob& b);

bool LegacyBiasGrad(const OpContext& ctx,
                    const LayoutInfo& li,
                    bool add_to,
                    const TBlob& db,
                    const TBlob& dy);

}  // namespace cudnn
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_CUDNN == 1

#endif  // MXNET_OPERATOR_CUDNN_OPS_H_
