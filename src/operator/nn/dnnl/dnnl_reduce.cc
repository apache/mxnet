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
 * \file dnnl_reduce.cc
 * \brief
 */

#if MXNET_USE_ONEDNN == 1

#include "./dnnl_reduce-inl.h"
#include "../../numpy/np_broadcast_reduce_op.h"

namespace mxnet {
namespace op {

template <>
NumpyReduceAxesParam ConvertReduceParamsToNumpy<ReduceAxesParam>(
    const ReduceAxesParam& original_param,
    const NDArray& input,
    const NDArray& output) {
  NumpyReduceAxesParam numpy_param;
  numpy_param.axis = dmlc::optional<mxnet::Tuple<int>>();
  if (original_param.axis.has_value()) {
    mxnet::Tuple<int> axes(original_param.axis.value().begin(), original_param.axis.value().end());
    std::sort(axes.begin(), axes.end());

    if (original_param.exclude) {
      const size_t in_ndim = input.shape().ndim();
      mxnet::Tuple<int> inverted_axes(in_ndim - axes.ndim(), -1);
      for (int i = 0, j = 0; i < input.shape().ndim(); i++) {
        if (j >= axes.ndim() || i != axes[j]) {
          inverted_axes[i - j] = i;
        } else {
          j++;
        }
      }
      numpy_param.axis = inverted_axes;
    } else {
      numpy_param.axis = axes;
    }
  }
  numpy_param.keepdims = original_param.keepdims;
  numpy_param.dtype    = dmlc::optional<int>(output.dtype());
  return numpy_param;
}

template <>
NumpyReduceAxesParam ConvertReduceParamsToNumpy<NumpyReduceAxesParam>(
    const NumpyReduceAxesParam& original_param,
    const NDArray& input,
    const NDArray& output) {
  return original_param;
}

mxnet::Tuple<int> CanonicalizeAndSortAxes(const NDArray& input,
                                          const NumpyReduceAxesParam& param,
                                          mxnet::Tuple<int> original_axes) {
  int in_ndim = input.shape().ndim();
  mxnet::Tuple<int> axes(param.axis.value());
  for (int i = 0; i < axes.ndim(); i++) {
    if (axes[i] < 0) {
      axes[i] += in_ndim;
    }
  }
  std::sort(axes.begin(), axes.end());
  return axes;
}

// Support for https://oneapi-src.github.io/oneDNN/v2.6/dev_guide_reduction.html
bool SupportDNNLReduceImpl(const NumpyReduceAxesParam& param,
                           const NDArray& input,
                           const NDArray& output) {
  bool param_supported = true;
  if (param.axis.has_value()) {
    auto axes    = CanonicalizeAndSortAxes(input, param, param.axis.value());
    int last_dim = *(axes.end() - 1);
    if (last_dim != input.shape().ndim() - 1) {
      // oneDNN (v2.3.2) not optimized case
      return false;
    } else {
      for (int i = 0; i < axes.ndim(); i++) {
        // oneDNN doesnt support reduction of axes with dimension 1
        // use oneDNN implementation only when dealing with consecutive trailing dimensions
        if (input.shape()[axes[i]] == 1 || (last_dim - axes[i]) != (axes.ndim() - 1 - i)) {
          return false;
        }
      }
    }

    // if `axis = ()` it is identity op and it is not supported by oneDNN
    param_supported = param.axis.value().ndim() > 0;
  }
  // initial value not supported by oneDNN
  param_supported = param_supported && !param.initial.has_value();
  // oneDNN does not support reduction of tensors with size equal to 1
  return param_supported && input.shape().Size() > 1 &&
         SupportDNNL<DNNLTypeMode::FloatTypes>(input);
}

void DNNLReduceForwardImpl(const NumpyReduceAxesParam& param,
                           const OpContext& ctx,
                           const NDArray& in_data,
                           const OpReqType& req,
                           const NDArray& out_data,
                           const dnnl::algorithm reduction_alg) {
  if (req == kNullOp)
    return;
  CHECK_NE(req, kAddTo);

  const bool is_train = ctx.is_train;
  const auto tensors  = DNNLReduceFwd::Tensors(in_data, out_data);
  const auto fwd      = DNNLReduceFwd::GetCached(param, tensors, is_train, reduction_alg);
  fwd.Execute(tensors);
}

DNNLReduceFwd::Tensors::Tensors(const NDArray& data, const NDArray& output)
    : data(data), out(output) {}

typedef ParamOpSign<NumpyReduceAxesParam> DNNLReduceSignature;
DNNLReduceFwd DNNLReduceFwd::GetCached(const NumpyReduceAxesParam& param,
                                       const Tensors& tensors,
                                       const bool is_train,
                                       const dnnl::algorithm reduction_alg) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLReduceSignature, DNNLReduceFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLReduceSignature, DNNLReduceFwd, OpHash> fwds;
#endif

  DNNLReduceSignature key(param);
  key.AddSign(is_train);
  key.AddSign(tensors.data);
  key.AddSign(tensors.out);
  key.AddSign(static_cast<int>(reduction_alg));
  if (param.axis.has_value()) {
    TShape ax(param.axis.value().begin(), param.axis.value().end());
    key.AddSign(ax);
  }
  if (param.dtype.has_value())
    key.AddSign(param.dtype.value());

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    DNNLReduceFwd fwd(param, tensors, is_train, reduction_alg);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

DNNLReduceFwd::DNNLReduceFwd(const NumpyReduceAxesParam& param,
                             const Tensors& tensors,
                             const bool is_train,
                             const dnnl::algorithm reduction_alg) {
  auto input_mem        = tensors.data.GetDNNLData();
  auto input_md         = input_mem->get_desc();
  const auto in_shape   = tensors.data.shape();
  const size_t in_ndim  = in_shape.ndim();
  const size_t out_ndim = tensors.out.shape().ndim();
  const auto out_dtype  = get_dnnl_type(tensors.out.dtype());
  dnnl::memory::desc out_md;

  if (in_ndim == out_ndim) {
    auto out_mem = tensors.out.GetDNNLData();
    out_md       = out_mem->get_desc();
  } else {
    if (param.axis.has_value()) {
      auto axes = CanonicalizeAndSortAxes(tensors.data, param, param.axis.value());
      dnnl::memory::dims out_shape(in_ndim);
      int axis_indice = 0;
      for (int i = 0; i < in_ndim; i++) {
        if (axis_indice < axes.ndim() && axes[axis_indice] == i) {
          out_shape[i] = 1;
          axis_indice++;
        } else {
          out_shape[i] = in_shape[i];
        }
      }
      out_md = dnnl::memory::desc(out_shape, out_dtype, dnnl::memory::format_tag::any);

    } else {
      // global reduction
      dnnl::memory::dims out_shape(in_ndim, 1);
      out_md = dnnl::memory::desc(out_shape, out_dtype, dnnl::memory::format_tag::any);
    }
  }

  reduce_pd  = std::make_shared<reduce_fwd_pd_t>(GetReduceFwdPd(input_md, out_md, reduction_alg));
  reduce_fwd = std::make_shared<reduce_fwd_t>(*reduce_pd);
}

reduce_fwd_pd_t DNNLReduceFwd::GetReduceFwdPd(const dnnl::memory::desc& input_md,
                                              const dnnl::memory::desc& output_md,
                                              const dnnl::algorithm reduction_alg) {
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto desc       = dnnl::reduction::desc(reduction_alg, input_md, output_md, 0.f, 0.f);
  return reduce_fwd_pd_t(desc, cpu_engine);
}

void DNNLReduceFwd::Execute(const Tensors& tensors) const {
  auto stream    = DNNLStream::Get();
  auto engine    = CpuEngine::Get()->get_engine();
  auto input_mem = tensors.data.GetDNNLData();
  if (tensors.out.shape().Size() == 1) {
    // scalar result
    auto out_mem = dnnl::memory(reduce_pd->dst_desc(), engine, tensors.out.data().dptr_);
    stream->RegisterPrimArgs(*reduce_fwd, {{DNNL_ARG_SRC, *input_mem}, {DNNL_ARG_DST, out_mem}});
  } else {
    auto desc    = reduce_pd->dst_desc();
    auto out_mem = tensors.out.GetDNNLData(&desc);
    stream->RegisterPrimArgs(*reduce_fwd, {{DNNL_ARG_SRC, *input_mem}, {DNNL_ARG_DST, *out_mem}});
  }
  stream->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif
