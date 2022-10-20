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
 * \file dnnl_transpose.cc
 * \brief Implement transpose operator via DNNL reorder primitive
 * \author Tao Lv
 */

#if MXNET_USE_ONEDNN == 1

#include "operator/tensor/matrix_op-inl.h"
#include "dnnl_transpose-inl.h"

namespace mxnet {
namespace op {

typedef ParamOpSign<NumpyTransposeParam> DNNLTransposeSignature;

DNNLTransposeFwd::DNNLTransposeFwd(const NumpyTransposeParam& param, const NDArray& data) {
  auto shape     = data.shape();
  auto data_ndim = shape.ndim();
  auto axes_ndim = param.axes.ndim();
  auto axes      = mxnet::TShape(data_ndim, -1);
  if (!ndim_is_known(axes_ndim)) {
    for (int i = 0; i < data_ndim; i++) {
      axes[i] = data_ndim - i - 1;
    }
  } else {
    axes = param.axes;
  }

  auto engine = CpuEngine::Get()->get_engine();
  auto in_mem = data.GetDNNLData();
  auto src_md = in_mem->get_desc();
  data_       = std::make_shared<dnnl::memory>(src_md, engine, nullptr);

  dnnl_dims_t strides;
  dnnl_dims_t sh;
  dim_t total_stride = 1;
  for (int i = data_ndim - 1; i >= 0; i--) {
    sh[i]            = shape[i];
    strides[axes[i]] = total_stride;
    total_stride *= shape[axes[i]];
  }

  dnnl_memory_desc_t dst_fmt;
  dnnl_memory_desc_init_by_strides(&dst_fmt, data_ndim, sh, get_dnnl_type_t(data.dtype()), strides);

  dst_md_ = std::make_shared<dnnl::memory::desc>(dst_fmt);
  out_    = std::make_shared<dnnl::memory>(*dst_md_, engine, nullptr);

  transpose_ = std::make_shared<dnnl::reorder>(*data_, *out_);
}

void DNNLTransposeFwd::SetNewMem(const NDArray& data, const NDArray& output) {
  if (data.IsDNNLData()) {
    this->data_->set_data_handle(data.GetDNNLData()->get_data_handle());
  } else {
    MSHADOW_TYPE_SWITCH(
        data.dtype(), DTYPE, { this->data_->set_data_handle(data.data().dptr<DTYPE>()); });
  }

  CHECK(!output.IsDNNLData());
  MSHADOW_TYPE_SWITCH(
      output.dtype(), DTYPE, { this->out_->set_data_handle(output.data().dptr<DTYPE>()); });
}

const dnnl::reorder& DNNLTransposeFwd::GetFwd() const {
  return *transpose_;
}

void DNNLTransposeFwd::Execute() const {
  auto stream = DNNLStream::Get();
  dnnl_args_map_t net_args;
  net_args.insert({{DNNL_ARG_FROM, *(data_)}, {DNNL_ARG_TO, *(out_)}});
  stream->RegisterPrimArgs(*transpose_, net_args);
  stream->Submit();
}

DNNLTransposeFwd& GetTransposeForward(const NumpyTransposeParam& param, const NDArray& data) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLTransposeSignature, DNNLTransposeFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLTransposeSignature, DNNLTransposeFwd, OpHash> fwds;
#endif
  DNNLTransposeSignature key(param);
  key.AddSign(data);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    DNNLTransposeFwd fwd(param, data);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

template <>
NumpyTransposeParam ConvertTransposeParamsToNumpy<NumpyTransposeParam>(
    const NumpyTransposeParam& param) {
  NumpyTransposeParam numpy_param;
  numpy_param.axes = common::CanonicalizeAxes(param.axes);
  return numpy_param;
}

template <>
NumpyTransposeParam ConvertTransposeParamsToNumpy<TransposeParam>(const TransposeParam& param) {
  NumpyTransposeParam numpy_param;
  if (param.axes.ndim() == 0) {
    numpy_param.axes = mxnet::TShape(-1, 0);
  } else {
    numpy_param.axes = param.axes;
  }
  return numpy_param;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
