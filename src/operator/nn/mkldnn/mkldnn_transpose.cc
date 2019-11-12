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
 * \file mkldnn_transpose.cc
 * \brief Implement transpose operator via MKL-DNN reorder primitive
 * \author Tao Lv
*/

#if MXNET_USE_MKLDNN == 1

#include <mkldnn.hpp>
#include "../../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {

bool SupportMKLDNNTranspose(const TransposeParam& param,
                            const NDArray &data) {
  auto data_ndim = data.shape().ndim();

  if (data_ndim > 4 || data_ndim == 0 || data.shape().Size() == 0 ||
      data.dtype() != mshadow::kFloat32)
    return false;

  return true;
}

typedef ParamOpSign<TransposeParam> MKLDNNTransposeSignature;

class MKLDNNTransposeForward {
 public:
  std::shared_ptr<mkldnn::memory> data_;
  std::shared_ptr<mkldnn::memory> out_;
  std::shared_ptr<mkldnn::memory::desc> dst_md_;
  std::shared_ptr<mkldnn::reorder> transpose_;

 public:
  MKLDNNTransposeForward(const TransposeParam& param,
                         const NDArray &data) {
    auto shape = data.shape();
    auto data_ndim = shape.ndim();
    auto axes_ndim = param.axes.ndim();
    auto axes = mxnet::TShape(data_ndim, -1);
    if (axes_ndim == 0) {
      for (int i = 0; i < data_ndim; i++) {
        axes[i] = data_ndim - i - 1;
      }
    } else {
      axes = param.axes;
    }

    auto engine = CpuEngine::Get()->get_engine();
    auto in_mem = data.GetMKLDNNData();
    auto src_md = in_mem->get_desc();
    data_ = std::make_shared<mkldnn::memory>(src_md, engine, nullptr);

    mkldnn_dims_t strides;
    mkldnn_dims_t sh;
    dim_t total_stride = 1;
    for (int i = data_ndim - 1; i >= 0; i--) {
      sh[i] = shape[i];
      strides[axes[i]] = total_stride;
      total_stride *= shape[axes[i]];
    }

    mkldnn_memory_desc_t dst_fmt;
    mkldnn_memory_desc_init_by_strides(&dst_fmt, data_ndim, sh, mkldnn_f32, strides);

    dst_md_ = std::make_shared<mkldnn::memory::desc>(dst_fmt);
    out_ = std::make_shared<mkldnn::memory>(*dst_md_, engine, nullptr);

    transpose_ = std::make_shared<mkldnn::reorder>(*data_, *out_);
  }

  void SetNewMem(const NDArray &data, const NDArray &output) {
    if (data.IsMKLDNNData()) {
      this->data_->set_data_handle(data.GetMKLDNNData()->get_data_handle());
    } else {
      MSHADOW_TYPE_SWITCH(data.dtype(), DTYPE, {
        this->data_->set_data_handle(data.data().dptr<DTYPE>());
      });
    }

    CHECK(!output.IsMKLDNNData());
    MSHADOW_TYPE_SWITCH(output.dtype(), DTYPE, {
      this->out_->set_data_handle(output.data().dptr<DTYPE>());
    });
  }

  const mkldnn::reorder &GetFwd() const {
    return *transpose_;
  }

  void Execute() const {
    auto stream = MKLDNNStream::Get();
    mkldnn_args_map_t net_args;
    net_args.insert({{MKLDNN_ARG_FROM, *(data_)}, {MKLDNN_ARG_TO, *(out_)}});
    stream->RegisterPrimArgs(*transpose_, net_args);
    stream->Submit();
  }
};

static MKLDNNTransposeForward &GetTransposeForward(const TransposeParam& param,
                                                   const NDArray &data) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNTransposeSignature,
                                         MKLDNNTransposeForward, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNTransposeSignature,
                                            MKLDNNTransposeForward, OpHash> fwds;
#endif
  MKLDNNTransposeSignature key(param);
  key.AddSign(data);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNTransposeForward fwd(param, data);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNTransposeForward(const nnvm::NodeAttrs& attrs,
                            const OpContext &ctx,
                            const NDArray &data,
                            const OpReqType &req,
                            const NDArray &output) {
  const TransposeParam& param = nnvm::get<TransposeParam>(attrs.parsed);

  auto fwd = GetTransposeForward(param, data);
  fwd.SetNewMem(data, output);
  fwd.Execute();
}
}  // namespace op
}  // namespace mxnet
#endif

