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
 * \brief
 * \author
*/

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <vector>
#include "../../operator_common.h"
#include "../../tensor/matrix_op-inl.h"
#include "./mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1

#include <mkldnn.hpp>

namespace mxnet {
namespace op {

bool SupportMKLDNNTranspose(const TransposeParam& param,
                            const NDArray &data) {
  auto data_ndim = data.shape().ndim();
  auto axes_ndim = param.axes.ndim();

  // currently, we dont support transposion for any internal format
  if (data.IsMKLDNNData()) return false;

  auto axes = mxnet::TShape(data_ndim);
  if (axes_ndim == 0) {
    for (size_t i = 0; i < data_ndim; i++) {
      axes[i] = data_ndim - i - 1;
    }
  } else {
    axes = param.axes;
  }

  CHECK_EQ(axes.ndim(), data_ndim);
  return true;
}


typedef ParamOpSign<TransposeParam> MKLDNNTransposeSignature;

class MKLDNNTransposeForward {
  std::shared_ptr<mkldnn::memory> data_;
  std::shared_ptr<mkldnn::memory> out_;
  std::shared_ptr<mkldnn::memory::primitive_desc> dst_pd_;
  std::shared_ptr<mkldnn::reorder> transpose_;

 public:
  MKLDNNTransposeForward(const TransposeParam& param,
                         const OpReqType &req,
                         const NDArray &data) {
    auto shape = data.shape();
    auto data_ndim = shape.ndim();
    auto axes_ndim = param.axes.ndim();
    auto axes = mxnet::TShape(data_ndim);
    if (axes_ndim == 0) {
      for (size_t i = 0; i < data_ndim; i++) {
        axes[i] = data_ndim - i - 1;
      }
    } else {
      axes = param.axes;
    }

    auto engine = CpuEngine::Get()->get_engine();
    auto in_mem = data.GetMKLDNNData();
    auto src_pd = in_mem->get_primitive_desc();
    data_ = std::make_shared<mkldnn::memory>(src_pd, nullptr);

    // destination
    mkldnn_memory_desc_t dst_fmt;
    dst_fmt.primitive_kind = mkldnn_memory;
    dst_fmt.ndims = data_ndim;
    dst_fmt.data_type = mkldnn_f32;
    dst_fmt.format = mkldnn_blocked;

    for (size_t i = 0; i < data_ndim; i++)
      dst_fmt.dims[i] = shape[i];

    unsigned int total_stride = 1;
    for (int i = data_ndim - 1; i >= 0; i--) {
      dst_fmt.layout_desc.blocking.padding_dims[i] = shape[i];
      dst_fmt.layout_desc.blocking.block_dims[i] = 1;
      dst_fmt.layout_desc.blocking.offset_padding_to_data[i]= 0;
      dst_fmt.layout_desc.blocking.strides[0][axes[i]] = total_stride;
      dst_fmt.layout_desc.blocking.strides[1][axes[i]] = 1;

      total_stride *= shape[axes[i]];
    }

    dst_fmt.layout_desc.blocking.offset_padding = 0;
    dst_pd_ = std::make_shared<mkldnn::memory::primitive_desc>(dst_fmt, engine);
    out_ = std::make_shared<mkldnn::memory>(*dst_pd_, nullptr);

    transpose_ = std::make_shared<mkldnn::reorder>(*data_, *out_);
  }

  void SetNewMem(const NDArray &data, const NDArray &output) {
    MSHADOW_TYPE_SWITCH(data.dtype(), DTYPE, {
      this->data_->set_data_handle(data.data().dptr<DTYPE>());
      this->out_->set_data_handle(output.data().dptr<DTYPE>());
    });
  }

  const mkldnn::reorder &GetFwd() const {
    return *transpose_;
  }
};

static MKLDNNTransposeForward &GetTransposeForward(const TransposeParam& param,
                                                   const OpReqType &req,
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
    MKLDNNTransposeForward fwd(param, req, data);
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

  CHECK_EQ(req, kWriteTo) << "Transpose does not support inplace";

  auto *stream = MKLDNNStream::Get();
  auto fwd = GetTransposeForward(param, req, data);

  fwd.SetNewMem(data, output);
  stream->RegisterPrim(fwd.GetFwd());
  stream->Submit();
}
}  // namespace op
}  // namespace mxnet

#endif
