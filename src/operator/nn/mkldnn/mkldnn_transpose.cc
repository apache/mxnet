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

// for 2D, 01-OI, 10-IO
// for 3D, 012-NCW, 021-NWC
// for 3D, 012-OIW, 210-WIO
// for 4D, 0123-NCHW, 0231-NHWC, 1230-CHWN
// for 4D, 0123-OIHW, 2310-HWIO, 1230-IHWO, 1023-IOHW
std::pair<mkldnn::memory::format, mkldnn::memory::format>
GetFormatFromAxes(const mxnet::TShape &axes) {
  auto src_fmt = mkldnn::memory::format::format_undef;
  auto dst_fmt = mkldnn::memory::format::format_undef;

  if (axes.ndim() == 2) {
    if (axes == mxnet::TShape({1, 0})) {
      src_fmt = mkldnn::memory::format::oi;
      dst_fmt = mkldnn::memory::format::io;
    }
  } else if (axes.ndim() == 3) {
    if (axes == mxnet::TShape({0, 2, 1})) {
      src_fmt = mkldnn::memory::format::ncw;
      dst_fmt = mkldnn::memory::format::nwc;
    } else if (axes == mxnet::TShape({2, 1, 0})) {
      src_fmt = mkldnn::memory::format::oiw;
      dst_fmt = mkldnn::memory::format::wio;
    } else {
      // do nothing
    }
  } else if (axes.ndim() == 4) {
    if (axes == mxnet::TShape({0, 2, 3, 1})) {
      src_fmt = mkldnn::memory::format::nchw;
      dst_fmt = mkldnn::memory::format::nhwc;
    } else if (axes == mxnet::TShape({1, 2, 3, 0})) {
      src_fmt = mkldnn::memory::format::nchw;
      dst_fmt = mkldnn::memory::format::chwn;
    } else if (axes == mxnet::TShape({2, 3, 1, 0})) {
      src_fmt = mkldnn::memory::format::oihw;
      dst_fmt = mkldnn::memory::format::hwio;
    // } else if (axes == mxnet::TShape({1, 0, 2, 3})) {
    //   src_fmt = mkldnn::memory::format::oihw;
    //   dst_fmt = mkldnn::memory::format::iohw;
    } else {
      // do nothing
    }
  }  else {
    // do nothing"
  }

  return std::make_pair(src_fmt, dst_fmt);
}


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

  auto fmt_pair = GetFormatFromAxes(axes);
  if (fmt_pair.first == mkldnn::memory::format::format_undef ||
      fmt_pair.second == mkldnn::memory::format::format_undef) {
    return false;
  }

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
    auto data_ndim = data.shape().ndim();
    auto axes_ndim = param.axes.ndim();

    auto axes = mxnet::TShape(data_ndim);
    if (axes_ndim == 0) {
      for (size_t i = 0; i < data_ndim; i++) {
        axes[i] = data_ndim - i - 1;
      }
    } else {
      axes = param.axes;
    }

    auto fmt_pair = GetFormatFromAxes(axes);

    auto engine = CpuEngine::Get()->get_engine();
    auto dims = mkldnn::memory::dims(data.shape().begin(), data.shape().end());
    auto src_md = mkldnn::memory::desc(dims, get_mkldnn_type(data.dtype()), fmt_pair.first);
    auto src_pd = mkldnn::memory::primitive_desc(src_md, engine);
    data_ = std::make_shared<mkldnn::memory>(src_pd, nullptr);

    auto dst_md = mkldnn::memory::desc(dims, get_mkldnn_type(data.dtype()), fmt_pair.second);
    dst_pd_ = std::make_shared<mkldnn::memory::primitive_desc>(dst_md, engine);
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
