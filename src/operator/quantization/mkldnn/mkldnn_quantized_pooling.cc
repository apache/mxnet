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
 * \file mkldnn_quantized_pooling.cc
 * \brief
 * \author Tao Lv
*/

#if MXNET_USE_MKLDNN == 1

#include "./mkldnn_quantized_pooling-inl.h"

namespace mxnet {
namespace op {

void MKLDNNQuantizedPoolingFwd::Init(const mxnet::NDArray &input, const mxnet::NDArray &output,
                            const int kernel_h,  const int kernel_w,
                            const int stride_h,  const int stride_w,
                            const int padding_t, const int padding_b,
                            const int padding_l, const int padding_r) {
  // mkldnn::memory::desc
  auto src_md = input.GetMKLDNNData()->get_primitive_desc().desc();
  mkldnn::memory::dims dims = {src_md.data.dims[0],
                               src_md.data.dims[1],
                               static_cast<int>(output.shape()[2]),
                               static_cast<int>(output.shape()[3])};
  auto dst_md = mkldnn::memory::desc({dims},
                                     static_cast<mkldnn::memory::data_type>(src_md.data.data_type),
                                     static_cast<mkldnn::memory::format>(src_md.data.format));
  const mkldnn::engine engine = CpuEngine::Get()->get_engine();
  const mkldnn::algorithm alg_kind = this->alg_kind_;
  if (alg_kind != mkldnn::algorithm::pooling_max &&
      alg_kind != mkldnn::algorithm::pooling_avg &&
      alg_kind != mkldnn::algorithm::pooling_avg_include_padding &&
      alg_kind != mkldnn::algorithm::pooling_avg_exclude_padding) {
    LOG(FATAL) << "MKLDNN Pooling: algorithm is not supported";
  }

  mkldnn::prop_kind prop = mkldnn::prop_kind::forward_scoring;

  const mkldnn::memory::dims strides = {stride_h,  stride_w  };
  const mkldnn::memory::dims pad_l   = {padding_t, padding_l };
  const mkldnn::memory::dims pad_r   = {padding_b, padding_r };
  const mkldnn::memory::dims kernel  = {kernel_h,  kernel_w  };
  // mkldnn::pooling_forward::desc
  const auto fwd_desc = mkldnn::pooling_forward::desc(prop, alg_kind, src_md, dst_md,
                                                      strides, kernel, pad_l, pad_r,
                                                      mkldnn::padding_kind::zero);
  this->fwd_pd_.reset(new mkldnn::pooling_forward::primitive_desc(fwd_desc, engine));
  this->data_.reset(new mkldnn::memory(input.GetMKLDNNData()->get_primitive_desc()));
  this->out_.reset(new mkldnn::memory(this->fwd_pd_->dst_primitive_desc()));
  this->fwd_.reset(new mkldnn::pooling_forward(*(this->fwd_pd_),
                                                 mkldnn::primitive::at(*(this->data_)),
                                                 *(this->out_)));
  return;
}

void MKLDNNQuantizedPoolingFwd::SetDataHandle(const mxnet::NDArray &data,
                                     const mxnet::NDArray &output) {
  // mkldnn::memory
  auto data_mem = data.GetMKLDNNData();
  auto out_mem = const_cast<NDArray&>(output).CreateMKLDNNData(
                                                  this->fwd_pd_->dst_primitive_desc());
  this->data_->set_data_handle(data_mem->get_data_handle());
  this->out_->set_data_handle(out_mem->get_data_handle());
}

void MKLDNNQuantizedPoolingFwd::Execute() {
  if (this->fwd_) {
    MKLDNNStream::Get()->RegisterPrim(*(this->fwd_));
    MKLDNNStream::Get()->Submit();
  } else {
    LOG(FATAL) << "MKLDNN Pooling: forward primitive is nullptr";
  }
}

mkldnn::algorithm GetMKLDNNQuantizedPoolAlgo(const PoolingParam &param) {
  switch (param.pool_type) {
    case pool_enum::kMaxPooling:
      return mkldnn::algorithm::pooling_max;
      break;
    case pool_enum::kAvgPooling:
      return mkldnn::algorithm::pooling_avg_include_padding;
      break;
    default:
      LOG(FATAL) << "MKLDNN Pooling: Unknown pooling method.";
      return mkldnn::algorithm::pooling_max;
  }
}

mkldnn::pooling_forward::primitive_desc GetQuantizedPoolingFwd(const PoolingParam &param,
                                                               const memory::desc &data_md,
                                                               const memory::desc &out_md) {
  CHECK_EQ(param.kernel.ndim(), 2) << "Not Implemented";
  int kernel_h_, kernel_w_;
  if (param.global_pool) {
    kernel_h_ = data_md.data.dims[2];
    kernel_w_ = data_md.data.dims[3];
  } else {
    kernel_h_ = param.kernel[0];
    kernel_w_ = param.kernel[1];
  }

  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";

  int pad_t_ = param.pad[0], pad_b_ = param.pad[0];
  int pad_l_ = param.pad[1], pad_r_ = param.pad[1];
  int stride_h_ = param.stride[0], stride_w_ = param.stride[1];

  const mkldnn::engine engine = CpuEngine::Get()->get_engine();
  if (param.global_pool) {
    pad_t_ = pad_b_ = pad_l_ = pad_r_ = 0;
    stride_h_ = stride_w_ = 1;
  }
  if (pad_t_ != 0 || pad_l_ != 0) {
    CHECK(param.pool_type == pool_enum::kAvgPooling ||
          param.pool_type == pool_enum::kMaxPooling)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_l_, kernel_w_);
    CHECK_LT(pad_t_, kernel_h_);
  }


  const mkldnn::algorithm alg = GetMKLDNNQuantizedPoolAlgo(param);
  mkldnn::prop_kind kind = mkldnn::prop_kind::forward_scoring;

  const pooling_forward::desc poolingFwd_desc(kind, alg, data_md, out_md,
                                              {static_cast<int>(stride_h_),
                                               static_cast<int>(stride_w_)},
                                              {kernel_h_, kernel_w_},
                                              {static_cast<int>(pad_t_),
                                               static_cast<int>(pad_l_)},
                                              {static_cast<int>(pad_b_),
                                               static_cast<int>(pad_r_)},
                                              padding_kind::zero);
  return mkldnn::pooling_forward::primitive_desc(poolingFwd_desc, engine);
}

MKLDNNQuantizedPoolingFwd &GetQuantizedPoolingFwd(const PoolingParam &param,
                                                  const NDArray &data,
                                                  const NDArray &output) {
  static thread_local std::unordered_map<MKLDNNPoolingSignature,
                                         MKLDNNQuantizedPoolingFwd,
                                         OpHash> pooling_fwds;

  MKLDNNPoolingSignature key(param);
  key.AddSign(data);
  key.AddSign(output);

  auto it = pooling_fwds.find(key);
  if (it == pooling_fwds.end()) {
    CHECK_EQ(param.kernel.ndim(), 2) << "Not Implemented";
    auto data_md = data.GetMKLDNNData()->get_primitive_desc().desc();
    int kernel_h_, kernel_w_;
    if (param.global_pool) {
      kernel_h_ = data_md.data.dims[2];
      kernel_w_ = data_md.data.dims[3];
    } else {
      kernel_h_ = param.kernel[0];
      kernel_w_ = param.kernel[1];
    }

    CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
    CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";

    int pad_t_ = param.pad[0], pad_b_ = param.pad[0];
    int pad_l_ = param.pad[1], pad_r_ = param.pad[1];
    int stride_h_ = param.stride[0], stride_w_ = param.stride[1];

    if (param.global_pool) {
        pad_t_ = pad_b_ = pad_l_ = pad_r_ = 0;
        stride_h_ = stride_w_ = 1;
    }

    if (pad_t_ != 0 || pad_l_ != 0) {
        CHECK(param.pool_type == pool_enum::kAvgPooling ||
              param.pool_type == pool_enum::kMaxPooling)
              << "Padding implemented only for average and max pooling.";
        CHECK_LT(pad_l_, kernel_w_);
        CHECK_LT(pad_t_, kernel_h_);
    }

    const mkldnn::algorithm alg = GetMKLDNNQuantizedPoolAlgo(param);
    MKLDNNQuantizedPoolingFwd fwd(data, output, kernel_h_, kernel_w_, stride_h_, stride_w_,
                         pad_t_, pad_b_, pad_l_, pad_r_, alg);
    auto ins_ret = pooling_fwds.insert(
        std::pair<MKLDNNPoolingSignature, MKLDNNQuantizedPoolingFwd>(key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

void MKLDNNQuantizedPoolingForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                                   const std::vector<NDArray> &in_data,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<NDArray> &out_data) {
  if (in_data[0].dtype() == mshadow::kUint8 || in_data[0].dtype() == mshadow::kInt8) {
    const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
    auto fwd = GetQuantizedPoolingFwd(param, in_data[0], out_data[0]);
    fwd.SetDataHandle(in_data[0], out_data[0]);
    fwd.Execute();
    out_data[1].data().dptr<float>()[0] = in_data[1].data().dptr<float>()[0];
    out_data[2].data().dptr<float>()[0] = in_data[2].data().dptr<float>()[0];
  } else {
    LOG(FATAL) << "mkldnn_quantized_pooling op only supports uint8 and int8 as input type";
  }
}

NNVM_REGISTER_OP(_contrib_quantized_pooling)
.set_attr<FComputeEx>("FComputeEx<cpu>", MKLDNNQuantizedPoolingForward);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
