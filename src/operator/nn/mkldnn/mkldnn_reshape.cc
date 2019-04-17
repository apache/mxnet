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
 * \file mkldnn_reshape.cc
 * \brief Implement reshape operator via MKL-DNN reorder primitive
 * \author Tao Lv
*/

#if MXNET_USE_MKLDNN == 1

#include <mkldnn.hpp>
#include "../../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {

bool SupportMKLDNNReshape(const ReshapeParam& param,
                          const NDArray &data) {
  auto data_ndim = data.shape().ndim();

  if (data_ndim > 4 ||
      data.dtype() != mshadow::kFloat32 ||
      param.shape.ndim() > 4)
    return false;

  return true;
}

typedef ParamOpSign<ReshapeParam> MKLDNNReshapeSignature;

class MKLDNNReshapeForward {
  std::shared_ptr<mkldnn::memory> data_;
  std::shared_ptr<mkldnn::memory> out_;
  std::shared_ptr<mkldnn::memory> temp_;
  std::vector<mkldnn::primitive> prims_;

  bool needInvalidateInput = false;

 public:
  MKLDNNReshapeForward(const ReshapeParam &param,
                       const OpReqType &req,
                       const NDArray &input,
                       const NDArray &output) {
    CHECK_NE(req, kNullOp);
    auto engine = CpuEngine::Get()->get_engine();

    // data_
    auto in_mem = input.GetMKLDNNData();
    auto in_pd = in_mem->get_primitive_desc();
    data_ = std::make_shared<mkldnn::memory>(in_pd, nullptr);

    // temp_
    auto temp_dims = mkldnn::memory::dims(input.shape().begin(), input.shape().end());
    auto temp_type = static_cast<mkldnn::memory::data_type>(in_pd.desc().data.data_type);
    auto temp_fmt = static_cast<mkldnn::memory::format>(GetDefaultFormat(in_pd.desc()));
    auto temp_desc = mkldnn::memory::desc(temp_dims, temp_type, temp_fmt);
    auto temp_pd = mkldnn::memory::primitive_desc(temp_desc, engine);
    temp_ = std::make_shared<mkldnn::memory>(temp_pd); // allocate memory?

    // destination
    out_ = std::make_shared<mkldnn::memory>(temp_pd, nullptr);

    if (req == kWriteInplace) {
      if (input.IsMKLDNNData()) {
        // reorder to default
        prims_.push_back(mkldnn::reorder(*data_, *temp_));
        prims_.push_back(mkldnn::reorder(*temp_, *out_));
        needInvalidateInput = true;
      }
    } else if (req == kWriteTo) {
      if (input.IsMKLDNNData()) {
        // reorder to default
        prims_.push_back(mkldnn::reorder(*data_, *temp_));
        prims_.push_back(mkldnn::reorder(*temp_, *out_));
        needInvalidateInput = false; 
      } else {
        prims_.push_back(mkldnn::reorder(*data_, *out_));
        needInvalidateInput = false;
      }
    } else if (req == kAddTo) {
      LOG(FATAL) << "kAddTo is not supported yet";
    } else {
      LOG(FATAL) << "not supported req type: " << req;
    }
  }

  void SetNewMem(const NDArray &input, const NDArray &output) {
    if (input.IsMKLDNNData()) {
      this->data_->set_data_handle(input.GetMKLDNNData()->get_data_handle());
    } else {
      MSHADOW_TYPE_SWITCH(input.dtype(), DTYPE, {
        this->data_->set_data_handle(input.data().dptr<DTYPE>());
      })
    }

    if (output.IsMKLDNNData()) {
      this->out_->set_data_handle(output.GetMKLDNNData()->get_data_handle());
    } else {
      MSHADOW_TYPE_SWITCH(output.dtype(), DTYPE, {
        this->out_->set_data_handle(output.data().dptr<DTYPE>());
      })
    }
  }

  void Execute(const NDArray &input,
               const NDArray &output) {
    // set memory handles
    SetNewMem(input, output);
    // register primitives
    auto stream = MKLDNNStream::Get();
    for (auto &v : this->prims_) {
      stream->RegisterPrim(v);
    }
    stream->Submit();
    // invalidate mkldnn memory in input
    if (needInvalidateInput) {
      const_cast<NDArray &>(input).InvalidateMKLDNNData();
    }
  }
};

static MKLDNNReshapeForward &GetReshapeForward(const ReshapeParam& param,
                                               const OpReqType &req,
                                               const NDArray &input,
                                               const NDArray &output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNReshapeSignature,
                                         MKLDNNReshapeForward, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNReshapeSignature,
                                            MKLDNNReshapeForward, OpHash> fwds;
#endif
  MKLDNNReshapeSignature key(param);
  key.AddSign(req);
  key.AddSign(input);
  key.AddSign(output);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNReshapeForward fwd(param, req, input, output);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNReshapeForward(const nnvm::NodeAttrs& attrs,
                          const OpContext &ctx,
                          const NDArray &input,
                          const OpReqType &req,
                          const NDArray &output) {
  const ReshapeParam& param = nnvm::get<ReshapeParam>(attrs.parsed);
  if (req == kNullOp) return;

  // if (req == kWriteInplace) printf("reshape req kWriteInplace \n");
  // else if (req == kWriteTo) printf("reshape req kWriteTo \n");
  // else printf("reshape req %d \n", req);

  // if (input.IsMKLDNNData()) {
  //   printf("input is internal, format %d \n", input.GetMKLDNNData()->get_primitive_desc().desc().data.format);
  // }

  auto fwd = GetReshapeForward(param, req, input, output);
  fwd.Execute(input, output);
}
}  // namespace op
}  // namespace mxnet
#endif
