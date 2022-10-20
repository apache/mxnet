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
 * \file dnnl_reshape.cc
 * \brief Implement reshape operator via DNNL reorder primitive
 * \author Tao Lv
 */

#if MXNET_USE_ONEDNN == 1
#include "operator/tensor/elemwise_unary_op.h"
#include "dnnl_base-inl.h"
#include "dnnl_reshape-inl.h"

namespace mxnet {
namespace op {

// Support for https://oneapi-src.github.io/oneDNN/v2.6/dev_guide_reorder.html
bool SupportDNNLReshape(const NDArray& input) {
  return SupportDNNL(input) && input.shape().Size() != 1;
}

DNNLReshapeFwd::DNNLReshapeFwd(const OpReqType& req, const NDArray& input, const NDArray& output) {
  const auto engine = CpuEngine::Get()->get_engine();
  auto in_mem       = input.GetDNNLData();

  // Create temp memory
  auto temp_dims = dnnl::memory::dims(input.shape().begin(), input.shape().end());
  auto temp_type = static_cast<dnnl::memory::data_type>(get_dnnl_type(input.dtype()));
  auto temp_fmt  = static_cast<dnnl::memory::format_tag>(GetDefaultFormat(input.shape().ndim()));
  auto temp_desc = dnnl::memory::desc(temp_dims, temp_type, temp_fmt);

  out_ = std::make_shared<dnnl::memory>(temp_desc, engine, nullptr);
  if (req == kWriteInplace) {
    // If the input has DNNL internal layout, we need reorder it to a temporal buffer with
    // default layout and copy from the temporal buffer back to output buffer which has the same
    // address with input buffer.
    // If the input has default layout, then nothing need to do.
    if (input.IsDNNLData()) {
      temp_ = std::make_shared<dnnl::memory>(temp_desc, engine, nullptr);
      prims_.push_back(dnnl::reorder(*in_mem, *temp_));  // reorder to default
      prims_.push_back(dnnl::reorder(*temp_, *out_));    // copy back
    }
  } else if (req == kWriteTo) {
    prims_.push_back(dnnl::reorder(*in_mem, *out_));
  } else {
    LOG(FATAL) << "not supported req type: " << req;
  }
}

int DNNLReshapeFwd::GetWorkspaceSize() {
  return temp_ ? temp_->get_desc().get_size() : 0;
}

void DNNLReshapeFwd::Execute(const NDArray& input,
                             const NDArray& output,
                             const OpReqType& req,
                             void* workspace) {
  auto stream = DNNLStream::Get();
  auto in_mem = input.GetDNNLData();
  // register primitives and arguments
  std::vector<dnnl_args_map_t> args_map;
  size_t prims_size = prims_.size();
  if (prims_size == 1) {
    args_map.push_back({{DNNL_ARG_FROM, *in_mem}, {DNNL_ARG_TO, *output.GetDNNLData()}});
  } else if (prims_size == 2) {
    if (workspace) {
      temp_->set_data_handle(workspace);
    }
    args_map.push_back({{DNNL_ARG_FROM, *in_mem}, {DNNL_ARG_TO, *temp_}});
    args_map.push_back({{DNNL_ARG_FROM, *temp_}, {DNNL_ARG_TO, *output.GetDNNLData()}});
  } else {
    CHECK(prims_size == 0 && req != kWriteTo) << "kWriteTo should never reach here.";
  }

  for (size_t i = 0; i < prims_size; i++) {
    stream->RegisterPrimArgs(prims_[i], args_map[i]);
  }
  stream->Submit();
  // invalidate dnnl memory in output
  const_cast<NDArray&>(output).InvalidateDNNLData();
}

DNNLReshapeFwd& GetReshapeForward(const OpReqType& req,
                                  const NDArray& input,
                                  const NDArray& output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLReshapeSignature, DNNLReshapeFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLReshapeSignature, DNNLReshapeFwd, OpHash> fwds;
#endif
  DNNLReshapeSignature key;
  key.AddSign(req);
  key.AddSign(input);
  key.AddSign(output);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    DNNLReshapeFwd fwd(req, input, output);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void DNNLReshapeForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const NDArray& input,
                        const OpReqType& req,
                        const NDArray& output) {
  if (req == kNullOp)
    return;
  CHECK_NE(req, kAddTo) << "kAddTo is not supported yet";
  auto fwd     = GetReshapeForward(req, input, output);
  auto ws_size = fwd.GetWorkspaceSize();
  void* ws_ptr = nullptr;
  if (ws_size) {
    mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
    mshadow::Tensor<cpu, 1, char> ws =
        ctx.requested[0].get_space_typed<cpu, 1, char>(mshadow::Shape1(ws_size), s);
    ws_ptr = static_cast<void*>(ws.dptr_);
  }
  fwd.Execute(input, output, req, ws_ptr);
}

}  // namespace op
}  // namespace mxnet
#endif
