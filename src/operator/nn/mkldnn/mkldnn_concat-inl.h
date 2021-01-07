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
 * \file mkldnn_concat-inl.h
 * \brief
 * \author
*/
#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_CONCAT_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_CONCAT_INL_H_


#if MXNET_USE_ONEDNN == 1
#include <vector>
#include <utility>
#include "../concat-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

class MKLDNNConcatFwd {
 public:
  mkldnn::concat::primitive_desc fwd_pd;

  MKLDNNConcatFwd(int concat_dim, const std::vector<mkldnn::memory::desc> &data_md)
      : fwd_pd(concat_dim, data_md, CpuEngine::Get()->get_engine()) {
    // "MKL-DNN introduced padded formats since 0.15 which require more memory
    // for computation compared with the actual tensor size. Currently, MKL-DNN
    // operators are still reusing those memory from memory planning and the
    // memory size may smaller than what MKL-DNN kernels require. So here we need
    // select suboptimal kernel for computation according to tensor sizes."

    // When fwd_pd uses a padded format, impose a plain one instead
    const auto dst_md = fwd_pd.dst_desc();
    if (dst_md.data.format_kind == mkldnn_blocked &&
        dst_md.data.format_desc.blocking.inner_nblks > 0) {
      auto plain_dst_tag = static_cast<mkldnn::memory::format_tag>(
          GetDefaultFormat(dst_md.data.ndims));
      auto plain_dst_md = mkldnn::memory::desc(
          dst_md.dims(), dst_md.data_type(), plain_dst_tag);
      fwd_pd = mkldnn::concat::primitive_desc(plain_dst_md, concat_dim, data_md,
                                              CpuEngine::Get()->get_engine());
    }
    fwd_ = std::make_shared<mkldnn::concat>(fwd_pd);
  }

  const mkldnn::concat &GetFwd() const;

 private:
  std::shared_ptr<mkldnn::concat> fwd_;
};

static MKLDNNConcatFwd &GetConcatForward(
    int concat_dim, const std::vector<NDArray> &in_data,
    const std::vector<mkldnn::memory::desc> &data_md) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<OpSignature, MKLDNNConcatFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<OpSignature, MKLDNNConcatFwd, OpHash> fwds;
#endif
  OpSignature key;
  key.AddSign(concat_dim);
  key.AddSign(in_data);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNConcatFwd fwd(concat_dim, data_md);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_CONCAT_INL_H_
