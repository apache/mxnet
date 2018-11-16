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
 * \author Wenting Jiang
*/
#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_CONCAT_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_CONCAT_INL_H_


#if MXNET_USE_MKLDNN == 1
#include <vector>
#include <utility>
#include "../concat-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

class MKLDNNConcatFwd {
  std::shared_ptr<mkldnn::concat> fwd;
  std::vector<std::shared_ptr<mkldnn::memory>> data;
  std::vector<mkldnn::primitive::at> data_mem;
  std::shared_ptr<mkldnn::memory> out;

 public:
  mkldnn::concat::primitive_desc fwd_pd;

  MKLDNNConcatFwd(
      int concat_dim,
      const std::vector<mkldnn::memory::primitive_desc> &data_md): fwd_pd(concat_dim, data_md) {
    data.resize(data_md.size());
  }

  void SetNewMem(const std::vector<const mkldnn::memory *> &in_data,
                 const mkldnn::memory &output) {
    CHECK_EQ(in_data.size(), data.size());
    for (size_t i = 0; i < data.size(); i++) {
      if (this->data[i] == nullptr) {
        this->data[i] = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
                in_data[i]->get_primitive_desc(), in_data[i]->get_data_handle()));
        this->data_mem.push_back(*this->data[i]);
      } else {
        this->data[i]->set_data_handle(in_data[i]->get_data_handle());
      }
    }
    if (this->out == nullptr)
      this->out = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              fwd_pd.dst_primitive_desc(), output.get_data_handle()));
    else
      this->out->set_data_handle(output.get_data_handle());

    if (this->fwd == nullptr)
      fwd.reset(new mkldnn::concat(fwd_pd, data_mem, *out));
  }

  const mkldnn::concat &GetFwd() const {
    return *fwd;
  }
};

static MKLDNNConcatFwd &GetConcatForward(
    int concat_dim, const std::vector<NDArray> &in_data,
    const std::vector<mkldnn::memory::primitive_desc> &data_md) {
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
    auto ins_ret = fwds.insert(std::pair<OpSignature, MKLDNNConcatFwd>(
            key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_CONCAT_INL_H_
