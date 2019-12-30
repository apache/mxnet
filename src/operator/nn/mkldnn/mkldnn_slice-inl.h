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
 * \file mkldnn_slice-inl.h
 * \brief
 * \author Zhiyuan Huang
*/

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_SLICE_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_SLICE_INL_H_

#if MXNET_USE_MKLDNN == 1

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <utility>
#include "../../operator_common.h"
#include "../../tensor/slice-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

class MKLDNNSliceFwd {
 public:
  MKLDNNSliceFwd(const SliceParam &param,
                 const NDArray &in,
                 const NDArray &out);
  void SetNewMem(const mkldnn::memory &input, const mkldnn::memory &output);
  void Register();

 private:
  std::shared_ptr<mkldnn::memory> data_;
  std::shared_ptr<mkldnn::memory> out_;
  std::shared_ptr<mkldnn::reorder> fwd_;
};

typedef ParamOpSign<SliceParam> MKLDNNSliceSignature;
MKLDNNSliceFwd &GetSliceForward(const SliceParam &param, const bool is_train,
                 const NDArray &in_data, const NDArray &out_data);

void MKLDNNSlice(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                 const NDArray &in, OpReqType req, const NDArray &out);

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_SLICE_INL_H_
