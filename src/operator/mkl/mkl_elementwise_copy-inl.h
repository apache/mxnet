/*******************************************************************************
* Copyright 2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* \file mkl_elementwise-inl.h
* \brief
* \author lingyan.guo@intel.com
*         zhenlin.luo@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKL_ELEMENTWISE_COPY_INL_H_
#define MXNET_OPERATOR_MKL_MKL_ELEMENTWISE_COPY_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "./mkl_util-inl.h"


namespace mxnet {
namespace op {

template<typename xpu, typename DType>
void MKLIdentityCompute(const nnvm::NodeAttrs& attrs,
  const OpContext& ctx,
  const std::vector<TBlob>& inputs,
  const std::vector<OpReqType>& req,
  const std::vector<TBlob>& outputs) {
  if (!req[0]) return;
#if MKL_EXPERIMENTAL == 1
  if (op::mkl_prv_data<DType>(inputs[0])) {
    std::shared_ptr<MKLMemHolder> in_data_mem = inputs[0].Mkl_mem_;
    // User copy to avoid potential problem
    std::shared_ptr<MKLData<DType> > top_data = MKLData<DType>::create();
    std::shared_ptr<MKLMemHolder> top_mem = outputs[0].Mkl_mem_;
    top_data->copy_from(in_data_mem);
    top_mem->set_prv_descriptor(top_data);
    return;
  }
#endif
  int in_blob_size = inputs[0].Size();
  int out_blob_size = outputs[0].Size();
  CHECK_EQ(in_blob_size, out_blob_size) << "MKLIdentityCompute CPU Size not Match ";
  memcpy(outputs[0].dptr_, inputs[0].dptr_, in_blob_size * sizeof(DType));
}



}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKL_ELEMENTWISE_COPY_INL_H_
