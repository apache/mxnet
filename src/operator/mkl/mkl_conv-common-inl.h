/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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
* \file mkl_convolution-inl.h
* \brief
* \author lingyan.guo@intel.com
*         zhenlin.luo@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKL_CONV_COMMON_INL_H_
#define MXNET_OPERATOR_MKL_MKL_CONV_COMMON_INL_H_

#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/storage.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "mkl_util-inl.h"


namespace mxnet {
namespace op {

template <typename xpu, typename DType>
class MKLConvCommon {
 public:
  MKLConvCommon(): width_(0), height_(0), width_out_(0),
    height_out_(0), kernel_w_(0), kernel_h_(0),
    stride_w_(0), stride_h_(0), pad_w_(0), pad_h_(0)  {}
  virtual ~MKLConvCommon() {}

  void AddToModeAllocAndStoreBuffer(void *src, int blob_size, Storage::Handle *pws) {
    int blob_byte_size = blob_size * sizeof(DType);
    *pws = Storage::Get()->Alloc(blob_byte_size, Context::CPU());
    memcpy(pws->dptr, src, blob_byte_size);
  }
  void AddToModeAddAndReleaseBuffer(Storage::Handle *pws, void *dst_, int blob_size) {
    DType *dst = reinterpret_cast<DType*>(dst_);
    DType *src = reinterpret_cast<DType*>(pws->dptr);
    for (int i = 0; i < blob_size; i++) {
      dst[i] += src[i];
    }
    if (pws->dptr)
      Storage::Get()->Free(*pws);
    pws->dptr = NULL;
  }

 protected:
  int width_,
    height_,
    width_out_,
    height_out_,
    kernel_w_,
    kernel_h_,
    stride_w_,
    stride_h_;
  int group_,
    num_,
    channel_output_;
  size_t channels_;
  int pad_w_,
    pad_h_;
};

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKL_CONV_COMMON_INL_H_
