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
#ifndef MXNET_OPERATOR_MKL_MKL_ELEMENTWISE_SUM_INL_H_
#define MXNET_OPERATOR_MKL_MKL_ELEMENTWISE_SUM_INL_H_

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
static void LayerSetUp(const std::vector<mshadow::Tensor<xpu, 1, DType> > &data,
  size_t data_shape_size,
  std::shared_ptr<MKLData<DType> > fwd_top_data) {
  // Whether to use an asymptotically slower (for >2 inputs) but stabler method
  // of computing the gradient for the PROD operation. (No effect for SUM op.)
  // stable_prod_grad_ = 1;
  size_t dim_src = data_shape_size;
  size_t *sizes_src = new size_t[dim_src];
  size_t *strides_src = new size_t[dim_src];
  for (size_t d = 0; d < dim_src; ++d) {
    sizes_src[d] = data[0].shape_[dim_src - d - 1];
    strides_src[d] = (d == 0) ? 1 : strides_src[d - 1] * sizes_src[d - 1];
  }

  fwd_top_data->create_user_layout(dim_src, sizes_src, strides_src);
  delete[] sizes_src;
  delete[] strides_src;
}

template<typename xpu, typename DType>
void MKLElementWiseSumCompute_(const nnvm::NodeAttrs& attrs,
  const OpContext& ctx,
  const std::vector<TBlob>& in_data,
  const std::vector<OpReqType>& req,
  const std::vector<TBlob>& out_data) {
  using namespace mshadow;
  using namespace mshadow::expr;
  if (req[0] == kNullOp) return;
  size_t size = in_data.size();
  Stream<xpu> *s = ctx.get_stream<xpu>();
  std::vector<Tensor<xpu, 1, DType> > data(size);
  Tensor<xpu, 1, DType> out = out_data[0].FlatTo1D<xpu, DType>(s);
  bool in_place_flag = false;
  int in_place_idx = 0;

  for (size_t i = 0; i < size; ++i) {
    data[i]  = in_data[i].FlatTo1D<xpu, DType>(s);
    if (data[i].dptr_ == out.dptr_) {
      in_place_idx = i;
      in_place_flag = true;
    }
  }
  std::shared_ptr<MKLData<DType> > fwd_top_data = MKLData<DType>::create();
  std::vector<DType> coeffs_  = std::vector<DType>(data.size(), 1);
  LayerSetUp(data, 1, fwd_top_data);


  dnnError_t e;
  void *eltwise_res[dnnResourceNumber];
  dnnPrimitive_t sumPrimitive = NULL;
  e = dnnSumCreate<DType>(&sumPrimitive, NULL, size, fwd_top_data->layout_usr,
    &coeffs_[0]);
  CHECK_EQ(e, E_SUCCESS);

  eltwise_res[dnnResourceDst] = reinterpret_cast<void*>(const_cast<DType*>(out.dptr_));
  eltwise_res[dnnResourceMultipleSrc] =
    reinterpret_cast<void *>(reinterpret_cast<void *>(in_data[in_place_idx].dptr_));
  for (size_t i = 1; i < size; ++i) {
    if (i == in_place_idx) continue;
    eltwise_res[dnnResourceMultipleSrc + i] =
      reinterpret_cast<void *>(reinterpret_cast<void *>(in_data[i].dptr_));
  }

  e = dnnExecute<DType>(sumPrimitive, eltwise_res);
  CHECK_EQ(e, E_SUCCESS);

  if (sumPrimitive != NULL) {
    dnnDelete<DType>(sumPrimitive);
    sumPrimitive = NULL;
  }
}



}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKL_ELEMENTWISE_SUM_INL_H_
