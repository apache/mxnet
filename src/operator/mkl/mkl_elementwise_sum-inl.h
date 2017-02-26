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
void LayerSetUp(const std::vector<mshadow::Tensor<xpu, 4, DType> > &data,
  const mshadow::Tensor<xpu, 4, DType> &out,
  size_t data_shape_size, size_t num_bottoms,
  std::vector< std::shared_ptr<MKLData<DType> > > *fwd_bottom_data_,
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

  for (size_t i = 0; i < num_bottoms; ++i) {
    fwd_bottom_data_->push_back(MKLData<DType>::create());
    CHECK_EQ(dim_src, data_shape_size);
    (*fwd_bottom_data_)[i]->create_user_layout(dim_src, sizes_src, strides_src);
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
  size_t size_ = in_data.size();
  Stream<xpu> *s = ctx.get_stream<xpu>();
  std::vector<Tensor<xpu, 4, DType> > data(size_);
  Tensor<xpu, 4, DType> out;
  if (in_data[0].ndim() == 1) {
    for (int i = 0; i < size_; ++i) {
      Shape<4> dshape = Shape4(in_data[i].shape_[0], 1, 1, 1);
      data[i] = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        in_data[i], dshape, s);
    }
    Shape<4> dshape = Shape4(out_data[0].shape_[0], 1, 1, 1);
    out = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
      out_data[0], dshape, s);
  } else if (in_data[0].ndim() == 2) {
    for (int i = 0; i < size_; ++i) {
      Shape<4> dshape = Shape4(in_data[i].shape_[0],
        in_data[i].shape_[1], 1, 1);
      data[i] = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        in_data[i], dshape, s);
    }
    Shape<4> dshape = Shape4(out_data[0].shape_[0],
      out_data[0].shape_[1], 1, 1);
    out = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
      out_data[0], dshape, s);
  } else if (in_data[0].ndim() == 3) {
    for (int i = 0; i < size_; ++i) {
      Shape<4> dshape = Shape4(in_data[i].shape_[0],
        in_data[i].shape_[1], in_data[i].shape_[2], 1);
      data[i] = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        in_data[i], dshape, s);
    }
    Shape<4> dshape = Shape4(out_data[0].shape_[0],
      out_data[0].shape_[1],
      out_data[0].shape_[2], 1);
    out = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
      out_data[0], dshape, s);
  } else {
    out = mkl_experimental_direct_get<xpu, 4, DType>(out_data[0], s);
    for (int i = 0; i < size_; ++i) {
      data[i] = mkl_experimental_direct_get<xpu, 4, DType>(in_data[i], s);
    }
  }
  std::vector< std::shared_ptr<MKLData<DType> > > fwd_bottom_data_;
  std::shared_ptr<MKLData<DType> > fwd_top_data = MKLData<DType>::create();
  std::vector<DType> coeffs_  = std::vector<DType>(data.size(), 1);;
  size_t num_bottoms = size_;
  LayerSetUp(data, out, 4, num_bottoms, &fwd_bottom_data_, fwd_top_data);

  dnnError_t e;
  std::vector<void*> bottom_data;

  int num_prv = 0;

  for (size_t i = 0; i < num_bottoms; i++) {
    void * i_data = NULL;
#if MKL_EXPERIMENTAL == 1
    i_data = reinterpret_cast<void *>(mkl_prv_data<DType>(in_data[i]));
    if (i_data != NULL) {
      bottom_data.push_back(i_data);
      num_prv += 1;
    }
#endif
    if (i_data == NULL) {
      bottom_data.push_back(reinterpret_cast<void *>(in_data[i].dptr_));
    }
  }
  dnnPrimitive_t sumPrimitive = NULL;
#if MKL_EXPERIMENTAL == 1
  if (num_prv > 0) {
    if (sumPrimitive == NULL) {
      dnnLayout_t int_layout = NULL;
      for (size_t i = 0; i < num_bottoms; ++i) {
        if (mkl_prv_data<DType>(in_data[i]) != NULL) {
          std::shared_ptr<MKLData<DType> > mem_descr =
            mkl_get_mem_desc<DType>(in_data[i].Mkl_mem_);
          fwd_bottom_data_[i] = mem_descr;
          if (int_layout == NULL) {
            int_layout = mem_descr->layout_int;
          }
        }
      }
      e = dnnSumCreate<DType>(&sumPrimitive, NULL,
        num_bottoms, int_layout, &coeffs_[0]);
      CHECK_EQ(e, E_SUCCESS);

      fwd_top_data->create_internal_layout(sumPrimitive, dnnResourceDst);

      for (size_t i = 0; i < num_bottoms; ++i) {
        if (mkl_prv_data<DType>(in_data[i]) == NULL) {
          fwd_bottom_data_[i]->create_internal_layout(sumPrimitive,
            (dnnResourceType_t)(dnnResourceMultipleSrc + i));
        }
      }
    }
  }
#endif
  if (num_prv == 0) {
    if (sumPrimitive == NULL) {
      e = dnnSumCreate<DType>(&sumPrimitive, NULL, num_bottoms,
        fwd_top_data->layout_usr, &coeffs_[0]);
      CHECK_EQ(e, E_SUCCESS);
    }
  }
  void *eltwise_res[dnnResourceNumber];
  for (size_t i = 0; i < num_bottoms; ++i) {
    if (fwd_bottom_data_[i]->conversion_needed()) {
      std::shared_ptr<MKLMemHolder> in_data_mem =
#if MKL_EXPERIMENTAL == 1
        in_data[i].Mkl_mem_;
#else
        NULL;
#endif
      eltwise_res[dnnResourceMultipleSrc + i] =
        fwd_bottom_data_[i]->get_converted_prv(data[i].dptr_, false, in_data_mem);
    } else {
      eltwise_res[dnnResourceMultipleSrc + i] =
        reinterpret_cast<void *>(bottom_data[i]);
    }
  }

  if (fwd_top_data->conversion_needed()) {
#if MKL_EXPERIMENTAL == 1
    std::shared_ptr<MKLMemHolder> top_mem = out_data[0].Mkl_mem_;
    if (top_mem->prv_data(false)) {
      fwd_top_data = mkl_get_mem_desc<DType>(top_mem);
    } else {
      top_mem->set_prv_descriptor(fwd_top_data);
    }
#endif
    eltwise_res[dnnResourceDst] =
      reinterpret_cast<void*>(fwd_top_data->prv_ptr());
  } else {
    eltwise_res[dnnResourceDst] =
      reinterpret_cast<void*>(const_cast<DType*>(out.dptr_));
  }

  e = dnnExecute<DType>(sumPrimitive, eltwise_res);
  CHECK_EQ(e, E_SUCCESS);
#if MKL_EXPERIMENTAL == 0
  if (fwd_top_data->conversion_needed()) {
    fwd_top_data->convert_from_prv(out.dptr_);
  }
#endif
}



}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKL_ELEMENTWISE_SUM_INL_H_
