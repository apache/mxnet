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
* \file mkl_relu-inl.h
* \brief
* \author zhenlin.luo@intel.com
*         lingyan.guo@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKL_RELU_INL_H_
#define MXNET_OPERATOR_MKL_MKL_RELU_INL_H_


#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "./mkl_util-inl.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class MKLReluOp : public Operator {
 public:
  static std::string getName() {
    return "MKLReluOp";
  }
  MKLReluOp():
      reluFwd_(NULL),
      reluBwd_(NULL) {
    init_mkldnn_ = false;
    fwd_top_data_ = MKLData<DType>::create();
    fwd_bottom_data_ = MKLData<DType>::create();
    bwd_top_diff_ = MKLData<DType>::create();
    bwd_bottom_diff_ = MKLData<DType>::create();
  }

  ~MKLReluOp() {
    if (reluFwd_ != NULL) {
      dnnDelete<DType>(reluFwd_);
      reluFwd_ = NULL;
    }
    if (reluBwd_ != NULL) {
      dnnDelete<DType>(reluBwd_);
      reluBwd_ = NULL;
    }
  }

 private:
  void LayerSetUp(const mshadow::Tensor<xpu, 4, DType> &data,
                  const mshadow::Tensor<xpu, 4, DType> &out) {
    size_t dim = 4;
    size_t *sizes = new size_t[dim];
    size_t *strides = new size_t[dim];
    for (size_t d = 0; d < dim; ++d) {
      (sizes)[d] = data.shape_[dim - 1 - d];
      (strides)[d] = (d == 0) ? 1 : (strides)[d - 1] * (sizes)[d - 1];
    }
    // Names are for debugging only
    fwd_bottom_data_->name = "fwd_bottom_data   @ " + getName();
    fwd_top_data_->name = "fwd_top_data      @ " + getName();
    bwd_bottom_diff_->name = "bwd_bottom_diff   @ " + getName();
    bwd_top_diff_->name = "bwd_top_diff      @ " + getName();
    fwd_bottom_data_->create_user_layout(dim, (sizes), (strides));
    fwd_top_data_->create_user_layout(dim, (sizes), (strides));
    bwd_bottom_diff_->create_user_layout(dim, (sizes), (strides));
    bwd_top_diff_->create_user_layout(dim, (sizes), (strides));
    delete[] sizes;
    delete[] strides;
  }

 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data;
    Tensor<xpu, 4, DType> out;
    if (in_data[activation::kData].ndim() == 1) {
      Shape<4> dshape = Shape4(in_data[activation::kData].shape_[0], 1, 1, 1);
      data = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        in_data[activation::kData], dshape, s);
      out = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        out_data[activation::kOut], dshape, s);
    } else if (in_data[activation::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[activation::kData].shape_[0],
      in_data[activation::kData].shape_[1], 1, 1);
      data = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        in_data[activation::kData], dshape, s);
      out = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        out_data[activation::kOut], dshape, s);
    } else if (in_data[activation::kData].ndim() == 3) {
      Shape<4> dshape = Shape4(in_data[activation::kData].shape_[0],
        in_data[activation::kData].shape_[1],
        in_data[activation::kData].shape_[2], 1);
      data = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        in_data[activation::kData], dshape, s);
      out = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        out_data[activation::kOut], dshape, s);
    } else {
      data = mkl_experimental_direct_get<xpu, 4, DType>(in_data[activation::kData], s);
      out = mkl_experimental_direct_get<xpu, 4, DType>(out_data[activation::kOut], s);
    }
    if (!init_mkldnn_) {
      LayerSetUp(data, out);
      init_mkldnn_ = true;
    }
    void* bottom_data = NULL;
#if MKL_EXPERIMENTAL == 1
    bottom_data =
          reinterpret_cast<void *>(mkl_prv_data<DType>(in_data[activation::kData]));
#endif
#if MKL_EXPERIMENTAL == 1
    if (bottom_data != NULL) {
      if (reluFwd_ == NULL) {
      std::shared_ptr<MKLData<DType> > mem_descr =
        mkl_get_mem_desc<DType>(in_data[activation::kData].Mkl_mem_);
      DType negative_slope = 0;
      dnnError_t e;
      e = dnnReLUCreateForward<DType>(&reluFwd_, NULL, mem_descr->layout_int,
                                      negative_slope);
      CHECK_EQ(e, E_SUCCESS);
      e = dnnReLUCreateBackward<DType>(&reluBwd_, NULL, mem_descr->layout_int,
                                       mem_descr->layout_int, negative_slope);
      CHECK_EQ(e, E_SUCCESS);

      fwd_bottom_data_ = mem_descr;
      fwd_top_data_->create_internal_layout(reluFwd_, dnnResourceDst);
      bwd_top_diff_->create_internal_layout(reluFwd_, dnnResourceDst);
      bwd_bottom_diff_->create_internal_layout(reluFwd_, dnnResourceSrc);
      }
    }
#endif
    if (bottom_data  == NULL) {
      bottom_data = data.dptr_;
      if (reluFwd_ == NULL) {
        dnnError_t e;
        DType negative_slope = 0;
        e = dnnReLUCreateForward<DType>(&reluFwd_, NULL,
                                        fwd_bottom_data_->layout_usr, negative_slope);
        CHECK_EQ(e, E_SUCCESS);
        e = dnnReLUCreateBackward<DType>(&reluBwd_, NULL,
                                         fwd_bottom_data_->layout_usr, fwd_bottom_data_->layout_usr,
                                         negative_slope);
        CHECK_EQ(e, E_SUCCESS);
      }
    }
    dnnError_t e;
    void* relu_res[dnnResourceNumber];
    relu_res[dnnResourceSrc] = bottom_data;

    relu_res[dnnResourceDst] = fwd_top_data_->get_output_ptr(
      out.dptr_, fwd_top_data_, out_data[activation::kOut], (data.dptr_ == out.dptr_));
    e = dnnExecute<DType>(reluFwd_, relu_res);
    CHECK_EQ(e, E_SUCCESS);
#if MKL_EXPERIMENTAL == 0
    if (fwd_top_data_->conversion_needed()) {
      fwd_top_data_->convert_from_prv(out.dptr_);
    }
#endif
  }
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    if (!req[0]) {
      return;
    }
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == 1 && in_grad.size() == 1);
    CHECK_EQ(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> m_out_grad;
    Tensor<xpu, 4, DType> m_out_data;
    Tensor<xpu, 4, DType> m_in_grad;

    if (out_grad[activation::kOut].ndim() == 1) {
      Shape<4> dshape = Shape4(out_grad[activation::kOut].shape_[0], 1, 1, 1);
      m_out_grad = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        out_grad[activation::kOut], dshape, s);
      m_out_data = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        out_data[activation::kOut], dshape, s);
      m_in_grad = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        in_grad[activation::kData], dshape, s);
    } else if (out_grad[activation::kOut].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[activation::kOut].shape_[0],
                               out_grad[activation::kOut].shape_[1], 1, 1);
      m_out_grad = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        out_grad[activation::kOut], dshape, s);
      m_out_data = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        out_data[activation::kOut], dshape, s);
      m_in_grad = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        in_grad[activation::kData], dshape, s);
    } else if (out_grad[activation::kOut].ndim() == 3) {
      Shape<4> dshape = Shape4(out_grad[activation::kOut].shape_[0],
        out_grad[activation::kOut].shape_[1],
        out_grad[activation::kOut].shape_[2], 1);
      m_out_grad = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        out_grad[activation::kOut], dshape, s);
      m_out_data = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        out_data[activation::kOut], dshape, s);
      m_in_grad = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        in_grad[activation::kData], dshape, s);
    } else {
      m_out_grad = mkl_experimental_direct_get<xpu, 4, DType>(out_grad[activation::kOut], s);
      m_out_data = mkl_experimental_direct_get<xpu, 4, DType>(out_data[activation::kOut], s);
      m_in_grad = mkl_experimental_direct_get<xpu, 4, DType>(in_grad[activation::kData], s);
    }
    dnnError_t e;
    void* relu_res[dnnResourceNumber];

    void* bottom_data = NULL;
#if MKL_EXPERIMENTAL == 1
    bottom_data = reinterpret_cast<void *>(mkl_prv_data<DType>(out_data[activation::kOut]));
#endif
    if (NULL == bottom_data) {
      bottom_data = reinterpret_cast<void *>(const_cast<DType*>(m_out_data.dptr_));
    }
    relu_res[dnnResourceSrc] = bottom_data;
    relu_res[dnnResourceDiffDst] = bwd_top_diff_->get_converted_prv(m_out_grad.dptr_,
                true, out_grad[activation::kOut]);
    relu_res[dnnResourceDiffSrc] = bwd_bottom_diff_->get_output_ptr(
      m_in_grad.dptr_, bwd_bottom_diff_, in_grad[activation::kData]);
    e = dnnExecute<DType>(reluBwd_, relu_res);
    CHECK_EQ(e, E_SUCCESS);
#if MKL_EXPERIMENTAL == 0
    if (bwd_bottom_diff_->conversion_needed()) {
      bwd_bottom_diff_->convert_from_prv(m_in_grad.dptr_);
    }
#endif
  }

 private:
  bool init_mkldnn_;
  std::shared_ptr<MKLData<DType> > fwd_top_data_;
  std::shared_ptr<MKLData<DType> > fwd_bottom_data_;
  std::shared_ptr<MKLData<DType> > bwd_top_diff_;
  std::shared_ptr<MKLData<DType> > bwd_bottom_diff_;
  dnnPrimitive_t reluFwd_, reluBwd_;
};  // class MKLReluOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKL_RELU_INL_H_
