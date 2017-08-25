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
* \file mkl_lrn-inl.h
* \brief
* \author zhenlin.luo@intel.com
*         lingyan.guo@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKL_LRN_INL_H_
#define MXNET_OPERATOR_MKL_MKL_LRN_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "./mkl_util-inl.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class MKLLRNOp : public Operator {
 public:
  static std::string getName() {
    return "MKLLRNOp";
  }

  explicit MKLLRNOp(LRNParam param) :
    lrnFwd(static_cast<dnnPrimitive_t>(NULL)),
    lrnBwd(static_cast<dnnPrimitive_t>(NULL)),
    lrn_buffer_(NULL) {
    this->param_ = param;
    fwd_top_data_ = MKLData<DType>::create();
    fwd_bottom_data_ = MKLData<DType>::create();
    bwd_top_diff_ = MKLData<DType>::create();
    bwd_bottom_diff_ = MKLData<DType>::create();
    init_mkldnn_ = false;
  }

  virtual ~MKLLRNOp() {
    if (lrnFwd != NULL) {
      dnnDelete<DType>(lrnFwd);
      lrnFwd = NULL;
    }
    if (lrnBwd != NULL) {
      dnnDelete<DType>(lrnBwd);
      lrnBwd = NULL;
    }
    dnnReleaseBuffer<DType>(lrn_buffer_);
  }

 private:
  void LayerSetup(const mshadow::Tensor<xpu, 4, DType> &data,
                  const mshadow::Tensor<xpu, 4, DType> &out) {
    size_ = param_.nsize;
    CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local size";

    alpha_ = param_.alpha;
    beta_ = param_.beta;
    k_ = param_.knorm;
    size_t dim = 4, sizes[4], strides[4];
    channels_ = data.shape_[1];
    height_ = data.shape_[2];
    width_ = data.shape_[3];
    num_ = data.shape_[0];
    sizes[0] = width_;
    sizes[1] = height_;
    sizes[2] = channels_;
    sizes[3] = num_;

    strides[0] = 1;
    strides[1] = sizes[0];
    strides[2] = sizes[0] * sizes[1];
    strides[3] = sizes[0] * sizes[1] * sizes[2];

    fwd_bottom_data_->name = "fwd_bottom_data_   @ " + getName();
    fwd_top_data_->name = "fwd_top_data_      @ " + getName();
    bwd_top_diff_->name = "bwd_top_diff_      @ " + getName();
    bwd_bottom_diff_->name = "bwd_bottom_diff_   @ " + getName();

    fwd_bottom_data_->create_user_layout(dim, sizes, strides);
    fwd_top_data_->create_user_layout(dim, sizes, strides);
    bwd_bottom_diff_->create_user_layout(dim, sizes, strides);
    bwd_top_diff_->create_user_layout(dim, sizes, strides);
  }

 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 2U);
    CHECK_EQ(param_.nsize % 2, 1U) << "LRN only supports odd values for local_size";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data = mkl_experimental_direct_get<xpu, 4, DType>(
      in_data[lrn_enum::kData], s);
    Tensor<xpu, 4, DType> out = mkl_experimental_direct_get<xpu, 4, DType>(
      out_data[lrn_enum::kOut], s);
    if (!init_mkldnn_) {
      LayerSetup(data, out);
      init_mkldnn_ = true;
    }

    const void* bottom_data = NULL;
#if MKL_EXPERIMENTAL == 1
    bottom_data =
          reinterpret_cast<void*>(mkl_prv_data<DType>(in_data[lrn_enum::kData]));
#endif
#if MKL_EXPERIMENTAL == 1
    if (NULL != bottom_data) {
      if (lrnFwd == NULL) {
        std::shared_ptr<MKLMemHolder> bottom_data_mem =
          in_data[lrn_enum::kData].Mkl_mem_;
        std::shared_ptr<PrvMemDescr> bottom_prv_descriptor =
          bottom_data_mem->get_prv_descriptor();
        CHECK_EQ(bottom_prv_descriptor->get_descr_type(),
            PrvMemDescr::PRV_DESCR_MKL2017);
        std::shared_ptr<MKLData<DType> > mem_descr
          = std::static_pointer_cast<MKLData<DType>>(bottom_prv_descriptor);
        CHECK(mem_descr != nullptr);
        fwd_bottom_data_ = mem_descr;

        dnnError_t e;
        dnnLayout_t lrn_buffer_l = NULL;

        e = dnnLRNCreateForward<DType>(&lrnFwd, NULL, fwd_bottom_data_->layout_int,
                                       size_, alpha_, beta_, k_);
        CHECK_EQ(e, E_SUCCESS);

        fwd_top_data_->create_internal_layout(lrnFwd, dnnResourceDst);

        e = dnnLRNCreateBackward<DType>(&lrnBwd, NULL,
                                        fwd_bottom_data_->layout_int, fwd_bottom_data_->layout_int,
                                        size_, alpha_, beta_, k_);
        CHECK_EQ(e, E_SUCCESS);

        e = dnnLayoutCreateFromPrimitive<DType>(
              &lrn_buffer_l, lrnFwd, dnnResourceWorkspace);
        CHECK_EQ(e, E_SUCCESS);
        e = dnnAllocateBuffer<DType>(
              reinterpret_cast<void **>(&lrn_buffer_), lrn_buffer_l);
        CHECK_EQ(e, E_SUCCESS);
        dnnLayoutDelete<DType>(lrn_buffer_l);

        bwd_top_diff_->create_internal_layout(lrnBwd, dnnResourceDiffDst);
        bwd_bottom_diff_->create_internal_layout(lrnBwd, dnnResourceDiffSrc);
      }
    }
#endif
    if (bottom_data == NULL) {
      if (lrnFwd == NULL) {
        dnnError_t e;
        dnnLayout_t lrn_buffer_l = NULL;
        e = dnnLRNCreateForward<DType>(&lrnFwd, NULL, fwd_bottom_data_->layout_usr,
                                       size_, alpha_, beta_, k_);
        CHECK_EQ(e, E_SUCCESS);

        e = dnnLayoutCreateFromPrimitive<DType>(
              &lrn_buffer_l, lrnFwd, dnnResourceWorkspace);
        CHECK_EQ(e, E_SUCCESS);
        e = dnnAllocateBuffer<DType>(
              reinterpret_cast<void **>(&lrn_buffer_), lrn_buffer_l);
        CHECK_EQ(e, E_SUCCESS);
        dnnLayoutDelete<DType>(lrn_buffer_l);

        e = dnnLRNCreateBackward<DType>(&lrnBwd, NULL,
                                        fwd_bottom_data_->layout_usr, fwd_bottom_data_->layout_usr,
                                        size_, alpha_, beta_, k_);
        CHECK_EQ(e, E_SUCCESS);
      }
      bottom_data = data.dptr_;
    }

    dnnError_t e;
    void* lrn_res[dnnResourceNumber];
    lrn_res[dnnResourceSrc] = const_cast<void*>(bottom_data);

    lrn_res[dnnResourceDst] = fwd_top_data_->get_output_ptr(
      out.dptr_, fwd_top_data_, out_data[lrn_enum::kOut]);
    lrn_res[dnnResourceWorkspace] = lrn_buffer_;
    e = dnnExecute<DType>(lrnFwd, lrn_res);
    CHECK_EQ(e, E_SUCCESS);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 2);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> grad = mkl_experimental_direct_get<xpu, 4, DType>(
      out_grad[lrn_enum::kOut], s);
    Tensor<xpu, 4, DType> data = mkl_experimental_direct_get<xpu, 4, DType>(
      in_data[lrn_enum::kData], s);
    Tensor<xpu, 4, DType> grad_in = mkl_experimental_direct_get<xpu, 4, DType>(
      in_grad[lrn_enum::kData], s);
    dnnError_t e;
    void* lrn_res[dnnResourceNumber];
    lrn_res[dnnResourceDiffDst] =
      bwd_top_diff_->get_converted_prv(grad.dptr_, true, out_grad[lrn_enum::kOut]);
    lrn_res[dnnResourceWorkspace] = lrn_buffer_;
    lrn_res[dnnResourceSrc] =
      fwd_bottom_data_->get_converted_prv(data.dptr_, false, in_data[lrn_enum::kData]);

    lrn_res[dnnResourceDiffSrc] = bwd_bottom_diff_->get_output_ptr(
      grad_in.dptr_, bwd_bottom_diff_, in_grad[lrn_enum::kData]);
    e = dnnExecute<DType>(lrnBwd, lrn_res);
    CHECK_EQ(e, E_SUCCESS);
  }

 private:
  LRNParam param_;
  int size_;
  int pre_pad_;
  DType alpha_;
  DType beta_;
  DType k_;
  int num_;
  int channels_;
  int height_;
  int width_;
  bool init_mkldnn_;

 private:
  dnnPrimitive_t lrnFwd, lrnBwd;
  std::shared_ptr<MKLData<DType> > fwd_top_data_;
  std::shared_ptr<MKLData<DType> > fwd_bottom_data_;

  std::shared_ptr<MKLData<DType> > bwd_top_diff_;
  std::shared_ptr<MKLData<DType> > bwd_bottom_diff_;

  DType *lrn_buffer_;
};  // class LocalResponseNormOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKL_LRN_INL_H_

