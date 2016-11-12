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
* \file mkl_batch_norm-inl.h
* \brief
* \author lingyan.guo@intel.com
*         zhenlin.luo@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKL_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_MKL_MKL_BATCH_NORM_INL_H_

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
class MKLBatchNormOp : public Operator {
 public:
  explicit MKLBatchNormOp(BatchNormParam param) {
    this->param_ = param;
    fwd_top_data = MKLData<DType>::create();
    fwd_bottom_data = MKLData<DType>::create();
    bwd_top_diff = MKLData<DType>::create();
    bwd_bottom_diff = MKLData<DType>::create();
  }
  virtual ~MKLBatchNormOp() {
    if (batchNormFwd != NULL) dnnDelete<DType>(batchNormFwd);
    if (batchNormBwdData != NULL) dnnDelete<DType>(batchNormBwdData);
    if (batchNormBwdScaleShift != NULL) dnnDelete<DType>(batchNormBwdScaleShift);
    dnnLayoutDelete<DType>(layout_usr_);
    dnnReleaseBuffer<DType>(workspace_buffer_);
    dnnReleaseBuffer<DType>(scaleShift_buffer_);
  }
  std::string getName() {
    return "MKLBatchNormOp";
  }

 private:
  void LayerSetUp(const mshadow::Tensor<xpu, 4, DType> &data,
                  const mshadow::Tensor<xpu, 4, DType> &out) {
    eps_ = param_.eps;
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

    // Names are for debugging only
    fwd_bottom_data->name = "fwd_bottom_data   @ " + getName();
    fwd_top_data->name = "fwd_top_data      @ " + getName();
    bwd_bottom_diff->name = "bwd_bottom_diff   @ " + getName();
    bwd_top_diff->name = "bwd_top_diff      @ " + getName();

    dnnError_t e;
    e = dnnLayoutCreate<DType>(&layout_usr_, dim, sizes, strides);
    CHECK_EQ(e, E_SUCCESS);

    fwd_bottom_data->create_user_layout(dim, sizes, strides);
    fwd_top_data->create_user_layout(dim, sizes, strides);
    bwd_bottom_diff->create_user_layout(dim, sizes, strides);
    bwd_top_diff->create_user_layout(dim, sizes, strides);

    workspace_buffer_ = NULL;
    scaleShift_buffer_ = NULL;

    // Primitives will be allocated during the first fwd pass
    batchNormFwd = NULL;
    batchNormBwdData = NULL;
    batchNormBwdScaleShift = NULL;
  }

 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(aux_states.size(), 2);
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 3);
      CHECK_EQ(req.size(), 3);
    } else {
      CHECK_GE(out_data.size(), 1);
      CHECK_GE(req.size(), 1);
      CHECK_EQ(req[batchnorm::kOut], kWriteTo);
    }

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType>  data;
    Tensor<xpu, 4, DType>  out;
    if (in_data[batchnorm::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[batchnorm::kData].shape_[0],
                               in_data[batchnorm::kData].shape_[1], 1, 1);
      data = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        in_data[batchnorm::kData], dshape, s);
      out = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        out_data[batchnorm::kOut], dshape, s);
    } else {
      data = mkl_experimental_direct_get<xpu, 4, DType>(in_data[batchnorm::kData], s);
      out = mkl_experimental_direct_get<xpu, 4, DType>(out_data[batchnorm::kOut], s);
    }
    Tensor<xpu, 1, DType> slope = in_data[batchnorm::kGamma].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> bias = in_data[batchnorm::kBeta].get<xpu, 1, DType>(s);
#if MKL_EXPERIMENTAL == 0
    Tensor<xpu, 1, DType> moving_mean = aux_states[batchnorm::kMovingMean].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> moving_var = aux_states[batchnorm::kMovingVar].get<xpu, 1, DType>(s);
#endif
    if (param_.fix_gamma) slope = 1.f;
    if (!init_mkldnn_) {
      LayerSetUp(data, out);
      init_mkldnn_ = true;
    }
    void* bottom_data = NULL;
#if MKL_EXPERIMENTAL == 1
    bottom_data =
          reinterpret_cast<void *>(mkl_prv_data<DType>(in_data[batchnorm::kData]));
#endif

    int is_first_pass = 0;
#if MKL_EXPERIMENTAL == 1
    if (NULL != bottom_data) {
      // Is it the first pass? Create a primitive.
      if (batchNormFwd == NULL) {
        is_first_pass = 1;
        std::shared_ptr<MKLMemHolder> bottom_data_mem = in_data[batchnorm::kData].Mkl_mem_;
        std::shared_ptr<PrvMemDescr> bottom_prv_desc =
          bottom_data_mem->get_prv_descriptor();
        CHECK(bottom_prv_desc->get_descr_type() ==
          PrvMemDescr::PRV_DESCR_MKL2017);
        std::shared_ptr<MKLData<DType> > mem_descr
          = std::static_pointer_cast<MKLData<DType>>(bottom_prv_desc);
        CHECK(mem_descr != NULL);
        fwd_bottom_data = mem_descr;
        dnnError_t e;
        e = dnnBatchNormalizationCreateForward<DType>(
              &batchNormFwd, NULL, mem_descr->layout_int, eps_);
        CHECK_EQ(e, E_SUCCESS);
        fwd_top_data->create_internal_layout(batchNormFwd, dnnResourceDst);
        bwd_top_diff->create_internal_layout(batchNormFwd, dnnResourceDst);
        bwd_bottom_diff->create_internal_layout(batchNormFwd, dnnResourceSrc);


        e = dnnBatchNormalizationCreateBackwardData<DType>(
              &batchNormBwdData, NULL, mem_descr->layout_int, eps_);
        CHECK_EQ(e, E_SUCCESS);
        if (true) {
          e = dnnBatchNormalizationCreateBackwardScaleShift<DType>(
                &batchNormBwdScaleShift, NULL, mem_descr->layout_int, eps_);
          CHECK_EQ(e, E_SUCCESS);
        }
      }
    }
#endif
    if (NULL == bottom_data) {
      if (batchNormFwd == NULL) {
        // First pass
        is_first_pass = 1;

        dnnError_t e;
        e = dnnBatchNormalizationCreateForward<DType>(
              &batchNormFwd, NULL, layout_usr_, eps_);
        CHECK_EQ(e, E_SUCCESS);

        e = dnnBatchNormalizationCreateBackwardData<DType>(
              &batchNormBwdData, NULL, layout_usr_, eps_);
        CHECK_EQ(e, E_SUCCESS);
        if (true) {
        e = dnnBatchNormalizationCreateBackwardScaleShift<DType>(
              &batchNormBwdScaleShift, NULL, layout_usr_, eps_);
        CHECK_EQ(e, E_SUCCESS);
        }
      }
      bottom_data =
        reinterpret_cast<void *>(data.dptr_);
    }
    if (is_first_pass == 1) {
      dnnError_t e;

      dnnLayout_t workspace_buffer_l = NULL;
      e = dnnLayoutCreateFromPrimitive<DType>(
            &workspace_buffer_l, batchNormFwd, dnnResourceWorkspace);
      CHECK_EQ(e, E_SUCCESS);
      e = dnnAllocateBuffer<DType>(
            reinterpret_cast<void**>(&workspace_buffer_), workspace_buffer_l);
      CHECK_EQ(e, E_SUCCESS);
      dnnLayoutDelete<DType>(workspace_buffer_l);

      dnnLayout_t scaleShift_buffer_l = NULL;
      e = dnnLayoutCreateFromPrimitive<DType>(
            &scaleShift_buffer_l, batchNormFwd, dnnResourceScaleShift);
      CHECK_EQ(e, E_SUCCESS);
      e = dnnAllocateBuffer<DType>(
            reinterpret_cast<void**>(&scaleShift_buffer_), scaleShift_buffer_l);
      CHECK_EQ(e, E_SUCCESS);
      dnnLayoutDelete<DType>(scaleShift_buffer_l);
      if (true /*!use_weight_bias_*/) {
        for (int i = 0; i < channels_; i++) {
          scaleShift_buffer_[i] = 1.0;
          scaleShift_buffer_[channels_ + i] = 0;
        }
      }
    }
    if (true) {  // use_weight_bias_
      for (int i = 0; i < channels_; i++) {
        scaleShift_buffer_[i] = (slope.dptr_)[i];
        scaleShift_buffer_[channels_ + i] = (bias.dptr_)[i];
      }
    }
#if MKL_EXPERIMENTAL == 0
    if (ctx.is_train && !param_.use_global_stats) {
      Tensor<xpu, 1, DType> mean = out_data[batchnorm::kMean].get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> var = out_data[batchnorm::kVar].get<xpu, 1, DType>(s);
      CHECK(req[batchnorm::kMean] == kNullOp || req[batchnorm::kMean] == kWriteTo);
      CHECK(req[batchnorm::kVar] == kNullOp || req[batchnorm::kVar] == kWriteTo);
      // The first three steps must be enforced.
      const DType scale = static_cast<DType>(in_data[batchnorm::kData].shape_[1]) /
          static_cast<DType>(in_data[batchnorm::kData].shape_.Size());
      mean = scale * sumall_except_dim<1>(data);
      var = scale * sumall_except_dim<1>(F<mshadow_op::square>(
          data - broadcast<1>(mean, data.shape_)));
#endif
      dnnError_t e;
      void* BatchNorm_res[dnnResourceNumber];
      BatchNorm_res[dnnResourceSrc] = bottom_data;
      BatchNorm_res[dnnResourceWorkspace] = workspace_buffer_;
      BatchNorm_res[dnnResourceScaleShift] = scaleShift_buffer_;
      if (fwd_top_data->conversion_needed()) {
#if MKL_EXPERIMENTAL == 1
      std::shared_ptr<MKLMemHolder> topDnnChunk = out_data[batchnorm::kOut].Mkl_mem_;
      topDnnChunk->set_prv_descriptor(fwd_top_data);
#endif
        BatchNorm_res[dnnResourceDst] =
          fwd_top_data->prv_ptr();
      } else {
        BatchNorm_res[dnnResourceDst] =
          reinterpret_cast<void *>(out.dptr_);
      }
      e = dnnExecute<DType>(batchNormFwd, BatchNorm_res);
      CHECK_EQ(e, E_SUCCESS);
#if MKL_EXPERIMENTAL == 0
      if (fwd_top_data->conversion_needed()) {
        fwd_top_data->convert_from_prv(out.dptr_);
      }
    } else {
      Assign(out, req[batchnorm::kOut], broadcast<1>(slope /
                                          F<mshadow_op::square_root>(moving_var + param_.eps),
                                          data.shape_) * data +
             broadcast<1>(bias - (slope * moving_mean) /
                          F<mshadow_op::square_root>(moving_var + param_.eps), data.shape_));
    }
#endif
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
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 3);
    CHECK_EQ(in_grad.size(), 3);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data, grad, grad_in;

    if (in_data[batchnorm::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[batchnorm::kOut].shape_[0],
                               out_grad[batchnorm::kOut].shape_[1], 1, 1);
      data = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        in_data[batchnorm::kData], dshape, s);
      grad = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        out_grad[batchnorm::kOut], dshape, s);
      grad_in = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        in_grad[batchnorm::kData], dshape, s);
    } else {
      data = mkl_experimental_direct_get<xpu, 4, DType>(in_data[batchnorm::kData], s);
      grad = mkl_experimental_direct_get<xpu, 4, DType>(out_grad[batchnorm::kOut], s);
      grad_in = mkl_experimental_direct_get<xpu, 4, DType>(in_grad[batchnorm::kData], s);
    }

    Tensor<xpu, 1, DType> slope = in_data[batchnorm::kGamma].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> gslope = in_grad[batchnorm::kGamma].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> gbias = in_grad[batchnorm::kBeta].get<xpu, 1, DType>(s);
#if MKL_EXPERIMENTAL == 0
    Tensor<xpu, 1, DType> mean = out_data[batchnorm::kMean].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> var = out_data[batchnorm::kVar].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> moving_mean = aux_states[batchnorm::kMovingMean].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> moving_var = aux_states[batchnorm::kMovingVar].get<xpu, 1, DType>(s);
#endif

    if (param_.fix_gamma) {
      slope = 1.f;
    }
#if MKL_EXPERIMENTAL == 0
    if (ctx.is_train && !param_.use_global_stats) {
      moving_mean = moving_mean * param_.momentum + mean * (1 - param_.momentum);
      moving_var = moving_var * param_.momentum + var * (1 - param_.momentum);
#endif
      void* bottom_data = NULL;
#if MKL_EXPERIMENTAL == 1
      bottom_data =
          reinterpret_cast<void *>(mkl_prv_data<DType>(in_data[batchnorm::kData]));
#endif
      if (NULL == bottom_data) {
        bottom_data =

        reinterpret_cast<void *>(data.dptr_);
      }

      dnnError_t e;
      void* BatchNorm_res[dnnResourceNumber];
      BatchNorm_res[dnnResourceSrc] = bottom_data;
      BatchNorm_res[dnnResourceWorkspace] = workspace_buffer_;
      BatchNorm_res[dnnResourceScaleShift] = scaleShift_buffer_;

      std::shared_ptr<MKLMemHolder> top_diff_mem =
#if MKL_EXPERIMENTAL == 1
        out_grad[batchnorm::kOut].Mkl_mem_;
#else
        NULL;
#endif
    BatchNorm_res[dnnResourceDiffDst] = bwd_top_diff->get_converted_prv(grad.dptr_,
                                                                        true, top_diff_mem);


    std::shared_ptr<MKLMemHolder> bottom_diff_mem =
#if MKL_EXPERIMENTAL == 1
      in_grad[batchnorm::kData].Mkl_mem_;
#else
      NULL;
#endif
      if (bwd_bottom_diff->conversion_needed()) {
#if MKL_EXPERIMENTAL == 1
      bottom_diff_mem->set_prv_descriptor(bwd_bottom_diff);
#endif
        BatchNorm_res[dnnResourceDiffSrc] = bwd_bottom_diff->prv_ptr();
      } else {
        BatchNorm_res[dnnResourceDiffSrc] = grad_in.dptr_;
      }

      e = dnnExecute<DType>(batchNormBwdData, BatchNorm_res);
      CHECK_EQ(e, E_SUCCESS);
#if MKL_EXPERIMENTAL == 0
      if (bwd_bottom_diff->conversion_needed()) {
        bwd_bottom_diff->convert_from_prv(grad_in.dptr_);
      }
#endif
      if (true) {  // use_weight_bias_
        void* BatchNormBwdScaleShift_res[dnnResourceNumber];
        BatchNormBwdScaleShift_res[dnnResourceSrc] = bottom_data;
        BatchNormBwdScaleShift_res[dnnResourceWorkspace] = workspace_buffer_;
        BatchNormBwdScaleShift_res[dnnResourceDiffScaleShift] = scaleShift_buffer_;
        BatchNormBwdScaleShift_res[dnnResourceDiffDst] =
          BatchNorm_res[dnnResourceDiffDst];
        e = dnnExecute<DType>(batchNormBwdScaleShift, BatchNormBwdScaleShift_res);
        CHECK_EQ(e, E_SUCCESS);
        // Store ScaleShift blobs
        DType* diff_scale = gslope.dptr_;
        DType* diff_shift = gbias.dptr_;
        for (int i = 0; i < channels_; i++) {
          diff_scale[i] = scaleShift_buffer_[i];
          diff_shift[i] = 0;
          if (true) {
            diff_shift[i] = scaleShift_buffer_[channels_ + i];
          }
        }
      }
#if MKL_EXPERIMENTAL == 0
    } else {
      // use global statistics with freeze moving mean and var.
      if (!param_.fix_gamma) {
        Assign(gslope, req[batchnorm::kGamma],
          sumall_except_dim<1>(
          grad * (data - broadcast<1>(moving_mean, data.shape_)) /
          F<mshadow_op::square_root>(broadcast<1>(moving_var + param_.eps, data.shape_))));
      } else {
        Assign(gslope, req[batchnorm::kGamma], 0.0f);
      }
      Assign(gbias, req[batchnorm::kBeta], sumall_except_dim<1>(grad));
      Assign(grad_in, req[batchnorm::kData], (grad * broadcast<1>(slope, data.shape_)) *
        broadcast<1>(
        1.0f / F<mshadow_op::square_root>(moving_var + param_.eps), data.shape_));
    }
#endif
  }

 private:
  BatchNormParam param_;
  DType eps_;
  bool use_weight_bias_;

  int num_;
  int channels_;
  int height_;
  int width_;
  bool init_mkldnn_ = false;
  std::shared_ptr<MKLData<DType> > fwd_top_data;
  std::shared_ptr<MKLData<DType> > fwd_bottom_data;
  std::shared_ptr<MKLData<DType> > bwd_top_diff;
  std::shared_ptr<MKLData<DType> > bwd_bottom_diff;
  dnnPrimitive_t batchNormFwd = NULL;
  dnnPrimitive_t batchNormBwdData = NULL;
  dnnPrimitive_t batchNormBwdScaleShift = NULL;
  DType *workspace_buffer_ = NULL;
  DType *scaleShift_buffer_ = NULL;
  dnnLayout_t layout_usr_ = NULL;
};  // class BatchNormOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKL_BATCH_NORM_INL_H_
