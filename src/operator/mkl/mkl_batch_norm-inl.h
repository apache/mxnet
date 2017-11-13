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
#include <mxnet/storage.h>
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
    scaleShift_space.dptr = NULL;
    scaleShiftDiff_space.dptr = NULL;
  }
  virtual ~MKLBatchNormOp() {
    if (batchNormFwdInference != NULL) dnnDelete<DType>(batchNormFwdInference);
    if (batchNormFwdTraining != NULL) dnnDelete<DType>(batchNormFwdTraining);
    if (batchNormBwdScaleShift != NULL) dnnDelete<DType>(batchNormBwdScaleShift);
    dnnLayoutDelete<DType>(layout_usr_);
    if (scaleShift_space.dptr)
      Storage::Get()->Free(scaleShift_space);
    if (scaleShiftDiff_space.dptr)
      Storage::Get()->Free(scaleShiftDiff_space);
  }
  static std::string getName() {
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

    // Primitives will be allocated during the first fwd pass
    batchNormFwdInference = NULL;
    batchNormFwdTraining = NULL;
    batchNormBwdScaleShift = NULL;
    int scaleShift_size = channels_*2*sizeof(DType);
    scaleShift_space = Storage::Get()->Alloc(scaleShift_size, Context::CPU());
    scaleShiftDiff_space = Storage::Get()->Alloc(scaleShift_size, Context::CPU());
    DType * scaleShift_buf = reinterpret_cast<DType*>(scaleShift_space.dptr);
    /*!use_weight_bias_*/
    for (int i = 0; i < channels_; i++) {
        scaleShift_buf[i] = 1.0;
        scaleShift_buf[channels_ + i] = 0;
    }
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

    // const real_t scale = static_cast<real_t>(in_data[batchnorm::kData].shape_[1]) /
    //   static_cast<real_t>(in_data[batchnorm::kData].shape_.Size());

    Tensor<xpu, 1, DType> slope = in_data[batchnorm::kGamma].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> bias = in_data[batchnorm::kBeta].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> moving_mean = aux_states[batchnorm::kMovingMean].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> moving_var = aux_states[batchnorm::kMovingVar].get<xpu, 1, DType>(s);

    if (param_.fix_gamma)
      slope = 1.f;

    dnnError_t e;
    if (!init_mkldnn_) {
      LayerSetUp(data, out);
      init_mkldnn_ = true;
    }
    void* bottom_data = NULL;
#if MKL_EXPERIMENTAL == 1
    bottom_data =
          reinterpret_cast<void *>(mkl_prv_data<DType>(in_data[batchnorm::kData]));
#endif
    int bwd_flags = dnnUseScaleShift;
    if (param_.use_global_stats)
      bwd_flags = dnnUseScaleShift | dnnUseInputMeanVariance;
#if MKL_EXPERIMENTAL == 1
    if (NULL != bottom_data) {
      // Is it the first pass? Create a primitive.
      if (batchNormFwdInference == NULL) {
        std::shared_ptr<MKLMemHolder> bottom_data_mem = in_data[batchnorm::kData].Mkl_mem_;
        std::shared_ptr<PrvMemDescr> bottom_prv_desc = bottom_data_mem->get_prv_descriptor();
        CHECK(bottom_prv_desc->get_descr_type() == PrvMemDescr::PRV_DESCR_MKL2017);
        std::shared_ptr<MKLData<DType> > mem_descr
          = std::static_pointer_cast<MKLData<DType>>(bottom_prv_desc);
        CHECK(mem_descr != NULL);
        fwd_bottom_data = mem_descr;

        e = dnnBatchNormalizationCreateForward_v2<DType>(
             &batchNormFwdInference, NULL, mem_descr->layout_int, eps_,
             dnnUseInputMeanVariance | dnnUseScaleShift);
        CHECK_EQ(e, E_SUCCESS);

        e = dnnBatchNormalizationCreateForward_v2<DType>(
              &batchNormFwdTraining, NULL, mem_descr->layout_int, eps_,
              dnnUseScaleShift);
        CHECK_EQ(e, E_SUCCESS);

        fwd_top_data->create_internal_layout(batchNormFwdInference, dnnResourceDst);
        bwd_top_diff->create_internal_layout(batchNormFwdInference, dnnResourceDst);
        bwd_bottom_diff->create_internal_layout(batchNormFwdInference, dnnResourceSrc);

        e = dnnBatchNormalizationCreateBackward_v2<DType>(
                &batchNormBwdScaleShift, NULL, mem_descr->layout_int, eps_, bwd_flags);
        CHECK_EQ(e, E_SUCCESS);
      }
    }
#endif
    if (NULL == bottom_data) {
      if (batchNormFwdInference == NULL) {
        e = dnnBatchNormalizationCreateForward_v2<DType>(
          &batchNormFwdInference, NULL, layout_usr_, eps_,
          dnnUseInputMeanVariance | dnnUseScaleShift);
        CHECK_EQ(e, E_SUCCESS);

        e = dnnBatchNormalizationCreateForward_v2<DType>(
              &batchNormFwdTraining, NULL, layout_usr_, eps_, dnnUseScaleShift);
        CHECK_EQ(e, E_SUCCESS);

        e = dnnBatchNormalizationCreateBackward_v2<DType>(
              &batchNormBwdScaleShift, NULL, layout_usr_, eps_, bwd_flags);
        CHECK_EQ(e, E_SUCCESS);
      }
      bottom_data = reinterpret_cast<void *>(data.dptr_);
    }

    DType * scaleShift_buf = reinterpret_cast<DType*>(scaleShift_space.dptr);
     // use_weight_bias_
    for (int i = 0; i < channels_; i++) {
        scaleShift_buf[i] = (slope.dptr_)[i];
    }
    for (int i = 0; i < channels_; i++) {
      scaleShift_buf[channels_ + i] = (bias.dptr_)[i];
    }

    void* BatchNorm_res[dnnResourceNumber];
    BatchNorm_res[dnnResourceSrc] = bottom_data;
    BatchNorm_res[dnnResourceScaleShift] = scaleShift_space.dptr;

    BatchNorm_res[dnnResourceDst] = fwd_top_data->get_output_ptr(out.dptr_,
      fwd_top_data, out_data[batchnorm::kOut]);
    if (ctx.is_train && !param_.use_global_stats) {
      Tensor<xpu, 1, DType> mean = out_data[batchnorm::kMean].get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> var = out_data[batchnorm::kVar].get<xpu, 1, DType>(s);
      CHECK(req[batchnorm::kMean] == kNullOp || req[batchnorm::kMean] == kWriteTo);
      CHECK(req[batchnorm::kVar] == kNullOp || req[batchnorm::kVar] == kWriteTo);
      BatchNorm_res[dnnResourceMean] = mean.dptr_;
      BatchNorm_res[dnnResourceVariance] = var.dptr_;
      e = dnnExecute<DType>(batchNormFwdTraining, BatchNorm_res);
      CHECK_EQ(e, E_SUCCESS);
    } else {
      BatchNorm_res[dnnResourceMean] = moving_mean.dptr_;
      BatchNorm_res[dnnResourceVariance] = moving_var.dptr_;
      e = dnnExecute<DType>(batchNormFwdInference, BatchNorm_res);
      CHECK_EQ(e, E_SUCCESS);
    }

#if MKL_EXPERIMENTAL == 0
    if (fwd_top_data->conversion_needed()) {
      fwd_top_data->convert_from_prv(out.dptr_);
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
    Tensor<xpu, 1, DType> mean = out_data[batchnorm::kMean].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> var = out_data[batchnorm::kVar].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> moving_mean = aux_states[batchnorm::kMovingMean].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> moving_var = aux_states[batchnorm::kMovingVar].get<xpu, 1, DType>(s);

    if (param_.fix_gamma)  slope = 1.f;

    void* bottom_data = NULL;
#if MKL_EXPERIMENTAL == 1
    bottom_data = reinterpret_cast<void *>(mkl_prv_data<DType>(in_data[batchnorm::kData]));
#endif
    if (NULL == bottom_data)
      bottom_data = reinterpret_cast<void *>(data.dptr_);

    dnnError_t e;
    void* BatchNorm_res[dnnResourceNumber];
    BatchNorm_res[dnnResourceSrc] = bottom_data;
    BatchNorm_res[dnnResourceScaleShift] = scaleShift_space.dptr;
    if (ctx.is_train && !param_.use_global_stats) {
      int size = mean.size(0);  // Tensor<xpu, 1, DType>
      float * moving_mean_ptr = reinterpret_cast<float*>(moving_mean.dptr_);
      float * mean_ptr = reinterpret_cast<float*>(mean.dptr_);
      float * moving_var_ptr = reinterpret_cast<float*>(moving_var.dptr_);
      float * var_ptr = reinterpret_cast<float*>(var.dptr_);
      float minus_mom = (1 - param_.momentum);
      for (int i = 0; i < size; i++) {
        moving_mean_ptr[i] = moving_mean_ptr[i] * param_.momentum
          + mean_ptr[i] * minus_mom;
      }
      for (int i = 0; i < size; i++) {
        moving_var_ptr[i] = moving_var_ptr[i] * param_.momentum
          + var_ptr[i] * minus_mom;
      }
      BatchNorm_res[dnnResourceMean] = mean.dptr_;
      BatchNorm_res[dnnResourceVariance] = var.dptr_;
    } else {
      BatchNorm_res[dnnResourceMean] = moving_mean.dptr_;
      BatchNorm_res[dnnResourceVariance] = moving_var.dptr_;
    }


    BatchNorm_res[dnnResourceDiffSrc] = bwd_bottom_diff->get_output_ptr(grad_in.dptr_,
      bwd_bottom_diff, in_grad[batchnorm::kData]);
    BatchNorm_res[dnnResourceDiffDst] = bwd_top_diff->get_converted_prv(grad.dptr_,
             true, out_grad[batchnorm::kOut]);
    BatchNorm_res[dnnResourceDiffScaleShift] = scaleShiftDiff_space.dptr;
    e = dnnExecute<DType>(batchNormBwdScaleShift, BatchNorm_res);
    CHECK_EQ(e, E_SUCCESS);
#if MKL_EXPERIMENTAL == 0
    if (bwd_bottom_diff->conversion_needed()) {
      bwd_bottom_diff->convert_from_prv(grad_in.dptr_);
    }
#endif
    DType * scaleShiftDiff_buf = reinterpret_cast<DType*>(scaleShiftDiff_space.dptr);
    if (!param_.fix_gamma) {
      // Store ScaleShift blobs
      DType* diff_scale = gslope.dptr_;
      for (int i = 0; i < channels_; i++) {
        diff_scale[i] = scaleShiftDiff_buf[i];
      }
    } else {
      int gslope_size = gslope.size(0);
      float * gslope_ptr = reinterpret_cast<float*>(gslope.dptr_);
      for (int i = 0; i < gslope_size; i++) {
        *gslope_ptr++ = 0.0f;
      }
    }
    DType* diff_shift = gbias.dptr_;
    for (int i = 0; i < channels_; i++) {
      diff_shift[i] = scaleShiftDiff_buf[channels_ + i];
    }
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
  dnnPrimitive_t batchNormFwdInference = NULL;
  dnnPrimitive_t batchNormFwdTraining = NULL;
  dnnPrimitive_t batchNormBwdScaleShift = NULL;
  Storage::Handle scaleShift_space;
  Storage::Handle scaleShiftDiff_space;
  dnnLayout_t layout_usr_ = NULL;
};  // class BatchNormOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKL_BATCH_NORM_INL_H_
