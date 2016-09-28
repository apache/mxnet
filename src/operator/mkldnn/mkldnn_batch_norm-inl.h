/*!
 * Copyright (c) 2016 by Contributors
 * \file mkldnn_batch_norm-inl.h
 * \brief
 * \author Chen, Xiaoming
*/

#ifndef MXNET_OPERATOR_MKLDNN_MKLDNN_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_MKLDNN_MKLDNN_BATCH_NORM_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include "dmlc/timer.h"
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "./mkldnn_cppwrapper.h"
#include "./mkldnn_memory-inl.h"
#include "mkl_service.h"

using namespace std;

namespace mxnet {
namespace op {
#if MXNET_USE_MKLDNN == 1

template <typename DType> class MKLBatchNormOp : public Operator {
 public:
  explicit MKLBatchNormOp(BatchNormParam param)
      : param_(param),
        fwd_in_data_(new MKLData<DType>()),
        fwd_out_data_(new MKLData<DType>()),
        bwd_in_diff_(new MKLData<DType>()),
        bwd_out_diff_(new MKLData<DType>()),
        batchNormFwd_(NULL),
        batchNormBwdData_(NULL),
        batchNormBwdScaleShift_(NULL) {
    init_mkl_ = false;
  }

  ~MKLBatchNormOp() {
    dnnDelete<DType>(batchNormFwd_);
    dnnDelete<DType>(batchNormBwdData_);
    dnnDelete<DType>(batchNormBwdScaleShift_);
  }

  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(aux_states.size(), 2);
    // CHECK_EQ(!(ctx.is_train && !param_.use_global_stats), false)
    //     <<"MKL does not support batch norm global stats";
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 3);
      CHECK_EQ(req.size(), 3);
    } else {
      CHECK_GE(out_data.size(), 1);
      CHECK_GE(req.size(), 1);
      CHECK_EQ(req[batchnorm::kOut], kWriteTo);
    }

    Stream<cpu> *s = ctx.get_stream<cpu>();
    if (!init_mkl_) {
      Init(ctx, s, in_data, out_data);
    }

    Tensor<cpu, 4> data;
    Tensor<cpu, 4> out;
    if (in_data[batchnorm::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[batchnorm::kData].shape_[0],
                               in_data[batchnorm::kData].shape_[1], 1, 1);
      data =
          in_data[batchnorm::kData].get_with_shape<cpu, 4, real_t>(dshape, s);
      out = out_data[batchnorm::kOut].get_with_shape<cpu, 4, real_t>(dshape, s);
    } else {
      data = in_data[batchnorm::kData].get<cpu, 4, real_t>(s);
      out = out_data[batchnorm::kOut].get<cpu, 4, real_t>(s);
    }
    Tensor<cpu, 1> slope = in_data[batchnorm::kGamma].get<cpu, 1, real_t>(s);
    Tensor<cpu, 1> bias = in_data[batchnorm::kBeta].get<cpu, 1, real_t>(s);
    Tensor<cpu, 1> moving_mean = aux_states[batchnorm::kMovingMean].get<cpu, 1, real_t>(s);
    Tensor<cpu, 1> moving_var = aux_states[batchnorm::kMovingVar].get<cpu, 1, real_t>(s);
    // yandai removes is_train
    // if (ctx.is_train && param_.fix_gamma) {
    if (param_.fix_gamma) {
      slope = 1.0f;
    }

    if (ctx.is_train && !param_.use_global_stats) {
      Tensor<cpu, 1> mean = out_data[batchnorm::kMean].get<cpu, 1, real_t>(s);
      Tensor<cpu, 1> var = out_data[batchnorm::kVar].get<cpu, 1, real_t>(s);
      CHECK(req[batchnorm::kMean] == kNullOp || req[batchnorm::kMean] == kWriteTo);
      CHECK(req[batchnorm::kVar] == kNullOp || req[batchnorm::kVar] == kWriteTo);
      // The first three steps must be enforced.
      const real_t scale = static_cast<real_t>(in_data[batchnorm::kData].shape_[1]) /
          static_cast<real_t>(in_data[batchnorm::kData].shape_.Size());
      mean = scale * sumall_except_dim<1>(data);
      var = scale * sumall_except_dim<1>(F<mshadow_op::square>(
          data - broadcast<1>(mean, data.shape_)));
      const size_t input_channels = data.size(1);
      for (int i = 0; i < input_channels; i++) {
        scaleShift_buffer_[i] = slope[i];
        scaleShift_buffer_[input_channels + i] = bias[i];
      }

      void *res_BatchNorm[dnnResourceNumber];
      res_BatchNorm[dnnResourceSrc] = data.dptr_;
      res_BatchNorm[dnnResourceWorkspace] = workspace_buffer_;
      res_BatchNorm[dnnResourceScaleShift] = scaleShift_buffer_;
      res_BatchNorm[dnnResourceDst] =
          reinterpret_cast<void *>(fwd_out_data_->set_output_ptr(out.dptr_));
      dnnError_t status = dnnExecute<DType>(batchNormFwd_, res_BatchNorm);
      CHECK_EQ(status, E_SUCCESS) << "batch norm execute failure with status "
                                  << status;

      fwd_out_data_->get_output_ptr(out.dptr_);
    } else {
      Assign(out, req[batchnorm::kOut], broadcast<1>(slope /
                                          F<mshadow_op::square_root>(moving_var + param_.eps),
                                          data.shape_) * data +
             broadcast<1>(bias - (slope * moving_mean) /
                          F<mshadow_op::square_root>(moving_var + param_.eps), data.shape_));
    }
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
    //  <<"MKL does not support batch norm global stats";
    Stream<cpu> *s = ctx.get_stream<cpu>();
    Tensor<cpu, 4> data, grad, grad_in;
    if (in_data[batchnorm::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[batchnorm::kOut].shape_[0],
                               out_grad[batchnorm::kOut].shape_[1], 1, 1);
      data =
          in_data[batchnorm::kData].get_with_shape<cpu, 4, real_t>(dshape, s);
      grad =
          out_grad[batchnorm::kOut].get_with_shape<cpu, 4, real_t>(dshape, s);
      grad_in =
          in_grad[batchnorm::kData].get_with_shape<cpu, 4, real_t>(dshape, s);
    } else {
      data = in_data[batchnorm::kData].get<cpu, 4, real_t>(s);
      grad = out_grad[batchnorm::kOut].get<cpu, 4, real_t>(s);
      grad_in = in_grad[batchnorm::kData].get<cpu, 4, real_t>(s);
    }

    Tensor<cpu, 1> gslope = in_grad[batchnorm::kGamma].get<cpu, 1, real_t>(s);
    Tensor<cpu, 1> gbias = in_grad[batchnorm::kBeta].get<cpu, 1, real_t>(s);
    Tensor<cpu, 1> mean = out_data[batchnorm::kMean].get<cpu, 1, real_t>(s);
    Tensor<cpu, 1> var = out_data[batchnorm::kVar].get<cpu, 1, real_t>(s);
    Tensor<cpu, 1> moving_mean = aux_states[batchnorm::kMovingMean].get<cpu, 1, real_t>(s);
    Tensor<cpu, 1> moving_var = aux_states[batchnorm::kMovingVar].get<cpu, 1, real_t>(s);

    moving_mean = moving_mean * param_.momentum + mean * (1 - param_.momentum);
    moving_var = moving_var * param_.momentum + var * (1 - param_.momentum);
    dnnError_t e;
    void *BatchNorm_res[dnnResourceNumber];
    BatchNorm_res[dnnResourceSrc] = data.dptr_;
    BatchNorm_res[dnnResourceWorkspace] = workspace_buffer_;
    BatchNorm_res[dnnResourceScaleShift] = scaleShift_buffer_;

    BatchNorm_res[dnnResourceDiffDst] =
        bwd_out_diff_->get_converted_prv(grad.dptr_, false);
    BatchNorm_res[dnnResourceDiffSrc] =
        bwd_in_diff_->set_output_ptr(grad_in.dptr_);
    e = dnnExecute<DType>(batchNormBwdData_, BatchNorm_res);
    CHECK_EQ(e, E_SUCCESS);
    bwd_in_diff_->get_output_ptr(grad_in.dptr_);

    void *BatchNormBwdScaleShift_res[dnnResourceNumber];
    BatchNormBwdScaleShift_res[dnnResourceSrc] = data.dptr_;
    BatchNormBwdScaleShift_res[dnnResourceWorkspace] = workspace_buffer_;
    BatchNormBwdScaleShift_res[dnnResourceDiffScaleShift] = scaleShift_buffer_;
    BatchNormBwdScaleShift_res[dnnResourceDiffDst] =
        BatchNorm_res[dnnResourceDiffDst];
    e = dnnExecute<DType>(batchNormBwdScaleShift_, BatchNormBwdScaleShift_res);
    CHECK_EQ(e, E_SUCCESS);

    const size_t input_channels = data.size(1);
    for (int i = 0; i < input_channels; i++) {
      gslope[i] = scaleShift_buffer_[i];
      gbias[i] = scaleShift_buffer_[i + input_channels];
    }
  }

 private:
  inline void Init(const OpContext &ctx, mshadow::Stream<cpu> *s,
                   const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    if (!init_mkl_) {
      init_mkl_ = true;
      dnnError_t status = E_SUCCESS;

      Tensor<cpu, 4> data;
      Tensor<cpu, 4> out;
      if (in_data[batchnorm::kData].ndim() == 2) {
        Shape<4> dshape = Shape4(in_data[batchnorm::kData].shape_[0],
                                 in_data[batchnorm::kData].shape_[1], 1, 1);
        data =
            in_data[batchnorm::kData].get_with_shape<cpu, 4, real_t>(dshape, s);
        out =
            out_data[batchnorm::kOut].get_with_shape<cpu, 4, real_t>(dshape, s);
      } else {
        data = in_data[batchnorm::kData].get<cpu, 4, real_t>(s);
        out = out_data[batchnorm::kOut].get<cpu, 4, real_t>(s);
      }
      Tensor<cpu, 1> slope = in_data[batchnorm::kGamma].get<cpu, 1, real_t>(s);
      Tensor<cpu, 1> bias = in_data[batchnorm::kBeta].get<cpu, 1, real_t>(s);
      if (ctx.is_train && param_.fix_gamma) {
        slope = 1.0f;
      }

      size_t dim = 4, sizes[4], strides[4];
      const size_t batch_size = data.size(0);
      const size_t input_channels = data.size(1);
      const size_t input_height = data.size(2);
      const size_t input_width = data.size(3);
      sizes[0] = input_width;
      sizes[1] = input_height;
      sizes[2] = input_channels;
      sizes[3] = batch_size;
      strides[0] = 1;
      strides[1] = sizes[0];
      strides[2] = sizes[0] * sizes[1];
      strides[3] = sizes[0] * sizes[1] * sizes[2];

      status = dnnLayoutCreate<DType>(&layout_usr_, dim, sizes, strides);
      CHECK_EQ(status, E_SUCCESS) << "create user layout failure with status "
                                  << status;

      fwd_in_data_->create_user_layout(dim, sizes, strides);
      fwd_out_data_->create_user_layout(dim, sizes, strides);
      bwd_in_diff_->create_user_layout(dim, sizes, strides);
      bwd_out_diff_->create_user_layout(dim, sizes, strides);

      workspace_buffer_ = NULL;
      scaleShift_buffer_ = NULL;

      status = dnnBatchNormalizationCreateForward<DType>(
          &batchNormFwd_, NULL, layout_usr_, param_.eps);
      CHECK_EQ(status, E_SUCCESS)
          << "create forward batch norm failure with status " << status;

      fwd_out_data_->create_internal_layout(batchNormFwd_, dnnResourceDst);
      bwd_in_diff_->create_internal_layout(batchNormFwd_, dnnResourceDst);
      bwd_out_diff_->create_internal_layout(batchNormFwd_, dnnResourceSrc);

      status = dnnBatchNormalizationCreateBackwardData<DType>(
          &batchNormBwdData_, NULL, bwd_in_diff_->layout_int, param_.eps);
      CHECK_EQ(status, E_SUCCESS);

      status = dnnBatchNormalizationCreateBackwardScaleShift<DType>(
          &batchNormBwdScaleShift_, NULL, bwd_in_diff_->layout_int, param_.eps);

      dnnLayout_t lt_workspace_buffer = NULL;
      status = dnnLayoutCreateFromPrimitive<DType>(
          &lt_workspace_buffer, batchNormFwd_, dnnResourceWorkspace);
      status = dnnAllocateBuffer<DType>(
          reinterpret_cast<void **>(&workspace_buffer_), lt_workspace_buffer);
      dnnLayoutDelete<DType>(lt_workspace_buffer);

      dnnLayout_t lt_scaleShift_buffer = NULL;
      status = dnnLayoutCreateFromPrimitive<DType>(
          &lt_scaleShift_buffer, batchNormFwd_, dnnResourceScaleShift);
      CHECK_EQ(status, E_SUCCESS);
      status = dnnAllocateBuffer<DType>(
          reinterpret_cast<void **>(&scaleShift_buffer_), lt_scaleShift_buffer);
      CHECK_EQ(status, E_SUCCESS);
      dnnLayoutDelete<DType>(lt_scaleShift_buffer);

      for (int i = 0; i < input_channels; i++) {
        scaleShift_buffer_[i] = slope[i];
        scaleShift_buffer_[input_channels + i] = bias[i];
      }
    }
  }

  bool init_mkl_;
  BatchNormParam param_;
  shared_ptr<MKLData<DType>> fwd_in_data_;
  shared_ptr<MKLData<DType>> fwd_out_data_;
  shared_ptr<MKLData<DType>> bwd_in_diff_;
  shared_ptr<MKLData<DType>> bwd_out_diff_;
  dnnPrimitive_t batchNormFwd_, batchNormBwdData_, batchNormBwdScaleShift_;
  DType *workspace_buffer_;
  DType *scaleShift_buffer_;
  dnnLayout_t layout_usr_;
};  // class MKLBatchNormOp
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKLDNN_MKLDNN_BATCH_NORM_INL_H_
