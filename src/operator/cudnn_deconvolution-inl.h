/*!
 * Copyright (c) 2017 by Contributors
 * \file cudnn_deconvolution-inl.h
 * \brief
 * \author Wei Wu, Leonard Lausen
*/
#ifndef MXNET_OPERATOR_CUDNN_DECONVOLUTION_INL_H_
#define MXNET_OPERATOR_CUDNN_DECONVOLUTION_INL_H_

#include <algorithm>
#include <vector>
#include <mutex>
#include <string>
#include "./deconvolution-inl.h"
#include "./cudnn_algoreg-inl.h"
#include "../common/cuda_utils.h"

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1

template<typename DType>
class CuDNNDeconvolutionOp : public Operator {
 public:
  explicit CuDNNDeconvolutionOp(DeconvolutionParam param,
                                const std::vector<TShape>& in_shape,
                                const std::vector<TShape>& out_shape,
                                const Context& ctx) {
    using namespace mshadow;
    this->param_ = param;

    // convert MB to words
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    init_cudnn_ = false;
    init_temp_size_ = false;
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;

#if CUDNN_MAJOR >= 5
    MSHADOW_LAYOUT_SWITCH(param_.layout.value(), Layout, {
        format_ = LayoutType<Layout>::kCudnnFlag;
      });
#else
    CHECK(param_.layout.value() == kNCHW || param_.layout.value() == kNCDHW)
      << "Need CuDNN > 5.0 for layout support";
#endif

    InitDescriptors(ctx, in_shape, out_shape);

    if (!param_.cudnn_tune) {
      param_.cudnn_tune = dmlc::GetEnv("MXNET_CUDNN_AUTOTUNE_DEFAULT", 1);
    }
    SelectAlgo(ctx, in_shape, out_shape);
  }

  ~CuDNNDeconvolutionOp() {
    if (init_cudnn_) {
      CHECK_EQ(cudnnDestroyTensorDescriptor(in_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(out_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(bias_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyFilterDescriptor(filter_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyConvolutionDescriptor(conv_desc_), CUDNN_STATUS_SUCCESS);
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t expected = param_.no_bias ? 2 : 3;
    DType *data_ptr = NULL;
    DType *wmat_ptr = NULL;
    DType *out_ptr = NULL;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    GetTempSize(ctx);
    Tensor<gpu, 1, DType> workspace =
      ctx.requested[deconv::kTempSpace].get_space_typed<gpu, 1, DType>(
        mshadow::Shape1(forward_workspace_), s);

    if (param_.kernel.ndim() == 2) {
      Tensor<gpu, 4, DType> data = in_data[deconv::kData].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> wmat = in_data[deconv::kWeight].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> out = out_data[deconv::kOut].get<gpu, 4, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      CHECK_EQ(wmat.CheckContiguous(), true);
      CHECK_EQ(out.CheckContiguous(), true);
      data_ptr = data.dptr_;
      wmat_ptr = wmat.dptr_;
      out_ptr = out.dptr_;
    } else {
      Tensor<gpu, 5, DType> data = in_data[deconv::kData].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> wmat = in_data[deconv::kWeight].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> out = out_data[deconv::kOut].get<gpu, 5, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      CHECK_EQ(wmat.CheckContiguous(), true);
      CHECK_EQ(out.CheckContiguous(), true);
      data_ptr = data.dptr_;
      wmat_ptr = wmat.dptr_;
      out_ptr = out.dptr_;
    }

    for (uint32_t g = 0; g < param_.num_group; ++g) {
      typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType beta  = 0.0f;
      #if CUDNN_MAJOR <= 4
      CHECK_EQ(cudnnConvolutionBackwardData_v3(s->dnn_handle_,
               &alpha,
               filter_desc_,
               wmat_ptr + weight_offset_ * g,
               in_desc_,
               data_ptr + data_offset_ * g,
               conv_desc_,
               back_algo_,
               workspace.dptr_,
               backward_workspace_byte_,
               &beta,
               out_desc_,
               out_ptr + out_offset_ * g), CUDNN_STATUS_SUCCESS);
      #elif CUDNN_MAJOR == 5
      CHECK_EQ(cudnnConvolutionBackwardData(s->dnn_handle_,
               &alpha,
               filter_desc_,
               wmat_ptr + weight_offset_ * g,
               in_desc_,
               data_ptr + data_offset_ * g,
               conv_desc_,
               back_algo_,
               workspace.dptr_,
               backward_workspace_byte_,
               &beta,
               out_desc_,
               out_ptr + out_offset_ * g), CUDNN_STATUS_SUCCESS);
      #endif
      if (!param_.no_bias) {
        beta = 1.0f;
        Tensor<gpu, 1, DType> bias = in_data[deconv::kBias].get<gpu, 1, DType>(s);
#if CUDNN_MAJOR >= 4
        CHECK_EQ(cudnnAddTensor(s->dnn_handle_,
                                &alpha,
                                bias_desc_,
                                bias.dptr_ + bias_offset_ * g,
                                &beta,
                                out_desc_,
                                out_ptr + out_offset_ * g), CUDNN_STATUS_SUCCESS);
#endif
#if CUDNN_MAJOR == 3
        CHECK_EQ(cudnnAddTensor(s->dnn_handle_,
                                CUDNN_ADD_SAME_C,
                                &alpha,
                                bias_desc_,
                                bias.dptr_ + bias_offset_ * g,
                                &beta,
                                out_desc_,
                                out_ptr + out_offset_ * g), CUDNN_STATUS_SUCCESS);
#endif
      }
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    DType *grad_ptr = NULL;
    DType *wmat_ptr = NULL;
    DType *gwmat_ptr = NULL;
    DType *data_ptr = NULL;
    DType *gdata_ptr = NULL;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    if (param_.kernel.ndim() == 2) {
      Tensor<gpu, 4, DType> grad = out_grad[deconv::kOut].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> wmat = in_data[deconv::kWeight].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> gwmat = in_grad[deconv::kWeight].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> data = in_data[deconv::kData].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> gdata = in_grad[deconv::kData].get<gpu, 4, DType>(s);
      grad_ptr = grad.dptr_;
      wmat_ptr = wmat.dptr_;
      gwmat_ptr = gwmat.dptr_;
      data_ptr = data.dptr_;
      gdata_ptr = gdata.dptr_;
    } else {
      Tensor<gpu, 5, DType> grad = out_grad[deconv::kOut].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> wmat = in_data[deconv::kWeight].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> gwmat = in_grad[deconv::kWeight].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> data = in_data[deconv::kData].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> gdata = in_grad[deconv::kData].get<gpu, 5, DType>(s);
      grad_ptr = grad.dptr_;
      wmat_ptr = wmat.dptr_;
      gwmat_ptr = gwmat.dptr_;
      data_ptr = data.dptr_;
      gdata_ptr = gdata.dptr_;
    }
    CHECK_NE(req[deconv::kWeight], kWriteInplace);
    CHECK_NE(req[deconv::kBias], kWriteInplace);
    CHECK_NE(req[deconv::kData], kWriteInplace);
    Tensor<gpu, 1, DType> workspace =
        ctx.requested[deconv::kTempSpace].get_space_typed<gpu, 1, DType>(
                                 mshadow::Shape1(backward_workspace_), s);
    for (uint32_t g = 0; g < param_.num_group; ++g) {
      typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType bias_beta =
        req[deconv::kBias] == kAddTo ? 1.0f : 0.0f;
      typename DataType<DType>::ScaleType data_beta =
        req[deconv::kData] == kAddTo ? 1.0f : 0.0f;
      typename DataType<DType>::ScaleType weight_beta =
        req[deconv::kWeight] == kAddTo ? 1.0f : 0.0f;
      if (!param_.no_bias) {
        CHECK_NE(req[deconv::kBias], kNullOp);
        Tensor<gpu, 1, DType> gbias = in_grad[deconv::kBias].get<gpu, 1, DType>(s);
        CHECK_EQ(cudnnConvolutionBackwardBias(s->dnn_handle_,
                                              &alpha,
                                              out_desc_,
                                              grad_ptr + out_offset_ * g,
                                              &bias_beta,
                                              bias_desc_,
                                              gbias.dptr_ + bias_offset_ * g),
                 CUDNN_STATUS_SUCCESS);
      }
      if (req[deconv::kWeight] != kNullOp) {
        #if CUDNN_MAJOR <= 4
        CHECK_EQ(cudnnConvolutionBackwardFilter_v3(s->dnn_handle_,
                 &alpha,
                 out_desc_,
                 grad_ptr + out_offset_ * g,
                 in_desc_,
                 data_ptr + data_offset_ * g,
                 conv_desc_,
                 back_algo_w_,
                 workspace.dptr_,
                 backward_workspace_byte_,
                 &weight_beta,
                 filter_desc_,
                 gwmat_ptr + weight_offset_ * g), CUDNN_STATUS_SUCCESS);
        #elif CUDNN_MAJOR == 5
        CHECK_EQ(cudnnConvolutionBackwardFilter(s->dnn_handle_,
                 &alpha,
                 out_desc_,
                 grad_ptr + out_offset_ * g,
                 in_desc_,
                 data_ptr + data_offset_ * g,
                 conv_desc_,
                 back_algo_w_,
                 workspace.dptr_,
                 backward_workspace_byte_,
                 &weight_beta,
                 filter_desc_,
                 gwmat_ptr + weight_offset_ * g), CUDNN_STATUS_SUCCESS);
        #endif
      }
      if (req[deconv::kData] != kNullOp) {
        CHECK_EQ(cudnnConvolutionForward(s->dnn_handle_,
                                         &alpha,
                                         out_desc_,
                                         grad_ptr + out_offset_ * g,
                                         filter_desc_,
                                         wmat_ptr + weight_offset_ * g,
                                         conv_desc_,
                                         algo_,
                                         workspace.dptr_,
                                         forward_workspace_byte_,
                                         &data_beta,
                                         in_desc_,
                                         gdata_ptr + data_offset_ * g), CUDNN_STATUS_SUCCESS);
      }
    }
  }

 private:
  inline void InitDescriptors(const Context& ctx,
                              const std::vector<TShape> &in_shape,
                              const std::vector<TShape> &out_shape) {
    using namespace mshadow;
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_shape.size(), expected);
    CHECK_EQ(out_shape.size(), 1U);
    CHECK_EQ(cudnnCreateTensorDescriptor(&in_desc_), CUDNN_STATUS_SUCCESS);
    CHECK_EQ(cudnnCreateTensorDescriptor(&out_desc_), CUDNN_STATUS_SUCCESS);
    CHECK_EQ(cudnnCreateTensorDescriptor(&bias_desc_), CUDNN_STATUS_SUCCESS);
    CHECK_EQ(cudnnCreateFilterDescriptor(&filter_desc_), CUDNN_STATUS_SUCCESS);
    CHECK_EQ(cudnnCreateConvolutionDescriptor(&conv_desc_), CUDNN_STATUS_SUCCESS);

    TShape dshape = in_shape[deconv::kData];
    TShape wshape = in_shape[deconv::kWeight];
    TShape oshape = out_shape[deconv::kOut];
    TShape dstride, ostride;
    wshape[0] /= param_.num_group;

    if (param_.kernel.ndim() == 2) {
      // 2d conv
      index_t o_pad[2];
      index_t o_adj[2];
      param_.InferPad(dshape, o_pad, o_adj);

      CHECK_EQ(cudnnSetConvolution2dDescriptor(conv_desc_,
                                               o_pad[0],
                                               o_pad[1],
                                               param_.stride[0],
                                               param_.stride[1],
                                               1,
                                               1,
                                               CUDNN_CROSS_CORRELATION), CUDNN_STATUS_SUCCESS);

      #if CUDNN_MAJOR >= 5
      wshape = ConvertLayout(wshape.get<4>(), param_.layout.value(), kNCHW);
      CHECK_EQ(cudnnSetFilter4dDescriptor(filter_desc_,
                                          dtype_,
                                          format_,
                                          wshape[0],
                                          wshape[1],
                                          wshape[2],
                                          wshape[3]), CUDNN_STATUS_SUCCESS);
      #else
      CHECK_EQ(param_.layout.value(), kNCHW) << "CuDNN V4 only support NCHW layout";
      CHECK_EQ(cudnnSetFilter4dDescriptor(filter_desc_,
                                          dtype_,
                                          wshape[0],
                                          wshape[1],
                                          wshape[2],
                                          wshape[3]), CUDNN_STATUS_SUCCESS);
      #endif

      dstride = ConvertLayout(Shape4(dshape[1] * dshape[2] * dshape[3],
                                     dshape[2] * dshape[3],
                                     dshape[3],
                                     1),
                              param_.layout.value(), kNCHW);
      dshape = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW);

      ostride = ConvertLayout(Shape4(oshape[1] * oshape[2] * oshape[3],
                                     oshape[2] * oshape[3],
                                     oshape[3],
                                     1),
                              param_.layout.value(), kNCHW);
      oshape = ConvertLayout(oshape.get<4>(), param_.layout.value(), kNCHW);
    } else if (param_.kernel.ndim() == 3) {
      // 3d conv
      std::vector<int> upscale_vec = {1, 1, 1};

      index_t o_pad[3];
      index_t o_adj[3];
      param_.InferPad(dshape, o_pad, o_adj);

      #if CUDNN_MAJOR >= 5
      CHECK_EQ(param_.layout.value(), kNCDHW) << "CuDNN only support 3D conv with NCDHW layout";
      CHECK_EQ(cudnnSetFilterNdDescriptor(filter_desc_,
                                          dtype_,
                                          CUDNN_TENSOR_NCHW,
                                          static_cast<int>(wshape.ndim()),
                                          reinterpret_cast<int*>(&wshape[0])),
               CUDNN_STATUS_SUCCESS);
      #else
      LOG(FATAL) << "Only support CUDNN V5 for 3D convolution";
      #endif
      CHECK_EQ(cudnnSetConvolutionNdDescriptor(conv_desc_,
                                               3,
                                               reinterpret_cast<int*>(&o_pad[0]),
                                               reinterpret_cast<int*>(&param_.stride[0]),
                                               &upscale_vec[0],
                                               CUDNN_CROSS_CORRELATION,
                                               dtype_), CUDNN_STATUS_SUCCESS);

      dstride = ConvertLayout(Shape5(dshape[1] * dshape[2] * dshape[3] * dshape[4],
                                     dshape[2] * dshape[3] * dshape[4],
                                     dshape[3] * dshape[4],
                                     dshape[4],
                                     1),
                              param_.layout.value(), kNCDHW);
      dshape = ConvertLayout(dshape.get<5>(), param_.layout.value(), kNCDHW);

      ostride = ConvertLayout(Shape5(oshape[1] * oshape[2] * oshape[3] * oshape[4],
                                     oshape[2] * oshape[3] * oshape[4],
                                     oshape[3] * oshape[4],
                                     oshape[4],
                                     1),
                              param_.layout.value(), kNCDHW);
      oshape = ConvertLayout(oshape.get<5>(), param_.layout.value(), kNCDHW);
    }
    dshape[1] /= param_.num_group;
    oshape[1] /= param_.num_group;
    weight_offset_ = wshape.Size();
    data_offset_ = dstride[1] * dshape[1];
    out_offset_ = ostride[1] * oshape[1];

    CHECK_EQ(cudnnSetTensorNdDescriptor(in_desc_,
                                        dtype_,
                                        static_cast<int>(dshape.ndim()),
                                        reinterpret_cast<int*>(&dshape[0]),
                                        reinterpret_cast<int*>(&dstride[0])),
             CUDNN_STATUS_SUCCESS);

    CHECK_EQ(cudnnSetTensorNdDescriptor(out_desc_,
                                        dtype_,
                                        static_cast<int>(oshape.ndim()),
                                        reinterpret_cast<int*>(&oshape[0]),
                                        reinterpret_cast<int*>(&ostride[0])),
             CUDNN_STATUS_SUCCESS);

    if (!param_.no_bias) {
      TShape bias = in_shape[deconv::kBias];
      bias_offset_ = bias[0] / param_.num_group;
      std::vector<int> bias_shape = {1,
                                     static_cast<int>(bias[0] / param_.num_group),
                                     1, 1};
      std::vector<int> bias_stride = {static_cast<int>(bias_offset_), 1, 1, 1};
      if (param_.kernel.ndim() == 3) {
        bias_shape.push_back(1);
        bias_stride.push_back(1);
      }
      CHECK_EQ(cudnnSetTensorNdDescriptor(bias_desc_,
                                          dtype_,
                                          static_cast<int>(bias_shape.size()),
                                          &bias_shape[0],
                                          &bias_stride[0]), CUDNN_STATUS_SUCCESS);
    }
    init_cudnn_ = true;
  }

  void SelectAlgo(const Context& ctx,
                  const std::vector<TShape>& in_shape,
                  const std::vector<TShape>& out_shape) {
    std::string key = CuDNNAlgoReg::Get()->GetKey(param_, in_shape, out_shape);
    if (CuDNNAlgoReg::Get()->Find(key, &algo_, &back_algo_, &back_algo_w_)) return;

    Engine::VarHandle var = Engine::Get()->NewVariable();
    Engine::Get()->PushSync([=](RunContext rctx) {
      mshadow::Stream<gpu> *s = rctx.get_stream<gpu>();
      CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
      size_t workspace_byte = static_cast<size_t>(param_.workspace * sizeof(DType));
      if (!param_.cudnn_tune.value()) {
        CHECK_EQ(cudnnGetConvolutionForwardAlgorithm(s->dnn_handle_,
                 out_desc_,
                 filter_desc_,
                 conv_desc_,
                 in_desc_,
                 CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                 workspace_byte,
                 &(this->algo_)), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnGetConvolutionBackwardFilterAlgorithm(s->dnn_handle_,
                 out_desc_,
                 in_desc_,
                 conv_desc_,
                 filter_desc_,
                 CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                 workspace_byte,
                 &(this->back_algo_w_)), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnGetConvolutionBackwardDataAlgorithm(s->dnn_handle_,
                 filter_desc_,
                 in_desc_,
                 conv_desc_,
                 out_desc_,
                 CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                 workspace_byte,
                 &(this->back_algo_)), CUDNN_STATUS_SUCCESS);
      } else {
        const int kMaxAlgos = 10;
        int nalgo = kMaxAlgos;
        int i;

        cudnnConvolutionFwdAlgoPerf_t fwd_algo[kMaxAlgos];
        CHECK_EQ(cudnnFindConvolutionForwardAlgorithm(s->dnn_handle_,
                 out_desc_,
                 filter_desc_,
                 conv_desc_,
                 in_desc_,
                 kMaxAlgos,
                 &nalgo,
                 fwd_algo), CUDNN_STATUS_SUCCESS);
        i = 0;
        while (i < nalgo
               && (fwd_algo[i].status != CUDNN_STATUS_SUCCESS
               || (param_.cudnn_tune.value() == deconv::kLimited
               && fwd_algo[i].memory > workspace_byte))) ++i;
        if (i == nalgo) {
          LOG(FATAL) << "Failed to find an convolution algorithm.";
        } else {
          this->algo_ = fwd_algo[i].algo;
        }

        cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_algo[kMaxAlgos];
        CHECK_EQ(cudnnFindConvolutionBackwardFilterAlgorithm(s->dnn_handle_,
                 out_desc_,
                 in_desc_,
                 conv_desc_,
                 filter_desc_,
                 kMaxAlgos,
                 &nalgo,
                 bwd_filter_algo), CUDNN_STATUS_SUCCESS);
        i = 0;
        while (i < nalgo
               && (bwd_filter_algo[i].status != CUDNN_STATUS_SUCCESS
               || (param_.cudnn_tune.value() == deconv::kLimited
               && bwd_filter_algo[i].memory > workspace_byte))) ++i;
        if (i == nalgo) {
          LOG(FATAL) << "Failed to find an convolution algorithm.";
        } else {
          this->back_algo_w_ = bwd_filter_algo[i].algo;
        }

        cudnnConvolutionBwdDataAlgoPerf_t bwd_data_algo[kMaxAlgos];
        CHECK_EQ(cudnnFindConvolutionBackwardDataAlgorithm(s->dnn_handle_,
                 filter_desc_,
                 in_desc_,
                 conv_desc_,
                 out_desc_,
                 kMaxAlgos,
                 &nalgo,
                 bwd_data_algo), CUDNN_STATUS_SUCCESS);
        i = 0;
        while (i < nalgo
               && (bwd_data_algo[i].status != CUDNN_STATUS_SUCCESS
               || (param_.cudnn_tune.value() == deconv::kLimited
               && bwd_data_algo[i].memory > workspace_byte))) ++i;
        if (i == nalgo) {
          LOG(FATAL) << "Failed to find an convolution algorithm.";
        } else {
          this->back_algo_ = bwd_data_algo[i].algo;
        }
        CuDNNAlgoReg::Get()->Register(key, this->algo_, this->back_algo_, this->back_algo_w_);
      }
    }, ctx, {}, {var});
    Engine::Get()->WaitForVar(var);
    Engine::Get()->DeleteVariable([](RunContext s) {}, ctx, var);
  }

  void GetTempSize(const OpContext& ctx) {
    if (init_temp_size_) return;
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    size_t back_size = 0, back_size_w = 0;
    CHECK_EQ(cudnnGetConvolutionBackwardDataWorkspaceSize(s->dnn_handle_,
               filter_desc_,
               in_desc_,
               conv_desc_,
               out_desc_,
               back_algo_,
               &back_size), CUDNN_STATUS_SUCCESS);
    CHECK_EQ(cudnnGetConvolutionBackwardFilterWorkspaceSize(s->dnn_handle_,
               out_desc_,
               in_desc_,
               conv_desc_,
               filter_desc_,
               back_algo_w_,
               &back_size_w), CUDNN_STATUS_SUCCESS);
    backward_workspace_byte_ = std::max(back_size, back_size_w);
    CHECK_EQ(cudnnGetConvolutionForwardWorkspaceSize(s->dnn_handle_,
               out_desc_,
               filter_desc_,
               conv_desc_,
               in_desc_,
               algo_,
               &forward_workspace_byte_), CUDNN_STATUS_SUCCESS);

    forward_workspace_ = forward_workspace_byte_ / sizeof(DType) + 1;
    backward_workspace_ = backward_workspace_byte_ / sizeof(DType) + 1;
    init_temp_size_ = true;
  }

  bool init_cudnn_;
  bool init_temp_size_;
  size_t forward_workspace_;
  size_t backward_workspace_;
  size_t forward_workspace_byte_;
  size_t backward_workspace_byte_;
  size_t data_offset_;
  size_t out_offset_;
  size_t weight_offset_;
  size_t bias_offset_;
  cudnnDataType_t dtype_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnConvolutionFwdAlgo_t algo_;
  cudnnConvolutionBwdDataAlgo_t back_algo_;
  cudnnConvolutionBwdFilterAlgo_t back_algo_w_;
  cudnnTensorFormat_t format_;
  DeconvolutionParam param_;
};
#endif  // CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CUDNN_DECONVOLUTION_INL_H_
