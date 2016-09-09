/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_convolution-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_CUDNN_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_CUDNN_CONVOLUTION_INL_H_

#include <algorithm>
#include <vector>
#include "./convolution-inl.h"
#include "../common/cuda_utils.h"

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1
void TuneCudnnConvolution(ConvolutionParam param,
                          std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape,
                          Context ctx,
                          cudnnDataType_t dtype,
                          cudnnConvolutionFwdAlgo_t *algo,
                          cudnnConvolutionBwdDataAlgo_t *back_algo,
                          cudnnConvolutionBwdFilterAlgo_t *back_algo_w);

template<typename DType>
class CuDNNConvolutionOp : public Operator {
 public:
  explicit CuDNNConvolutionOp(ConvolutionParam param,
                              std::vector<TShape> *in_shape,
                              std::vector<TShape> *out_shape,
                              Context ctx) {
    using namespace mshadow;
    this->param_ = param;
    // convert MB to words
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    init_cudnn_ = false;
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;

    if (param.cudnn_tune != conv::kOff) {
      TuneCudnnConvolution(param, in_shape, out_shape, ctx, dtype_,
                           &algo_, &back_algo_, &back_algo_w_);
    }
  }

  ~CuDNNConvolutionOp() {
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
    CHECK_EQ(out_data.size(), 1);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    if (!init_cudnn_) {
      Init(s, in_data, out_data);
    }
    Tensor<gpu, 1, DType> workspace =
        ctx.requested[conv::kTempSpace].get_space_typed<gpu, 1, DType>(
                                 mshadow::Shape1(forward_workspace_), s);

    if (param_.kernel.ndim() == 2) {
      Tensor<gpu, 4, DType> data = in_data[conv::kData].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> wmat = in_data[conv::kWeight].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> out = out_data[conv::kOut].get<gpu, 4, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      CHECK_EQ(wmat.CheckContiguous(), true);
      CHECK_EQ(out.CheckContiguous(), true);
      data_ptr = data.dptr_;
      wmat_ptr = wmat.dptr_;
      out_ptr = out.dptr_;
    } else {
      Tensor<gpu, 5, DType> data = in_data[conv::kData].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> wmat = in_data[conv::kWeight].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> out = out_data[conv::kOut].get<gpu, 5, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      CHECK_EQ(wmat.CheckContiguous(), true);
      CHECK_EQ(out.CheckContiguous(), true);
      data_ptr = data.dptr_;
      wmat_ptr = wmat.dptr_;
      out_ptr = out.dptr_;
    }
    for (uint32_t g = 0; g < param_.num_group; ++g) {
      typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType beta = 0.0f;
      CHECK_EQ(cudnnConvolutionForward(s->dnn_handle_,
                                       &alpha,
                                       in_desc_,
                                       data_ptr + data_offset_ * g,
                                       filter_desc_,
                                       wmat_ptr + weight_offset_ * g,
                                       conv_desc_,
                                       algo_,
                                       workspace.dptr_,
                                       forward_workspace_byte_,
                                       &beta,
                                       out_desc_,
                                       out_ptr + out_offset_ * g), CUDNN_STATUS_SUCCESS);
      if (!param_.no_bias) {
        beta = 1.0f;
        Tensor<gpu, 1, DType> bias = in_data[conv::kBias].get<gpu, 1, DType>(s);
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
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    if (param_.kernel.ndim() == 2) {
      Tensor<gpu, 4, DType> grad = out_grad[conv::kOut].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> wmat = in_data[conv::kWeight].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> gwmat = in_grad[conv::kWeight].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> data = in_data[conv::kData].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> gdata = in_grad[conv::kData].get<gpu, 4, DType>(s);
      grad_ptr = grad.dptr_;
      wmat_ptr = wmat.dptr_;
      gwmat_ptr = gwmat.dptr_;
      data_ptr = data.dptr_;
      gdata_ptr = gdata.dptr_;
    } else {
      Tensor<gpu, 5, DType> grad = out_grad[conv::kOut].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> wmat = in_data[conv::kWeight].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> gwmat = in_grad[conv::kWeight].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> data = in_data[conv::kData].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> gdata = in_grad[conv::kData].get<gpu, 5, DType>(s);
      grad_ptr = grad.dptr_;
      wmat_ptr = wmat.dptr_;
      gwmat_ptr = gwmat.dptr_;
      data_ptr = data.dptr_;
      gdata_ptr = gdata.dptr_;
    }
    Tensor<gpu, 1, DType> workspace =
      ctx.requested[conv::kTempSpace].get_space_typed<gpu, 1, DType>(
      mshadow::Shape1(backward_workspace_), s);
    for (uint32_t g = 0; g < param_.num_group; ++g) {
      typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType beta = 0.0f;
      typename DataType<DType>::ScaleType beta_add = 1.0f;
      if (!param_.no_bias) {
        Tensor<gpu, 1, DType> gbias = in_grad[conv::kBias].get<gpu, 1, DType>(s);
        CHECK_EQ(cudnnConvolutionBackwardBias(s->dnn_handle_,
                                              &alpha,
                                              out_desc_,
                                              grad_ptr + out_offset_ * g,
                                              req[conv::kBias] == kWriteTo ? &beta : &beta_add,
                                              bias_desc_,
                                              gbias.dptr_ + bias_offset_ * g),
                 CUDNN_STATUS_SUCCESS);
      }
      #if CUDNN_MAJOR <= 4
      CHECK_EQ(cudnnConvolutionBackwardFilter_v3(s->dnn_handle_,
               &alpha,
               in_desc_,
               data_ptr + data_offset_ * g,
               out_desc_,
               grad_ptr + out_offset_ * g,
               conv_desc_,
               back_algo_w_,
               workspace.dptr_,
               backward_workspace_byte_,
               req[conv::kWeight] == kWriteTo? &beta : &beta_add,
               filter_desc_,
               gwmat_ptr + weight_offset_ * g), CUDNN_STATUS_SUCCESS);
      #elif CUDNN_MAJOR == 5
      CUDNN_CALL(cudnnConvolutionBackwardFilter(s->dnn_handle_,
               &alpha,
               in_desc_,
               data_ptr + data_offset_ * g,
               out_desc_,
               grad_ptr + out_offset_ * g,
               conv_desc_,
               back_algo_w_,
               workspace.dptr_,
               backward_workspace_byte_,
               req[conv::kWeight] == kWriteTo? &beta : &beta_add,
               filter_desc_,
               gwmat_ptr + weight_offset_ * g));
      #endif
      #if CUDNN_MAJOR <= 4
      CHECK_EQ(cudnnConvolutionBackwardData_v3(s->dnn_handle_,
               &alpha,
               filter_desc_,
               wmat_ptr + weight_offset_ * g,
               out_desc_,
               grad_ptr + out_offset_ * g,
               conv_desc_,
               back_algo_,
               workspace.dptr_,
               backward_workspace_byte_,
               &beta,
               in_desc_,
               gdata_ptr + data_offset_ * g), CUDNN_STATUS_SUCCESS);
      #elif CUDNN_MAJOR == 5
      CHECK_EQ(cudnnConvolutionBackwardData(s->dnn_handle_,
               &alpha,
               filter_desc_,
               wmat_ptr + weight_offset_ * g,
               out_desc_,
               grad_ptr + out_offset_ * g,
               conv_desc_,
               back_algo_,
               workspace.dptr_,
               backward_workspace_byte_,
               &beta,
               in_desc_,
               gdata_ptr + data_offset_ * g), CUDNN_STATUS_SUCCESS);
      #endif
    }
  }

 private:
  inline void Init(mshadow::Stream<gpu> *s,
                   const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    size_t expected = param_.no_bias ? 2 : 3;
    #if CUDNN_MAJOR == 5
    format_ = CUDNN_TENSOR_NCHW;
    #endif
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    if (!init_cudnn_) {
      init_cudnn_ = true;
      size_t workspace_byte = static_cast<size_t>(param_.workspace * sizeof(DType));
      size_t back_size = 0;
      size_t back_size_w = 0;
      CHECK_EQ(cudnnCreateTensorDescriptor(&in_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&out_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&bias_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateFilterDescriptor(&filter_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateConvolutionDescriptor(&conv_desc_), CUDNN_STATUS_SUCCESS);
      if (param_.kernel.ndim() == 2) {
        // 2d conv
        Tensor<gpu, 4, DType> data = in_data[conv::kData].get<gpu, 4, DType>(s);
        Tensor<gpu, 4, DType> out = out_data[conv::kOut].get<gpu, 4, DType>(s);
        data_offset_ = data.shape_[1] / param_.num_group * data.shape_[2] * data.shape_[3];
        out_offset_ = out.shape_[1] /param_.num_group * out.shape_[2] * out.shape_[3];
        weight_offset_ = param_.num_filter / param_.num_group * data.shape_[1] / param_.num_group
                        * param_.kernel[0] * param_.kernel[1];
        #if CUDNN_MAJOR == 5
        CHECK_EQ(cudnnSetFilter4dDescriptor(filter_desc_,
                                            dtype_,
                                            format_,
                                            param_.num_filter / param_.num_group,
                                            data.shape_[1] / param_.num_group,
                                            param_.kernel[0],
                                            param_.kernel[1]), CUDNN_STATUS_SUCCESS);
        #else
        CHECK_EQ(cudnnSetFilter4dDescriptor(filter_desc_,
                                            dtype_,
                                            param_.num_filter / param_.num_group,
                                            data.shape_[1] / param_.num_group,
                                            param_.kernel[0],
                                            param_.kernel[1]), CUDNN_STATUS_SUCCESS);
        #endif
        CHECK_EQ(cudnnSetConvolution2dDescriptor(conv_desc_,
                                                param_.pad[0],
                                                param_.pad[1],
                                                param_.stride[0],
                                                param_.stride[1],
                                                1,
                                                1,
                                                CUDNN_CROSS_CORRELATION), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnSetTensor4dDescriptorEx(in_desc_,
                                              dtype_,
                                              data.shape_[0],
                                              data.shape_[1] / param_.num_group,
                                              data.shape_[2],
                                              data.shape_[3],
                                              data.shape_[1] * data.shape_[2] * data.shape_[3],
                                              data.shape_[2] * data.shape_[3],
                                              data.shape_[3],
                                              1), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnSetTensor4dDescriptorEx(out_desc_,
                                              dtype_,
                                              out.shape_[0],
                                              out.shape_[1] / param_.num_group,
                                              out.shape_[2],
                                              out.shape_[3],
                                              out.shape_[1] * out.shape_[2] * out.shape_[3],
                                              out.shape_[2] * out.shape_[3],
                                              out.shape_[3],
                                              1), CUDNN_STATUS_SUCCESS);
      } else if (param_.kernel.ndim() == 3) {
        // 3d conv
        Tensor<gpu, 5, DType> data = in_data[conv::kData].get<gpu, 5, DType>(s);
        Tensor<gpu, 5, DType> out = out_data[conv::kOut].get<gpu, 5, DType>(s);
        data_offset_ = data.shape_[1] / param_.num_group * data.shape_[2] * \
                                                           data.shape_[3] * \
                                                           data.shape_[4];
        out_offset_ = out.shape_[1] / param_.num_group * out.shape_[2] * \
                                                         out.shape_[3] * \
                                                         out.shape_[4];
        weight_offset_ = param_.num_filter / param_.num_group * data.shape_[1] / param_.num_group
                        * param_.kernel[0] * param_.kernel[1] * param_.kernel[2];
        std::vector<int> filter_vec = {static_cast<int>(param_.num_filter / param_.num_group),
                                       static_cast<int>(data.shape_[1] / param_.num_group),
                                       static_cast<int>(param_.kernel[0]),
                                       static_cast<int>(param_.kernel[1]),
                                       static_cast<int>(param_.kernel[2])};

        std::vector<int> pad_vec = {static_cast<int>(param_.pad[0]),
                                    static_cast<int>(param_.pad[1]),
                                    static_cast<int>(param_.pad[2])};

        std::vector<int> stride_vec = {static_cast<int>(param_.stride[0]),
                                       static_cast<int>(param_.stride[1]),
                                       static_cast<int>(param_.stride[2])};

        std::vector<int> upscale_vec = {1, 1, 1};

        std::vector<int> ishape = {static_cast<int>(data.shape_[0]),
                                   static_cast<int>(data.shape_[1]),
                                   static_cast<int>(data.shape_[2]),
                                   static_cast<int>(data.shape_[3]),
                                   static_cast<int>(data.shape_[4])};

        std::vector<int> istride = {static_cast<int>(ishape[1] * ishape[2] * ishape[3] * ishape[4]),
                                    static_cast<int>(ishape[2] * ishape[3] * ishape[4]),
                                    static_cast<int>(ishape[3] * ishape[4]),
                                    static_cast<int>(ishape[4]),
                                    1};

        std::vector<int> oshape = {static_cast<int>(out.shape_[0]),
                                   static_cast<int>(out.shape_[1]),
                                   static_cast<int>(out.shape_[2]),
                                   static_cast<int>(out.shape_[3]),
                                   static_cast<int>(out.shape_[4])};

        std::vector<int> ostride = {static_cast<int>(oshape[1] * oshape[2] * oshape[3] * oshape[4]),
                                    static_cast<int>(oshape[2] * oshape[3] * oshape[4]),
                                    static_cast<int>(oshape[3] * oshape[4]),
                                    static_cast<int>(oshape[4]),
                                    1};

        #if CUDNN_MAJOR == 5
        CHECK_EQ(cudnnSetFilterNdDescriptor(filter_desc_,
                                            dtype_,
                                            format_,
                                            static_cast<int>(filter_vec.size()),
                                            &filter_vec[0]), CUDNN_STATUS_SUCCESS);
        #else
        LOG(FATAL) << "Only support CUDNN V5 for 3D convolution";
        #endif
        CHECK_EQ(cudnnSetConvolutionNdDescriptor(conv_desc_,
                                                 3,
                                                 &pad_vec[0],
                                                 &stride_vec[0],
                                                 &upscale_vec[0],
                                                 CUDNN_CROSS_CORRELATION,
                                                 dtype_), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnSetTensorNdDescriptor(in_desc_,
                                              dtype_,
                                              static_cast<int>(ishape.size()),
                                              &ishape[0],
                                              &istride[0]), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnSetTensorNdDescriptor(out_desc_,
                                              dtype_,
                                              static_cast<int>(oshape.size()),
                                              &oshape[0],
                                              &ostride[0]), CUDNN_STATUS_SUCCESS);
      }
      if (!param_.no_bias) {
        Tensor<gpu, 1, DType> bias = in_data[conv::kBias].get<gpu, 1, DType>(s);
        bias_offset_ = bias.shape_[0] / param_.num_group;
        std::vector<int> bias_shape = {1,
                                       static_cast<int>(bias.shape_[0] / param_.num_group),
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

      if (!param_.cudnn_tune) {
        CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
        CHECK_EQ(cudnnGetConvolutionForwardAlgorithm(s->dnn_handle_,
                 in_desc_,
                 filter_desc_,
                 conv_desc_,
                 out_desc_,
                 CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                 workspace_byte,
                 &algo_), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnGetConvolutionBackwardFilterAlgorithm(s->dnn_handle_,
                 in_desc_,
                 out_desc_,
                 conv_desc_,
                 filter_desc_,
                 CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                 workspace_byte,
                 &back_algo_w_), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnGetConvolutionBackwardDataAlgorithm(s->dnn_handle_,
                 filter_desc_,
                 out_desc_,
                 conv_desc_,
                 in_desc_,
                 CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                 workspace_byte,
                 &back_algo_), CUDNN_STATUS_SUCCESS);
      }

      CHECK_EQ(cudnnGetConvolutionBackwardDataWorkspaceSize(s->dnn_handle_,
               filter_desc_,
               out_desc_,
               conv_desc_,
               in_desc_,
               back_algo_,
               &back_size), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnGetConvolutionBackwardFilterWorkspaceSize(s->dnn_handle_,
               in_desc_,
               out_desc_,
               conv_desc_,
               filter_desc_,
               back_algo_w_,
               &back_size_w), CUDNN_STATUS_SUCCESS);
      backward_workspace_byte_ = std::max(back_size, back_size_w);
      CHECK_EQ(cudnnGetConvolutionForwardWorkspaceSize(s->dnn_handle_,
               in_desc_,
               filter_desc_,
               conv_desc_,
               out_desc_,
               algo_,
               &forward_workspace_byte_), CUDNN_STATUS_SUCCESS);

      forward_workspace_ = forward_workspace_byte_ / sizeof(DType) + 1;
      backward_workspace_ = backward_workspace_byte_ / sizeof(DType) + 1;
      // ugly fix CUDNN algorithm selection
      // safe to remove after CuDNN fix 3D conv selection
      // if (param_.kernel.ndim() == 3) {
      //   back_algo_w_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
      // }
    }
  }

  bool init_cudnn_;
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
  #if CUDNN_MAJOR == 5
  cudnnTensorFormat_t format_;
  #endif
  ConvolutionParam param_;
};
#endif  // __CUDACC__ && CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CUDNN_CONVOLUTION_INL_H_
