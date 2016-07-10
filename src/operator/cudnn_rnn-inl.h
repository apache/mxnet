/*!
 * Copyright (c) 2016 by Contributors
 * \file cudnn_spatial_transformer-inl.h
 * \brief
 * \author Sebastian Bodenstein
*/
#ifndef MXNET_OPERATOR_CUDNN_RNN_INL_H_
#define MXNET_OPERATOR_CUDNN_RNN_INL_H_

#include <algorithm>
#include <vector>
#include "./rnn-inl.h"
namespace mxnet {
namespace op {
#if defined(__CUDACC__) && MXNET_USE_CUDNN == 1 && CUDNN_MAJOR == 5
template<typename DType>
class CuDNNRNNOp : public Operator {
 public:
  explicit CuDNNRNNOp(RNNParam param) {
    this->param_ = param;
    init_cudnn_ = false;
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
    // RNN Mode
    switch (param_.mode) {
      case rnn_enum::kRnnRelu:
        rnn_mode_ = CUDNN_RNN_RELU;
        break;
      case rnn_enum::kRnnTanh:
        rnn_mode_ = CUDNN_RNN_TANH;
        break;
      case rnn_enum::kLstm:
        rnn_mode_ = CUDNN_LSTM;
        break;
      case rnn_enum::kGru:
        rnn_mode_ = CUDNN_GRU;
        break;
      default:
        LOG(FATAL) << "Not implmented";
    }
    // RNN Direction
    switch (param_.direction) {
      case rnn_enum::kUnidirectional:
        rnn_direction_ = CUDNN_UNIDIRECTIONAL;
        break;
      case rnn_enum::kBidirectional:
        rnn_direction_ = CUDNN_BIDIRECTIONAL;
        break;
      default:
        LOG(FATAL) << "Not implmented";
    }
  }
  // ~CuDNNRNNOp() {
  //   if (init_cudnn_) {
  //     CHECK_EQ(cudnnDestroyRNNDescriptor(rnn_desc_), CUDNN_STATUS_SUCCESS);
  //     // CHECK_EQ(cudnnDestroyTensorDescriptor(rnn_desc_), CUDNN_STATUS_SUCCESS);
  //     // CHECK_EQ(cudnnDestroyTensorDescriptor(_desc_), CUDNN_STATUS_SUCCESS);
  //   }
  // }
 
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
 //    CHECK_EQ(in_data.size(), 2);
 //    CHECK_EQ(out_data.size(), 3);
 //    Stream<gpu> *s = ctx.get_stream<gpu>();
 //    Tensor<gpu, 4, DType> data = in_data[st::kData].get<gpu, 4, DType>(s);
 //    Tensor<gpu, 4, DType> out = out_data[st::kOut].get<gpu, 4, DType>(s);
 //    Shape<3> loc_shape = Shape3(data.size(0), 2, 3);
 //    Shape<4> grid_shape = Shape4(out.size(0), out.size(2), out.size(3), 2);
 //    Tensor<gpu, 3, DType> loc = in_data[st::kLoc].get_with_shape<gpu, 3, DType>(loc_shape, s);
 //    Tensor<gpu, 4, DType> grid = out_data[st::kGridSrc]
 //                                .get_with_shape<gpu, 4, DType>(grid_shape, s);
 //    if (!init_cudnn_) {
 //     Init(s, in_data, out_data);
 //    }
 //    CHECK_EQ(data.CheckContiguous(), true);
 //    CHECK_EQ(out.CheckContiguous(), true);
 //    typename DataType<DType>::ScaleType alpha = 1.0f;
 //    typename DataType<DType>::ScaleType beta = 0.0f;
 //    if (param_.transform_type == st::kAffine) {
 //      CHECK_EQ(cudnnSpatialTfGridGeneratorForward(s->dnn_handle_,
 //                                                  st_desc_,
 //                                                  loc.dptr_,
 //                                                  grid.dptr_/*output*/), CUDNN_STATUS_SUCCESS);
 //    }
 //    CHECK_EQ(cudnnSpatialTfSamplerForward(s->dnn_handle_,
 //                                          st_desc_,
 //                                          &alpha,
 //                                          in_desc_,
 //                                          data.dptr_,
 //                                          grid.dptr_,
 //                                          &beta,
 //                                          out_desc_,
 //                                          out.dptr_/*output*/), CUDNN_STATUS_SUCCESS);
  }
 //
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
 //    CHECK_EQ(in_data.size(), 2);
 //    CHECK_EQ(out_data.size(), 3);
 //    CHECK_EQ(out_grad.size(), 1);
 //    Stream<gpu> *s = ctx.get_stream<gpu>();
 //    Tensor<gpu, 4, DType> data = in_data[st::kData].get<gpu, 4, DType>(s);
 //    Tensor<gpu, 4, DType> grad = out_grad[st::kOut].get<gpu, 4, DType>(s);
 //    Tensor<gpu, 4, DType> ddata = in_grad[st::kData].get<gpu, 4, DType>(s);
 //    Shape<3> loc_shape = Shape3(data.size(0), 2, 3);
 //    Shape<4> grid_shape = Shape4(grad.size(0), grad.size(2), grad.size(3), 2);
 //    Tensor<gpu, 3, DType> dloc = in_grad[st::kLoc].get_with_shape<gpu, 3, DType>(loc_shape, s);
 //    Tensor<gpu, 4, DType> grid = out_data[st::kGridSrc]
 //                    .get_with_shape<gpu, 4, DType>(grid_shape, s);
 //    // do not use out_grad[st::kGridSrc], because dgrid is a intermediate tensor, and not include in
 //    // DeclareBackwardDependency, another, we can we reuse grid for inplace operator
 //    typename DataType<DType>::ScaleType alpha = 1.0f;
 //    typename DataType<DType>::ScaleType beta = 0.0f;
 //    typename DataType<DType>::ScaleType alpha_dgrid = 1.0f;
 //    typename DataType<DType>::ScaleType beta_dgrid = 0.0f;
 //    CHECK_EQ(cudnnSpatialTfSamplerBackward(s->dnn_handle_,
 //                                           st_desc_,
 //                                           &alpha,
 //                                           in_desc_,
 //                                           data.dptr_,
 //                                           &beta,
 //                                           in_desc_/*reuse in_desc_*/,
 //                                           ddata.dptr_/*output*/,
 //                                           &alpha_dgrid,
 //                                           out_desc_/*reuse out_desc_*/,
 //                                           grad.dptr_,
 //                                           grid.dptr_,
 //                                           &beta_dgrid,
 //                                           grid.dptr_/*output, reuse grid*/), CUDNN_STATUS_SUCCESS);
 //    if (param_.transform_type == st::kAffine) {
 //      CHECK_EQ(cudnnSpatialTfGridGeneratorBackward(s->dnn_handle_,
 //                                                   st_desc_,
 //                                                   grid.dptr_,
 //                                                   dloc.dptr_/*out*/), CUDNN_STATUS_SUCCESS);
 //    }
  }
 //
 private:
  inline void Init(mshadow::Stream<gpu> *s,
                   const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    // CHECK_EQ(in_data.size(), 2);
    // CHECK_EQ(out_data.size(), 3);
    // if (!init_cudnn_) {
    //   init_cudnn_ = true;
    //   // Tensor<gpu, 4, DType> data = in_data[st::kData].get<gpu, 4, DType>(s);
    //   // Tensor<gpu, 4, DType> out = out_data[st::kOut].get<gpu, 4, DType>(s);
    //   CHECK_EQ(cudnnCreateRNNDescriptor(&rnn_desc_), CUDNN_STATUS_SUCCESS);
    //   CHECK_EQ(cudnnCreateDropoutDescriptor(&rnn_dropout_), CUDNN_STATUS_SUCCESS);

    //   CHECK_EQ(cudnnCreateTensorDescriptor(&in_desc_), CUDNN_STATUS_SUCCESS);
    //   CHECK_EQ(cudnnCreateTensorDescriptor(&out_desc_), CUDNN_STATUS_SUCCESS);
    //   CHECK_EQ(cudnnSetTensor4dDescriptor(in_desc_,
    //                                       format_,
    //                                       dtype_,
    //                                       data.size(0),
    //                                       data.size(1),
    //                                       data.size(2),
    //                                       data.size(3)), CUDNN_STATUS_SUCCESS);
    //   CHECK_EQ(cudnnSetTensor4dDescriptor(out_desc_,
    //                                       format_,
    //                                       dtype_,
    //                                       out.size(0),
    //                                       out.size(1),
    //                                       out.size(2),
    //                                       out.size(3)), CUDNN_STATUS_SUCCESS);
    //   if (param_.sampler_type == st::kBilinear) {
    //     int dim[] = {static_cast<int>(out.size(0)), static_cast<int>(out.size(1)),
    //                  static_cast<int>(out.size(2)), static_cast<int>(out.size(3))};
    //     CHECK_EQ(cudnnSetSpatialTransformerNdDescriptor(st_desc_,
    //                                                     sampler_,
    //                                                     dtype_,
    //                                                     4,
    //                                                     dim) , CUDNN_STATUS_SUCCESS);
    //   }
    // }
  }
 
  bool init_cudnn_;
  cudnnDataType_t dtype_;
  cudnnRNNDescriptor_t rnn_desc_;
  cudnnRNNMode_t rnn_mode_;
  cudnnDirectionMode_t rnn_direction_;
  cudnnRNNInputMode_t rnn_input_mode_;
  cudnnDropoutDescriptor_t rnn_dropout_;
  // cudnnTensorDescriptor_t in_desc_;
  // cudnnTensorDescriptor_t out_desc_;
  #if CUDNN_MAJOR == 5
  cudnnTensorFormat_t format_;
  #endif
  RNNParam param_;
};
#endif  // __CUDACC__ && CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CUDNN_SPATIAL_TRANSFORMER_INL_H_
