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
    // Defaults
    input_mode_ = CUDNN_LINEAR_INPUT; 
    // RNN Mode
    switch (param_.mode) {
      case rnn_enum::kRnnRelu:
        mode_ = CUDNN_RNN_RELU;
        break;
      case rnn_enum::kRnnTanh:
        mode_ = CUDNN_RNN_TANH;
        break;
      case rnn_enum::kLstm:
        mode_ = CUDNN_LSTM;
        break;
      case rnn_enum::kGru:
        mode_ = CUDNN_GRU;
        break;
      default:
        LOG(FATAL) << "Not implmented";
    }
    // RNN Direction
    switch (param_.direction) {
      case rnn_enum::kUnidirectional:
        direction_ = CUDNN_UNIDIRECTIONAL;
        break;
      case rnn_enum::kBidirectional:
        direction_ = CUDNN_BIDIRECTIONAL;
        break;
      default:
        LOG(FATAL) << "Not implmented";
    }
  }

  ~CuDNNRNNOp() {
    if (init_cudnn_) {
      CHECK_EQ(cudnnDestroyTensorDescriptor(x_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(hx_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(y_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(hy_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyFilterDescriptor(w_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyRNNDescriptor(rnn_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyDropoutDescriptor(dropout_desc_), CUDNN_STATUS_SUCCESS);
      if (param_.mode == rnn_enum::kLstm){
            CHECK_EQ(cudnnDestroyTensorDescriptor(cx_desc_), CUDNN_STATUS_SUCCESS);
            CHECK_EQ(cudnnDestroyTensorDescriptor(cy_desc_), CUDNN_STATUS_SUCCESS);
      }
    }
  }
 
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    Stream<gpu> *s = ctx.get_stream<gpu>();
    if(!init_cudnn_){
      Init(s, in_data, out_data);
    }
    // get input + output tensors
    Tensor<gpu, 3, DType> data = in_data[rnn_enum::kData].get<gpu, 3, DType>(s);
    Tensor<gpu, 1, DType> params = in_data[rnn_enum::kParams].get<gpu, 1, DType>(s);
    Tensor<gpu, 3, DType> state = in_data[rnn_enum::kStateIn].get<gpu, 3, DType>(s);

    Tensor<gpu, 3, DType> out = out_data[rnn_enum::kOut].get<gpu, 3, DType>(s);
    Tensor<gpu, 3, DType> out_state = out_data[rnn_enum::kStateOut].get<gpu, 3, DType>(s);

    if (param_.mode == rnn_enum::kLstm){
      Tensor<gpu, 3, DType> cell_state = 
        in_data[rnn_enum::kCellStateIn].get<gpu, 3, DType>(s);
      Tensor<gpu, 3, DType> out_cell_state = 
        in_data[rnn_enum::kCellStateOut].get<gpu, 3, DType>(s);
    }
    // if (param_.mode == rnn_enum::kLstm){
    //   CHECK_EQ(in_data.size(), 4);
    //   CHECK_EQ(out_data.size(), 3);
    // }
    // else{
    //   CHECK_EQ(in_data.size(), 3);
    //   CHECK_EQ(out_data.size(), 2);
    // }
    // // Get tensors
    // 
    // Tensor<gpu, 3, DType> data = in_data[rnn_enum::kData].get<gpu, 3, DType>(s);
    // Tensor<gpu, 1, DType> params = in_data[rnn_enum::kParams].get<gpu, 1, DType>(s);
    // Tensor<gpu, 3, DType> state = in_data[rnn_enum::kStateIn].get<gpu, 3, DType>(s);

    // Tensor<gpu, 3, DType> out = out_data[rnn_enum::kOut].get<gpu, 3, DType>(s);
    // Tensor<gpu, 3, DType> out_state = out_data[rnn_enum::kOut].get<gpu, 3, DType>(s);

    // if (param_.mode == rnn_enum::kLstm){
    //   Tensor<gpu, 3, DType> cell_state = 
    //     in_data[rnn_enum::kCellStateIn].get<gpu, 3, DType>(s);
    //   Tensor<gpu, 3, DType> out_cell_state = 
    //     in_data[rnn_enum::kCellStateOut].get<gpu, 3, DType>(s);
    // }
 //    
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
    #if CUDNN_MAJOR == 5
    format_ = CUDNN_TENSOR_NCHW;
    #endif
    if(param_.mode == rnn_enum::kLstm){
      CHECK_EQ(in_data.size(), 4);
      CHECK_EQ(out_data.size(), 3);
    }
    else{
      CHECK_EQ(in_data.size(), 3);
      CHECK_EQ(out_data.size(), 2);
    }
    if (!init_cudnn_) {
      init_cudnn_ = true;
      // get input + output tensors
      Tensor<gpu, 3, DType> data = in_data[rnn_enum::kData].get<gpu, 3, DType>(s);
      Tensor<gpu, 1, DType> params = in_data[rnn_enum::kParams].get<gpu, 1, DType>(s);
      Tensor<gpu, 3, DType> state = in_data[rnn_enum::kStateIn].get<gpu, 3, DType>(s);

      Tensor<gpu, 3, DType> out = out_data[rnn_enum::kOut].get<gpu, 3, DType>(s);
      Tensor<gpu, 3, DType> out_state = out_data[rnn_enum::kStateOut].get<gpu, 3, DType>(s);

      if(param_.mode == rnn_enum::kLstm){
        Tensor<gpu, 3, DType> cell_state = 
          in_data[rnn_enum::kCellStateIn].get<gpu, 3, DType>(s);
        Tensor<gpu, 3, DType> out_cell_state = 
          in_data[rnn_enum::kCellStateOut].get<gpu, 3, DType>(s);
      }

      // Create descriptors
      CHECK_EQ(cudnnCreateRNNDescriptor(&rnn_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateDropoutDescriptor(&dropout_desc_), CUDNN_STATUS_SUCCESS);

      CHECK_EQ(cudnnCreateFilterDescriptor(&w_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&x_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&hx_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&y_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&hy_desc_), CUDNN_STATUS_SUCCESS);

      if (param_.mode == rnn_enum::kLstm){
        CHECK_EQ(cudnnCreateTensorDescriptor(&cx_desc_), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnCreateTensorDescriptor(&cy_desc_), CUDNN_STATUS_SUCCESS);
      }     
      // set dropout 
      // cudnnSetDropoutDescriptor(dropout_desc_,
      //                           s->dnn_handle_,
      //                           param_.p,
      //                           void * states,
      //                           size_t stateSizeInBytes,
      //                           unsigned long long seed)
      // set RNN 
      CHECK_EQ(cudnnSetRNNDescriptor(rnn_desc_,
                                    param_.state_size,
                                    param_.num_layers,
                                    dropout_desc_,
                                    input_mode_,
                                    direction_,
                                    mode_,
                                    dtype_), CUDNN_STATUS_SUCCESS);
      // Set params
      int dim_params[3] = {params.shape_[0], 1, 1};
      CHECK_EQ(cudnnSetFilterNdDescriptor(w_desc_,
                                          dtype_,
                                          format_,
                                          3,
                                          dim_params
                                         ), CUDNN_STATUS_SUCCESS);
      // Get strides
      int stride_data[3] = {data.shape_[2]*data.shape_[1], data.shape_[2], 1};
      int stride_state[3] = {state.shape_[2]*state.shape_[1], state.shape_[2], 1};
      int stride_out[3] = {out.shape_[2]*out.shape_[1], out.shape_[2], 1};   
      int stride_out_state[3] = 
        {out_state.shape_[2]*out_state.shape_[1], out_state.shape_[2], 1};
 
      // cuDNN needs int arrays for dim, not index_t array used in Shape
      int dim_data[3];
      int dim_state[3];
      int dim_out[3];
      int dim_out_state[3];
      std::copy(std::begin(data.shape_.shape_), std::end(data.shape_.shape_), std::begin(dim_data));
      std::copy(std::begin(state.shape_.shape_), std::end(state.shape_.shape_), std::begin(dim_state));
      std::copy(std::begin(out.shape_.shape_), std::end(out.shape_.shape_), std::begin(dim_out));
      std::copy(std::begin(out_state.shape_.shape_), std::end(out_state.shape_.shape_), std::begin(dim_out_state));

      // set the tensor descriptors
      CHECK_EQ(cudnnSetTensorNdDescriptor(x_desc_,
                                          dtype_,
                                          3,
                                          dim_data,
                                          stride_data
                                         ), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensorNdDescriptor(hx_desc_,
                                          dtype_,
                                          3,
                                          dim_state,
                                          stride_state
                                         ), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensorNdDescriptor(y_desc_,
                                          dtype_,
                                          3,
                                          dim_out,
                                          stride_out
                                         ), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensorNdDescriptor(hy_desc_,
                                          dtype_,
                                          3,
                                          dim_out_state,
                                          stride_out_state
                                         ), CUDNN_STATUS_SUCCESS);
      // LSTM has two extra descriptors
      if (param_.mode == rnn_enum::kLstm){
        CHECK_EQ(cudnnSetTensorNdDescriptor(cx_desc_,
                                            dtype_,
                                            3,
                                            dim_state,
                                            stride_state
                                          ), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnSetTensorNdDescriptor(cy_desc_,
                                            dtype_,
                                            3,
                                            dim_out_state,
                                            stride_out_state
                                          ), CUDNN_STATUS_SUCCESS);
      }   
    }
  }

  cudnnDataType_t dtype_;
  bool init_cudnn_;
  cudnnRNNDescriptor_t rnn_desc_;
  cudnnRNNMode_t mode_;
  cudnnDirectionMode_t direction_;
  cudnnRNNInputMode_t input_mode_;
  cudnnDropoutDescriptor_t dropout_desc_;

  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t hx_desc_;
  cudnnTensorDescriptor_t cx_desc_;    
  cudnnTensorDescriptor_t y_desc_; 
  cudnnTensorDescriptor_t hy_desc_; 
  cudnnTensorDescriptor_t cy_desc_; 

  cudnnFilterDescriptor_t w_desc_;   

  #if CUDNN_MAJOR == 5
  cudnnTensorFormat_t format_;
  #endif
  RNNParam param_;
};
#endif  // __CUDACC__ && CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CUDNN_SPATIAL_TRANSFORMER_INL_H_
