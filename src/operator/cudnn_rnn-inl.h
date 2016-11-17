/*!
 * Copyright (c) 2016 by Contributors
 * \file cudnn_rnn-inl.h
 * \brief
 * \author Sebastian Bodenstein
*/
#ifndef MXNET_OPERATOR_CUDNN_RNN_INL_H_
#define MXNET_OPERATOR_CUDNN_RNN_INL_H_

#include <mxnet/storage.h>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <cstdint>
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
    input_mode_ = CUDNN_LINEAR_INPUT;  // Don't support this yet
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
    direction_ = param_.bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
    // Other
    if (param_.mode == rnn_enum::kLstm)
      param_.lstm_q_ = true;
    else
      param_.lstm_q_ = false;
  }

  ~CuDNNRNNOp() {
    if (init_cudnn_) {
      for (int i = 0; i < x_desc_vec_.size(); ++i) {
        CHECK_EQ(cudnnDestroyTensorDescriptor(x_desc_vec_[i]), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnDestroyTensorDescriptor(y_desc_vec_[i]), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnDestroyTensorDescriptor(dx_desc_vec_[i]), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnDestroyTensorDescriptor(dy_desc_vec_[i]), CUDNN_STATUS_SUCCESS);
      }
      CHECK_EQ(cudnnDestroyTensorDescriptor(hx_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(cx_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(hy_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(cy_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(dhx_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(dcx_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(dhy_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(dcy_desc_), CUDNN_STATUS_SUCCESS);

      CHECK_EQ(cudnnDestroyFilterDescriptor(w_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyRNNDescriptor(rnn_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyDropoutDescriptor(dropout_desc_), CUDNN_STATUS_SUCCESS);
      Storage::Get()->Free(dropout_states_);
      Storage::Get()->Free(reserve_space_);
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t in_expected = param_.lstm_q_ ? 4 : 3;
    size_t out_expected = param_.lstm_q_ ? 3 : 2;
    if (!param_.state_outputs)
        out_expected = 1;

    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    // get input + output tensors
    Tensor<gpu, 3, DType> x = in_data[rnn_enum::kData].get<gpu, 3, DType>(s);
    Tensor<gpu, 1, DType> w = in_data[rnn_enum::kParams].get<gpu, 1, DType>(s);
    Tensor<gpu, 3, DType> hx = in_data[rnn_enum::kState].get<gpu, 3, DType>(s);
    Tensor<gpu, 3, DType> y = out_data[rnn_enum::kOut].get<gpu, 3, DType>(s);

    void * hy_ptr = NULL;
    if (param_.state_outputs)
      hy_ptr = out_data[rnn_enum::kStateOut].get<gpu, 3, DType>(s).dptr_;

    DType * cx_ptr = NULL;
    DType * cy_ptr = NULL;

    if (param_.lstm_q_)
      cx_ptr = (in_data[rnn_enum::kStateCell].get<gpu, 3, DType>(s)).dptr_;
    if (param_.lstm_q_ && param_.state_outputs)
      cy_ptr = (out_data[rnn_enum::kStateCellOut].get<gpu, 3, DType>(s)).dptr_;

    CHECK_EQ(x.CheckContiguous(), true);
    CHECK_EQ(w.CheckContiguous(), true);
    CHECK_EQ(hx.CheckContiguous(), true);
    CHECK_EQ(y.CheckContiguous(), true);

    if (!init_cudnn_) {
      Init(s, in_data, out_data);
    }
    // Get temp space
    int temp_size = workspace_size_;
    Tensor<gpu, 1, DType> temp_space =
      ctx.requested[rnn_enum::kTempSpace].get_space_typed<gpu, 1, DType>(
                              mshadow::Shape1(temp_size), s);
    if (ctx.is_train) {
      CHECK_EQ(cudnnRNNForwardTraining(s->dnn_handle_,
                                      rnn_desc_,
                                      param_.seq_length_,
                                      x_desc_vec_.data(),
                                      x.dptr_,
                                      hx_desc_,
                                      hx.dptr_,
                                      cx_desc_,
                                      cx_ptr,
                                      w_desc_,
                                      w.dptr_,
                                      y_desc_vec_.data(),
                                      y.dptr_,
                                      hy_desc_,
                                      hy_ptr,
                                      cy_desc_,
                                      cy_ptr,
                                      temp_space.dptr_,
                                      workspace_byte_,
                                      reserve_space_.dptr,
                                      reserve_space_byte_), CUDNN_STATUS_SUCCESS);
    } else {
      // inference mode
      CHECK_EQ(cudnnRNNForwardInference(s->dnn_handle_,
                                      rnn_desc_,
                                      param_.seq_length_,
                                      x_desc_vec_.data(),
                                      x.dptr_,
                                      hx_desc_,
                                      hx.dptr_,
                                      cx_desc_,
                                      cx_ptr,
                                      w_desc_,
                                      w.dptr_,
                                      y_desc_vec_.data(),
                                      y.dptr_,
                                      hy_desc_,
                                      hy_ptr,
                                      cy_desc_,
                                      cy_ptr,
                                      temp_space.dptr_,
                                      workspace_byte_), CUDNN_STATUS_SUCCESS);
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
    size_t in_expected = param_.lstm_q_ ? 4 : 3;
    size_t out_expected = param_.lstm_q_ ? 3 : 2;
    if (!param_.state_outputs)
      out_expected = 1;

    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);
    CHECK_EQ(in_grad.size(), in_expected);
    CHECK_EQ(out_grad.size(), out_expected);
    CHECK_EQ(req.size(), in_expected);
    CHECK_NE(req[rnn_enum::kData], kAddTo) << "AddTo is not supported for data";
    CHECK_NE(req[rnn_enum::kState], kAddTo) << "AddTo is not supported for state";
    Stream<gpu> *s = ctx.get_stream<gpu>();
    // get input + output tensors
    Tensor<gpu, 3, DType> x = in_data[rnn_enum::kData].get<gpu, 3, DType>(s);
    Tensor<gpu, 3, DType> dx = in_grad[rnn_enum::kData].get<gpu, 3, DType>(s);
    Tensor<gpu, 1, DType> w = in_data[rnn_enum::kParams].get<gpu, 1, DType>(s);
    Tensor<gpu, 1, DType> dw = in_grad[rnn_enum::kParams].get<gpu, 1, DType>(s);
    Tensor<gpu, 3, DType> hx = in_data[rnn_enum::kState].get<gpu, 3, DType>(s);
    Tensor<gpu, 3, DType> dhx = in_grad[rnn_enum::kState].get<gpu, 3, DType>(s);
    Tensor<gpu, 3, DType> y = out_data[rnn_enum::kOut].get<gpu, 3, DType>(s);
    Tensor<gpu, 3, DType> dy = out_grad[rnn_enum::kOut].get<gpu, 3, DType>(s);
    if (req[rnn_enum::kParams] != kAddTo) {
      dw = mshadow::expr::ScalarExp<DType>(0.0f);
    }
    // only need kStateOut grad output_states is true
    void * dhy_ptr = NULL;
    if (param_.state_outputs)
      dhy_ptr = out_grad[rnn_enum::kStateOut].get<gpu, 3, DType>(s).dptr_;

    // Deal with lstm
    void * dcx_ptr = NULL;
    void * dcy_ptr = NULL;
    void * cx_ptr = NULL;

    if (param_.mode == rnn_enum::kLstm) {
      CHECK_NE(req[rnn_enum::kStateCell], kAddTo) << "AddTo is not supported for state cell";
      cx_ptr = (in_data[rnn_enum::kStateCell].get<gpu, 3, DType>(s)).dptr_;
      dcx_ptr = (in_grad[rnn_enum::kStateCell].get<gpu, 3, DType>(s)).dptr_;
    }
    if ((param_.mode == rnn_enum::kLstm) && param_.state_outputs)
        dcy_ptr = (out_grad[rnn_enum::kStateCellOut].get<gpu, 3, DType>(s)).dptr_;

    CHECK_EQ(x.CheckContiguous(), true);
    CHECK_EQ(w.CheckContiguous(), true);
    CHECK_EQ(dw.CheckContiguous(), true);
    CHECK_EQ(hx.CheckContiguous(), true);
    CHECK_EQ(dhx.CheckContiguous(), true);
    CHECK_EQ(y.CheckContiguous(), true);
    CHECK_EQ(dy.CheckContiguous(), true);

    if (!init_cudnn_) {
      Init(s, in_data, out_data);
    }

    // Get temp space
    int temp_size = workspace_size_;
    Tensor<gpu, 1, DType> temp_space =
      ctx.requested[rnn_enum::kTempSpace].get_space_typed<gpu, 1, DType>(
                              mshadow::Shape1(temp_size), s);
    CHECK_EQ(cudnnRNNBackwardData(s->dnn_handle_,
                                rnn_desc_,
                                param_.seq_length_,
                                y_desc_vec_.data(),
                                y.dptr_,
                                dy_desc_vec_.data(),
                                dy.dptr_,
                                dhy_desc_,
                                dhy_ptr,
                                dcy_desc_,
                                dcy_ptr,
                                w_desc_,
                                w.dptr_,
                                hx_desc_,
                                hx.dptr_,
                                cx_desc_,
                                cx_ptr,
                                dx_desc_vec_.data(),
                                dx.dptr_,
                                dhx_desc_,
                                dhx.dptr_,
                                dcx_desc_,
                                dcx_ptr,
                                temp_space.dptr_,
                                workspace_byte_,
                                reserve_space_.dptr,
                                reserve_space_byte_), CUDNN_STATUS_SUCCESS);
    CHECK_EQ(cudnnRNNBackwardWeights(s->dnn_handle_,
                                    rnn_desc_,
                                    param_.seq_length_,
                                    x_desc_vec_.data(),
                                    x.dptr_,
                                    hx_desc_,
                                    hx.dptr_,
                                    y_desc_vec_.data(),
                                    y.dptr_,
                                    temp_space.dptr_,
                                    workspace_byte_,
                                    dw_desc_,
                                    dw.dptr_,
                                    reserve_space_.dptr,
                                    reserve_space_byte_), CUDNN_STATUS_SUCCESS);
  }

 private:
  inline void Init(mshadow::Stream<gpu> *s,
                   const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    #if CUDNN_MAJOR == 5
    format_ = CUDNN_TENSOR_NCHW;
    #endif
    size_t in_expected = param_.lstm_q_ ? 4 : 3;
    size_t out_expected = param_.lstm_q_ ? 3 : 2;
    if (!param_.state_outputs)
      out_expected = 1;

    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);
    if (!init_cudnn_) {
      init_cudnn_ = true;
      // get input + output tensors
      Tensor<gpu, 3, DType> x = in_data[rnn_enum::kData].get<gpu, 3, DType>(s);
      Tensor<gpu, 1, DType> w = in_data[rnn_enum::kParams].get<gpu, 1, DType>(s);
      param_.seq_length_ = x.shape_[0];
      param_.batch_size_ = x.shape_[1];
      param_.input_size_ = x.shape_[2];

      // Tensor Descriptors
      std::vector<cudnnTensorDescriptor_t> x_vec(param_.seq_length_);
      std::vector<cudnnTensorDescriptor_t> y_vec(param_.seq_length_);
      std::vector<cudnnTensorDescriptor_t> dx_vec(param_.seq_length_);
      std::vector<cudnnTensorDescriptor_t> dy_vec(param_.seq_length_);
      int dimA[3];
      int strideA[3];
      for (int i = 0; i < param_.seq_length_; i++) {
        CHECK_EQ(cudnnCreateTensorDescriptor(&x_vec[i]), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnCreateTensorDescriptor(&y_vec[i]), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnCreateTensorDescriptor(&dx_vec[i]), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnCreateTensorDescriptor(&dy_vec[i]), CUDNN_STATUS_SUCCESS);

        dimA[0] = param_.batch_size_;
        dimA[1] = param_.input_size_;
        dimA[2] = 1;
        dimA[0] = param_.batch_size_;
        dimA[1] = param_.input_size_;
        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        CHECK_EQ(cudnnSetTensorNdDescriptor(x_vec[i],
                                  dtype_,
                                  3,
                                  dimA,
                                  strideA), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnSetTensorNdDescriptor(dx_vec[i],
                                  dtype_,
                                  3,
                                  dimA,
                                  strideA), CUDNN_STATUS_SUCCESS);
        dimA[0] = param_.batch_size_;
        dimA[1] = param_.bidirectional ? param_.state_size * 2 : param_.state_size;
        dimA[2] = 1;
        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        CHECK_EQ(cudnnSetTensorNdDescriptor(y_vec[i],
                                  dtype_,
                                  3,
                                  dimA,
                                  strideA), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnSetTensorNdDescriptor(dy_vec[i],
                                  dtype_,
                                  3,
                                  dimA,
                                  strideA), CUDNN_STATUS_SUCCESS);
      }
      x_desc_vec_ = x_vec;
      y_desc_vec_ = y_vec;
      dx_desc_vec_ = dx_vec;
      dy_desc_vec_ = dy_vec;

      // set the state tensors
      dimA[0] = param_.num_layers * (param_.bidirectional ? 2 : 1);
      dimA[1] = param_.batch_size_;
      dimA[2] = param_.state_size;
      strideA[0] = dimA[2] * dimA[1];
      strideA[1] = dimA[2];
      strideA[2] = 1;

      CHECK_EQ(cudnnCreateTensorDescriptor(&hx_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&cx_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&hy_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&cy_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&dhx_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&dcx_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&dhy_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&dcy_desc_), CUDNN_STATUS_SUCCESS);

      CHECK_EQ(cudnnSetTensorNdDescriptor(hx_desc_,
                                          dtype_,
                                          3,
                                          dimA,
                                          strideA), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensorNdDescriptor(cx_desc_,
                                          dtype_,
                                          3,
                                          dimA,
                                          strideA), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensorNdDescriptor(hy_desc_,
                                          dtype_,
                                          3,
                                          dimA,
                                          strideA), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensorNdDescriptor(cy_desc_,
                                          dtype_,
                                          3,
                                          dimA,
                                          strideA), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensorNdDescriptor(dhx_desc_,
                                          dtype_,
                                          3,
                                          dimA,
                                          strideA), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensorNdDescriptor(dcx_desc_,
                                          dtype_,
                                          3,
                                          dimA,
                                          strideA), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensorNdDescriptor(dhy_desc_,
                                          dtype_,
                                          3,
                                          dimA,
                                          strideA), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensorNdDescriptor(dcy_desc_,
                                          dtype_,
                                          3,
                                          dimA,
                                          strideA), CUDNN_STATUS_SUCCESS);

      // Create Dropout descriptors
      CHECK_EQ(cudnnCreateDropoutDescriptor(&dropout_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDropoutGetStatesSize(s->dnn_handle_,
                                        &dropout_byte_), CUDNN_STATUS_SUCCESS);
      dropout_size_ = dropout_byte_ / sizeof(DType);
      dropout_states_ = Storage::Get()->Alloc(dropout_byte_, Context::GPU());
      CHECK_EQ(cudnnSetDropoutDescriptor(dropout_desc_,
                                        s->dnn_handle_,
                                        param_.p,  // keep probability
                                        dropout_states_.dptr,
                                        dropout_byte_,
                                        seed_), CUDNN_STATUS_SUCCESS);
      // RNN descriptors
      CHECK_EQ(cudnnCreateRNNDescriptor(&rnn_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetRNNDescriptor(rnn_desc_,
                                    param_.state_size,
                                    param_.num_layers,
                                    dropout_desc_,
                                    input_mode_,
                                    direction_,
                                    mode_,
                                    dtype_), CUDNN_STATUS_SUCCESS);
      // Get temp space sizes
      CHECK_EQ(cudnnGetRNNWorkspaceSize(s->dnn_handle_,
                                        rnn_desc_,
                                        param_.seq_length_,
                                        x_desc_vec_.data(),
                                        &workspace_byte_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnGetRNNTrainingReserveSize(s->dnn_handle_,
                                        rnn_desc_,
                                        param_.seq_length_,
                                        x_desc_vec_.data(),
                                        &reserve_space_byte_), CUDNN_STATUS_SUCCESS);
      workspace_size_ = workspace_byte_ / sizeof(DType);
      // Allocate the reserve space
      reserve_space_ = Storage::Get()->Alloc(reserve_space_byte_, Context::GPU());

      // Check that number of params are correct
      size_t cudnn_param_size;
      CHECK_EQ(cudnnGetRNNParamsSize(s->dnn_handle_,
                                    rnn_desc_,
                                    x_desc_vec_[0],
                                    &cudnn_param_size,
                                    dtype_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(w.shape_[0] * sizeof(DType), cudnn_param_size);

      // Set param descriptors
      CHECK_EQ(cudnnCreateFilterDescriptor(&w_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateFilterDescriptor(&dw_desc_), CUDNN_STATUS_SUCCESS);
      int dim_w[3] = {1, 1, 1};
      dim_w[0] = w.shape_[0];
      CHECK_EQ(cudnnSetFilterNdDescriptor(w_desc_,
                                          dtype_,
                                          format_,
                                          3,
                                          dim_w), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetFilterNdDescriptor(dw_desc_,
                                          dtype_,
                                          format_,
                                          3,
                                          dim_w), CUDNN_STATUS_SUCCESS);
    }
  }

  cudnnDataType_t dtype_;
  bool init_cudnn_;
  cudnnRNNDescriptor_t rnn_desc_;
  cudnnRNNMode_t mode_;
  cudnnDirectionMode_t direction_;
  cudnnRNNInputMode_t input_mode_;
  cudnnDropoutDescriptor_t dropout_desc_;
  Storage::Handle dropout_states_, reserve_space_;
  uint64_t seed_ = 1337ull;
  size_t workspace_byte_, reserve_space_byte_, dropout_byte_;
  int workspace_size_, dropout_size_;
  std::vector<cudnnTensorDescriptor_t> x_desc_vec_, y_desc_vec_, dx_desc_vec_, dy_desc_vec_;
  cudnnTensorDescriptor_t hx_desc_, cx_desc_;
  cudnnTensorDescriptor_t hy_desc_, cy_desc_;
  cudnnTensorDescriptor_t dhx_desc_, dcx_desc_;
  cudnnTensorDescriptor_t dhy_desc_, dcy_desc_;

  cudnnFilterDescriptor_t w_desc_, dw_desc_;

  #if CUDNN_MAJOR == 5
  cudnnTensorFormat_t format_;
  #endif
  RNNParam param_;
};
#endif  // __CUDACC__ && CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CUDNN_RNN_INL_H_
