/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_relu.cu
 * \brief
 * \author Ziheng Jiang
*/
#include "./quantized_relu-inl.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {


template<typename DType>
class QuantizedReluCuDNNOp : public Operator {
 public:
  explicit QuantizedReluCuDNNOp() {
    init_cudnn_ = false;
    alpha_ = 1.0f;
    beta_  = 0.0f;
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
    mode_  = CUDNN_ACTIVATION_RELU;
    nan_prop_ = CUDNN_NOT_PROPAGATE_NAN;
  }

  ~QuantizedReluCuDNNOp() {
    if (init_cudnn_) {
      CHECK_EQ(cudnnDestroyTensorDescriptor(shape_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyActivationDescriptor(act_desc_), CUDNN_STATUS_SUCCESS);
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;

    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(out_data.size(), 3U);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->dnn_handle_ownership_, Stream<gpu>::OwnHandle);

    TBlob data = in_data[0], out = out_data[0];
    CHECK(data.shape_.ndim() <= 4) << "Not support yet";
    TShape shape{1, 1, 1, 1};
    for (size_t i = 0; i < data.shape_.ndim(); ++i) shape[i] = data.shape_[i];
    if (!init_cudnn_) {
      InitDescriptors(shape);
      init_cudnn_ = true;
    }

    // (TODO) problem here, threshold is invalid in CuDNN API
    // float *imin_range = (float*)malloc(sizeof(float));
    // float *imax_range = (float*)malloc(sizeof(float));
    // cudaMemcpy(imin_range, in_data[1].dptr_,
    //   sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(imax_range, in_data[2].dptr_,
    //   sizeof(float), cudaMemcpyDeviceToHost);
    // LOG(INFO) << "min_range: " << *imin_range;
    // LOG(INFO) << "max_range: " << *imax_range;
    // DType zero_as_quantized =
    //       FloatToQuantized<DType>(0.0f, *imin_range, *imax_range);
    // double threshold = static_cast<double>(zero_as_quantized);
    // LOG(INFO) << "threshold: " << threshold;
    CUDNN_CALL(cudnnSetActivationDescriptor(act_desc_,
                                            mode_,
                                            nan_prop_,
                                            0.0f));
    CUDNN_CALL(cudnnActivationForward(s->dnn_handle_,
                                      act_desc_,
                                      &alpha_,
                                      shape_desc_,
                                      data.dptr_,
                                      &beta_,
                                      shape_desc_,
                                      out.dptr_));
    Tensor<gpu, 1, float> omin_range = out_data[1].FlatTo1D<gpu, float>(s);
    Tensor<gpu, 1, float> omax_range = out_data[2].FlatTo1D<gpu, float>(s);
    ASSIGN_DISPATCH(omin_range, req[1],
      F<mshadow_op::identity>(in_data[1].FlatTo1D<gpu, float>(s)));
    ASSIGN_DISPATCH(omax_range, req[2],
      F<mshadow_op::identity>(in_data[2].FlatTo1D<gpu, float>(s)));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    LOG(FATAL) << "backward of quantized_relu not implemented yet.";
  }

 private:
  bool init_cudnn_;
  float alpha_;
  float beta_;
  cudnnDataType_t dtype_;
  cudnnActivationMode_t mode_;
  cudnnTensorDescriptor_t shape_desc_;
  cudnnActivationDescriptor_t act_desc_;
  cudnnNanPropagation_t nan_prop_;

  void InitDescriptors(TShape shape) {
    CHECK(!init_cudnn_);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&shape_desc_));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(shape_desc_,
                                          CUDNN_TENSOR_NCHW,
                                          dtype_,
                                          shape[0],
                                          shape[1],
                                          shape[2],
                                          shape[3]));
  }
};  // class QuantizedReluCuDNNOp


template<>
Operator *CreateOp<gpu>(int dtype) {
  Operator *op = NULL;
  CHECK(dtype == mshadow::kInt8);
  op = new QuantizedReluCuDNNOp<int8_t>();
  return op;
}
}  // namespace op
}  // namespace mxnet

