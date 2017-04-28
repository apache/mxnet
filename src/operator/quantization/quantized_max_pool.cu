/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_max_pool.cu
 * \brief
 * \author Ziheng Jiang
*/
#include <vector>
#include "./quantized_max_pool-inl.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

template<typename DType>
class QuantizedMaxPoolCuDNNOp : public Operator {
 public:
  explicit QuantizedMaxPoolCuDNNOp(QuantizedMaxPoolParam p) {
    param_ = p;
    if (param_.layout == mshadow::kNCHW) {
      N = 0, H = 2, W = 3, C = 1;
      format_ = CUDNN_TENSOR_NCHW;
    } else if (param_.layout == mshadow::kNHWC) {
      N = 0, H = 1, W = 2, C = 3;
      format_ = CUDNN_TENSOR_NHWC;
    }
    init_cudnn_ = false;
    alpha_ = 1.0f;
    beta_  = 0.0f;
    dtype_ = CUDNN_DATA_INT8;
    mode_ = CUDNN_POOLING_MAX;
    nan_prop_ = CUDNN_NOT_PROPAGATE_NAN;
  }

  ~QuantizedMaxPoolCuDNNOp() {
    if (init_cudnn_) {
      CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc_));
      CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_desc_));
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
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    CHECK(param_.kernel.ndim() == 2) << "Only support 2D pooling";
    if (!init_cudnn_) this->Init(s, in_data, out_data);
    CUDNN_CALL(cudnnPoolingForward(s->dnn_handle_,
                                   pool_desc_,
                                   &alpha_,
                                   in_desc_,
                                   in_data[0].dptr_,
                                   &beta_,
                                   out_desc_,
                                   out_data[0].dptr_));

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
    LOG(FATAL) << "backward is not supported yet";
  }

 private:
  inline void Init(mshadow::Stream<gpu> *s,
                   const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    CHECK(!init_cudnn_)
      << "Init should only be called when init_cudnn is false";
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(out_data.size(), 3U);
    CHECK(param_.kernel.ndim() == 2) << "only support 2d pooling";
    const TBlob& data = in_data[0];
    const TBlob& out  = out_data[0];
    TShape dshape = data.shape_;
    TShape oshape = out.shape_;
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(in_desc_,
                                          format_,
                                          dtype_,
                                          dshape[N],
                                          dshape[C],
                                          dshape[H],
                                          dshape[W]));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc_,
                                          format_,
                                          dtype_,
                                          oshape[N],
                                          oshape[C],
                                          oshape[H],
                                          oshape[W]));
    CUDNN_CALL(cudnnSetPooling2dDescriptor(
      pool_desc_,
      mode_,
      nan_prop_,
      param_.kernel[0],
      param_.kernel[1],
      param_.pad[0],
      param_.pad[1],
      param_.stride[0],
      param_.stride[1]));
  }
  bool init_cudnn_;
  uint32_t N, H, W, C;
  float alpha_;
  float beta_;
  cudnnHandle_t handle_;
  cudnnDataType_t dtype_;
  cudnnTensorFormat_t format_;
  cudnnPoolingMode_t mode_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnPoolingDescriptor_t pool_desc_;
  cudnnNanPropagation_t nan_prop_;
  QuantizedMaxPoolParam param_;
};  // class QuantizedMaxPoolCuDNNOp

template<>
Operator *CreateOp<gpu>(QuantizedMaxPoolParam param, int dtype) {
  Operator *op = NULL;
  op = new QuantizedMaxPoolCuDNNOp<int8_t>(param);
  return op;
}

}  // namespace op
}  // namespace mxnet

