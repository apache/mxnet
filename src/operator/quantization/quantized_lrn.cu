/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_lrn.cu
 * \brief
 * \author Ziheng Jiang
*/

#include "./quantized_lrn-inl.h"

namespace mxnet {
namespace op {

template<typename DType>
class QuantizedLRNCuDNNOp : public Operator {
 public:
  explicit QuantizedLRNCuDNNOp(QuantizedLRNParam param) {
    param_ = param;
    init_cudnn_ = false;
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
  }

  ~QuantizedLRNCuDNNOp() {
    if (init_cudnn_) {
      CHECK_EQ(cudnnDestroyLRNDescriptor(lrn_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(shape_desc_), CUDNN_STATUS_SUCCESS);
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(out_data.size(), 3U);
    float alpha = 1.0f;
    float beta  = 0.0f;
    Stream<gpu> *s = ctx.get_stream<gpu>();
    const TBlob& data = in_data[0];
    const TBlob& out = out_data[0];
    if (!init_cudnn_) this->Init(s, in_data, out_data);
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    CUDNN_CALL(cudnnLRNCrossChannelForward(s->dnn_handle_,
                                           lrn_desc_,
                                           CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                           &alpha,
                                           shape_desc_,
                                           data.dptr_,
                                           &beta,
                                           shape_desc_,
                                           out.dptr_));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    LOG(FATAL) << "not implemented";
  }

 private:
  inline void Init(mshadow::Stream<gpu> *s,
                   const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    CHECK(!init_cudnn_) << "Init should only be called when not initialized";
    init_cudnn_ = true;
    const TBlob& data = in_data[0];
    unsigned lrn_n = param_.nsize;
    double alpha   = param_.alpha;
    double beta    = param_.beta;
    double lrn_k   = param_.knorm;
    CUDNN_CALL(cudnnCreateLRNDescriptor(&lrn_desc_));
    CUDNN_CALL(cudnnSetLRNDescriptor(lrn_desc_,
                                     lrn_n,
                                     alpha,
                                     beta,
                                     lrn_k))
    CUDNN_CALL(cudnnCreateTensorDescriptor(&shape_desc_));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(shape_desc_,
                                          CUDNN_TENSOR_NCHW,
                                          dtype_,
                                          data.shape_[0],
                                          data.shape_[1],
                                          data.shape_[2],
                                          data.shape_[3]));
  }
  bool init_cudnn_;
  QuantizedLRNParam param_;
  cudnnDataType_t dtype_;
  cudnnLRNDescriptor_t lrn_desc_;
  cudnnTensorDescriptor_t shape_desc_;
};  // class CuDNNLocalResponseNormOp


template<>
Operator* CreateOp<gpu>(QuantizedLRNParam param, int dtype) {
  Operator *op = NULL;
  op = new QuantizedLRNCuDNNOp<int8_t>(param);
  return op;
}

}  // namespace op
}  // namespace mxnet


