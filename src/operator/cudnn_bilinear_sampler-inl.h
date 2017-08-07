/*!
 * Copyright (c) 2016 by Contributors
 * \file cudnn_bilinear_sampler-inl.h
 * \brief
 * \author Xu Dong
*/
#ifndef MXNET_OPERATOR_CUDNN_BILINEAR_SAMPLER_INL_H_
#define MXNET_OPERATOR_CUDNN_BILINEAR_SAMPLER_INL_H_

#include <algorithm>
#include <vector>
#include "./bilinear_sampler-inl.h"
namespace mxnet {
namespace op {
#if defined(__CUDACC__) && MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
template<typename DType>
class CuDNNBilinearSamplerOp : public Operator {
 public:
  explicit CuDNNBilinearSamplerOp(BilinearSamplerParam param) {
    this->param_ = param;
    init_cudnn_ = false;
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
    sampler_ = CUDNN_SAMPLER_BILINEAR;
  }

  ~CuDNNBilinearSamplerOp() {
    if (init_cudnn_) {
      CUDNN_CALL(cudnnDestroySpatialTransformerDescriptor(st_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc_));
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(req[bs::kOut], kWriteTo);
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 2U);
    Stream<gpu> *s = ctx.get_stream<gpu>();

    Tensor<gpu, 4, DType> data = in_data[bs::kData].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> grid = in_data[bs::kGrid].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> grid_tmp = out_data[bs::kTmp].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> out = out_data[bs::kOut].get<gpu, 4, DType>(s);
    // grid_tmp : (batch, h, w, 2)
    grid_tmp = transpose(grid, Shape4(0, 2, 3, 1));
    if (!init_cudnn_) {
     Init(s, in_data, out_data);
    }
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    CHECK_EQ(grid_tmp.CheckContiguous(), true);
    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;
    CUDNN_CALL(cudnnSpatialTfSamplerForward(s->dnn_handle_,
                                            st_desc_,
                                            &alpha,
                                            in_desc_,
                                            data.dptr_,
                                            grid_tmp.dptr_,
                                            &beta,
                                            out_desc_,
                                            out.dptr_));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_NE(req[bs::kData], kWriteInplace);
    CHECK_NE(req[bs::kGrid], kWriteInplace);
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 2U);
    CHECK_EQ(out_grad.size(), 1U);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4, DType> data = in_data[bs::kData].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> grid_tmp = out_data[bs::kTmp].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> gdata = in_grad[bs::kData].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> ggrid = in_grad[bs::kGrid].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> grad = out_grad[bs::kOut].get<gpu, 4, DType>(s);

    typename DataType<DType>::ScaleType alpha = (req[bs::kData] == kNullOp) ? 0.0f : 1.0f;
    typename DataType<DType>::ScaleType beta = (req[bs::kData] == kAddTo) ? 1.0f : 0.0f;
    typename DataType<DType>::ScaleType alpha_dgrid = 1.0f;
    typename DataType<DType>::ScaleType beta_dgrid = 0.0f;
    CUDNN_CALL(cudnnSpatialTfSamplerBackward(s->dnn_handle_,
                                             st_desc_,
                                             &alpha,
                                             in_desc_,
                                             data.dptr_,
                                             &beta,
                                             in_desc_/*reuse in_desc_*/,
                                             gdata.dptr_/*output*/,
                                             &alpha_dgrid,
                                             out_desc_/*reuse out_desc_*/,
                                             grad.dptr_,
                                             grid_tmp.dptr_,
                                             &beta_dgrid,
                                             grid_tmp.dptr_));
    Assign(ggrid, req[bs::kGrid], transpose(grid_tmp, Shape4(0, 3, 1, 2)));
  }

 private:
  inline void Init(mshadow::Stream<gpu> *s,
                   const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    #if CUDNN_MAJOR >= 5
    format_ = CUDNN_TENSOR_NCHW;
    #endif
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 2U);
    if (!init_cudnn_) {
      init_cudnn_ = true;
      Tensor<gpu, 4, DType> data = in_data[bs::kData].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> out = out_data[bs::kOut].get<gpu, 4, DType>(s);
      CUDNN_CALL(cudnnCreateSpatialTransformerDescriptor(&st_desc_));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(in_desc_,
                                            format_,
                                            dtype_,
                                            data.size(0),
                                            data.size(1),
                                            data.size(2),
                                            data.size(3)));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc_,
                                            format_,
                                            dtype_,
                                            out.size(0),
                                            out.size(1),
                                            out.size(2),
                                            out.size(3)));
      int dim[] = {static_cast<int>(out.size(0)), static_cast<int>(out.size(1)),
                   static_cast<int>(out.size(2)), static_cast<int>(out.size(3))};
      CUDNN_CALL(cudnnSetSpatialTransformerNdDescriptor(st_desc_,
                                                        sampler_,
                                                        dtype_,
                                                        4,
                                                        dim));
    }
  }

  bool init_cudnn_;
  cudnnDataType_t dtype_;
  cudnnSpatialTransformerDescriptor_t st_desc_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnSamplerType_t sampler_;
  #if CUDNN_MAJOR >= 5
  cudnnTensorFormat_t format_;
  #endif
  BilinearSamplerParam param_;
};
#endif  // __CUDACC__ && CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CUDNN_BILINEAR_SAMPLER_INL_H_
