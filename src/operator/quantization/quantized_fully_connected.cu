/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_fully_connected.cu
 * \brief
 * \author Ziheng Jiang
*/
#include "./quantized_fully_connected-inl.h"
#include "./quantization_utils.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

// value + bias_value * (range1 / limit_range1) * (limit_range2 / range2)
struct QuantizedBiasAddStruct {
  MSHADOW_XINLINE static void Map(int i, size_t k, int32_t *out,
    const int8_t *bias, const float *min_out, const float *max_out,
    const float *min_bias, const float *max_bias) {
    typedef int32_t T1;
    typedef int8_t  T2;
    float float_for_one_out_quant = (max_out[0] - min_out[0]) /
        (static_cast<double>(MaxValue<T1>()) -
         static_cast<double>(MinValue<T1>()));
    float float_for_one_bias_quant = (max_bias[0] - min_bias[0]) /
        (static_cast<double>(MaxValue<T2>()) -
         static_cast<double>(MinValue<T2>()));
    out[i] = (out[i] * float_for_one_out_quant +
              bias[i%k] * float_for_one_bias_quant) /
             float_for_one_out_quant;
  }
};

// value + bias_value * (range1 / limit_range1) * (limit_range2 / range2)
struct QuantizedBiasAddStruct2 {
  MSHADOW_XINLINE static void Map(int i, size_t k, int32_t *out,
    const int8_t *bias, const float *min_out, const float *max_out,
    const float *min_bias, const float *max_bias) {
    typedef int32_t T1;
    typedef int8_t  T2;
    float float_for_one_out_quant  =
      *max_out / static_cast<double>(MaxValue<T1>());
    float float_for_one_bias_quant =
      *max_bias / static_cast<double>(MaxValue<T2>());
    out[i] = (out[i] * float_for_one_out_quant +
              bias[i%k] * float_for_one_bias_quant) /
             float_for_one_out_quant;
  }
};

template<typename SrcType, typename DstType, typename CmpType>
class QuantizedFullyConnectedCublasOp : public Operator {
 public:
  explicit QuantizedFullyConnectedCublasOp(const Context& ctx,
                                   const std::vector<TShape>& in_shape,
                                   const std::vector<TShape>& out_shape,
                                   const QuantizedFullyConnectedParam& param) {
    alpha_ = 1.0f;
    beta_  = 0.0f;
    src_type_ = mshadow::DataType<SrcType>::kCudaFlag;
    dst_type_ = mshadow::DataType<DstType>::kCudaFlag;
    cmp_type_ = mshadow::DataType<CmpType>::kCudaFlag;
    param_ = param;
  }

  ~QuantizedFullyConnectedCublasOp() {
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mxnet_op;
    size_t num_inputs = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(),  num_inputs * 3);
    CHECK_EQ(out_data.size(), 3U);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->blas_handle_ownership_, Stream<gpu>::OwnHandle);
    const TBlob& data   =  in_data[0];
    const TBlob& weight =  in_data[1];
    const TBlob& out    = out_data[0];
    TShape dshape = data.shape_;
    TShape wshape = weight.shape_;
    TShape oshape = out.shape_;
    // (m, n) * (k, n).T = (m, k)
    // A * B.T = C

    // row_C = col_C(T) = cublas(col_B * col_A(T)) = cublas(row_B(T), row_A)
    // row_C = col_C(T) = cublas(col_B(T) * col_A(T)) = cublas(row_B, row_A)
    size_t m = dshape[0], n = dshape[1], k = wshape[0];
    CUBLAS_CALL(cublasGemmEx(s->blas_handle_,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             k,
                             m,
                             n,
                             &alpha_,
                             weight.dptr_,
                             src_type_,
                             n,
                             data.dptr_,
                             src_type_,
                             n,
                             &beta_,
                             out.dptr_,
                             dst_type_,
                             k,
                             cmp_type_,
                             CUBLAS_GEMM_DFALT));

    Kernel<QuantizationRangeForMultiplicationStruct, gpu>::Launch(s, 1,
      out_data[1].dptr<float>(), out_data[2].dptr<float>(),
       in_data[num_inputs].dptr<float>(),   in_data[num_inputs+1].dptr<float>(),
       in_data[num_inputs+2].dptr<float>(), in_data[num_inputs+3].dptr<float>());

    if (!param_.no_bias) {
      const TBlob& bias = in_data[2];
      Kernel<QuantizedBiasAddStruct2, gpu>::Launch(s, out.Size(),
          k, out.dptr<int32_t>(), bias.dptr<int8_t>(),
          out_data[1].dptr<float>(), out_data[2].dptr<float>(),
           in_data[7].dptr<float>(),  in_data[8].dptr<float>());
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    LOG(INFO) << "Not implemented";
  }


 private:
  CmpType alpha_;
  CmpType beta_;
  cudaDataType src_type_;
  cudaDataType dst_type_;
  cudaDataType cmp_type_;
  QuantizedFullyConnectedParam param_;

};  // class QuantizedFullyConnectedCublasOp


template<>
Operator* CreateOp<gpu>(int dtype,
                        const Context& ctx,
                        const std::vector<TShape>& in_shape,
                        const std::vector<TShape>& out_shape,
                        const QuantizedFullyConnectedParam& param) {
  Operator *op = NULL;
  op = new QuantizedFullyConnectedCublasOp<int8_t, int32_t, int32_t>(ctx,
    in_shape, out_shape, param);
  return op;
}

}  // namespace op
}  // namespace mxnet

