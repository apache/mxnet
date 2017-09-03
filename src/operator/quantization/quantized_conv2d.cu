/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_conv2d.cu
 * \brief
 * \author Ziheng Jiang
*/
#include "./quantized_conv2d-inl.h"
#include "./quantization_utils.h"
#include "../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {

template<typename SrcType, typename DstType, typename CmpType>
class QuantizedConv2DCuDNNOp : public Operator {
 public:
  explicit QuantizedConv2DCuDNNOp(const Context& ctx,
                                  const std::vector<TShape>& in_shape,
                                  const std::vector<TShape>& out_shape,
                                  const QuantizedConv2DParam& param) {
    param_ = param;
    if (param_.layout == mshadow::kNCHW) {
      N = 0, H = 2, W = 3, C = 1;
    } else if (param_.layout == mshadow::kNHWC) {
      N = 0, H = 1, W = 2, C = 3;
    }
    src_type_ = mshadow::DataType<SrcType>::kCudnnFlag;
    dst_type_ = mshadow::DataType<DstType>::kCudnnFlag;
    cmp_type_ = mshadow::DataType<CmpType>::kCudnnFlag;
    algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    format_ = CUDNN_TENSOR_NHWC;
    init_temp_size_ = false;
    InitDescriptors(ctx, in_shape, out_shape);
  }

  ~QuantizedConv2DCuDNNOp() {
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc_));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 6U);
    CHECK_EQ(out_data.size(), 3U);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->dnn_handle_ownership_, Stream<gpu>::OwnHandle);

    TBlob data   = in_data[0];
    TBlob filter = in_data[1];
    TBlob out    = out_data[0];
    const TShape& dshape = data.shape_;
    const TShape& fshape = filter.shape_;
    const TShape& oshape = out.shape_;

    int res_cnt = 0;
    // allocate workspace
    if (!init_temp_size_) GetTempSize(ctx);
    Tensor<gpu, 1, SrcType> workspace =
      ctx.requested[res_cnt++].get_space_typed<gpu, 1, SrcType>(mshadow::Shape1(workspace_), s);

    if (param_.layout == mshadow::kNCHW) {
      TBlob data_(ctx.requested[res_cnt++].get_space_typed<gpu, 4, SrcType>(
          mshadow::Shape4(dshape[N], dshape[H], dshape[W], dshape[C]), s));
      TBlob filter_(ctx.requested[res_cnt++].get_space_typed<gpu, 4, SrcType>(
          mshadow::Shape4(fshape[N], fshape[H], fshape[W], fshape[C]), s));

      // input:  [NCHW] => [NHWC](batch, in_height, in_width, in_channels)
      // filter: [NCHW] => [NHWC](out_channels, filter_height, filter_width, in_channels)
      TransposeImpl<gpu>(ctx.run_ctx, data,   data_,   TShape({N, H, W, C}));
      TransposeImpl<gpu>(ctx.run_ctx, filter, filter_, TShape({N, H, W, C}));

      TBlob out_(ctx.requested[res_cnt++].get_space_typed<gpu, 4, DstType>(
          mshadow::Shape4(oshape[N], oshape[H], oshape[W], oshape[C]), s));
      TBlob out_tcast(ctx.requested[res_cnt++].get_space_typed<gpu, 4, int32_t>(
          mshadow::Shape4(oshape[N], oshape[H], oshape[W], oshape[C]), s));
      // input:  [NHWC](batch, in_height, in_width, in_channels)
      // filter: [HWNC](out_channels, filter_height, filter_width, in_channels)
      // output: [NHWC](batch, out_height, out_width, out_channels)

      CUDNN_CALL(cudnnConvolutionForward(s->dnn_handle_,
                                         &alpha_,
                                         data_desc_,
                                         data_.dptr_,
                                         filter_desc_,
                                         filter_.dptr_,
                                         conv_desc_,
                                         algo_,
                                         workspace.dptr_,
                                         workspace_byte_,
                                         &beta_,
                                         out_desc_,
                                         out_.dptr_));

      Tensor<gpu, 1, DstType> out_tensor = out_.FlatTo1D<gpu, DstType>(s);
      Tensor<gpu, 1, int32_t> out_tcast_tensor = out_tcast.FlatTo1D<gpu, int32_t>(s);
      Assign(out_tcast_tensor, kWriteTo, mshadow::expr::tcast<int32_t>(out_tensor));
      // output: [NHWC](batch, out_height, out_width, out_channels) => [NCHW]
      TransposeImpl<gpu>(ctx.run_ctx, out_tcast, out, TShape({0, 3, 1, 2}));
    } else if (param_.layout == mshadow::kNHWC) {
      TBlob out_float(ctx.requested[res_cnt++].get_space_typed<gpu, 4, DstType>(
          mshadow::Shape4(oshape[N], oshape[H], oshape[W], oshape[C]), s));
      CUDNN_CALL(cudnnConvolutionForward(s->dnn_handle_,
                                         &alpha_,
                                         data_desc_,
                                         data.dptr_,
                                         filter_desc_,
                                         filter.dptr_,
                                         conv_desc_,
                                         algo_,
                                         workspace.dptr_,
                                         workspace_byte_,
                                         &beta_,
                                         out_desc_,
                                         out_float.dptr_));
      Tensor<gpu, 1, DstType> out_float_tensor = out_float.FlatTo1D<gpu, DstType>(s);
      Tensor<gpu, 1, int32_t> out_tensor = out.FlatTo1D<gpu, int32_t>(s);
      Assign(out_tensor, kWriteTo, mshadow::expr::tcast<int32_t>(out_float_tensor));
    }

    mxnet_op::Kernel<QuantizationRangeForMultiplicationStruct, gpu>::Launch(s, 1,
      out_data[1].dptr<float>(), out_data[2].dptr<float>(),
       in_data[2].dptr<float>(),  in_data[3].dptr<float>(),
       in_data[4].dptr<float>(),  in_data[5].dptr<float>());
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    LOG(FATAL) << "Not implemented";
  }


  void InitDescriptors(const Context& ctx,
                       const std::vector<TShape>& in_shape,
                       const std::vector<TShape>& out_shape) {
    TShape dshape =  in_shape[0];
    TShape kshape =  in_shape[1];
    TShape oshape = out_shape[0];
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&data_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc_));

    CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_desc_,
                                               param_.pad[0],
                                               param_.pad[1],
                                               param_.stride[0],
                                               param_.stride[1],
                                               1,
                                               1,
                                               CUDNN_CROSS_CORRELATION,
                                               cmp_type_));

    CUDNN_CALL(cudnnSetTensor4dDescriptor(data_desc_,
                                          format_,
                                          src_type_,
                                          dshape[N],
                                          dshape[C],
                                          dshape[H],
                                          dshape[W]));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc_,
                                          format_,
                                          dst_type_,
                                          oshape[N],
                                          oshape[C],
                                          oshape[H],
                                          oshape[W]));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc_,
                                          src_type_,
                                          format_,
                                          kshape[N],
                                          kshape[C],
                                          kshape[H],
                                          kshape[W]));
  }

  void GetTempSize(const OpContext& ctx) {
    CHECK(!init_temp_size_)
      << "GetTempSize should only be called once.";
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(s->dnn_handle_,
                                                       data_desc_,
                                                       filter_desc_,
                                                       conv_desc_,
                                                       out_desc_,
                                                       algo_,
                                                       &workspace_byte_));
    workspace_ = workspace_byte_ / sizeof(SrcType) + 1;
    init_temp_size_ = true;
  }


 private:
  bool init_temp_size_ = false;
  QuantizedConv2DParam param_;
  size_t workspace_;
  size_t workspace_byte_;
  cudnnDataType_t src_type_;
  cudnnDataType_t dst_type_;
  cudnnDataType_t cmp_type_;
  cudnnTensorFormat_t format_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnConvolutionFwdAlgo_t algo_;
  uint32_t N, H, W, C;
  float alpha_ = 1.0f;
  float beta_ = 0.0f;
};  // class QuantizedReluCuDNNOp


template<>
Operator* CreateOp<gpu>(int dtype,
                        const Context& ctx,
                        const std::vector<TShape>& in_shape,
                        const std::vector<TShape>& out_shape,
                        const QuantizedConv2DParam& param) {
  Operator *op = NULL;
  op = new QuantizedConv2DCuDNNOp<int8_t, float, int32_t>(ctx,
    in_shape, out_shape, param);
  return op;
}

}  // namespace op
}  // namespace mxnet

