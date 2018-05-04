/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_conv.cu
 * \brief
 * \author Ziheng Jiang, Jun Wu
*/
#include "../nn/convolution-inl.h"
#include "./quantization_utils.h"
#include "../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {

// value + bias_value * (range1 / limit_range1) * (limit_range2 / range2)
struct QuantizedBiasAddKernel {
  MSHADOW_XINLINE static void Map(int i, size_t bias_size, int32_t *out,
                                  const int8_t *bias, const float *min_out,
                                  const float *max_out, const float *min_bias,
                                  const float *max_bias, const size_t spatial_size) {
    using mshadow::red::limits::MinValue;
    using mshadow::red::limits::MaxValue;
    float float_for_one_out_quant  =
      MaxAbs(*min_out, *max_out) / static_cast<double>(MaxValue<int32_t>());
    float float_for_one_bias_quant =
      MaxAbs(*min_bias, *max_bias) / static_cast<double>(MaxValue<int8_t>());
    const size_t channel_id = (i / spatial_size) % bias_size;
    out[i] = (out[i] * float_for_one_out_quant +
              bias[channel_id] * float_for_one_bias_quant) /
             float_for_one_out_quant;
  }
};

#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 6 && CUDA_VERSION >= 8000
template<typename SrcType, typename DstType, typename CmpType>
class QuantizedCuDNNConvOp {
 public:
  QuantizedCuDNNConvOp() {
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&data_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc_));
  }

  void Init(const ConvolutionParam& param,
            const OpContext& ctx,
            const std::vector<TShape>& in_shape,
            const std::vector<TShape>& out_shape) {
    param_ = param;
    CHECK_EQ(param_.kernel.ndim(), 2U)
      << "QuantizedCuDNNConvOp only supports 2D convolution for now";
    if (param_.layout.has_value()) {
      CHECK_EQ(param_.layout.value(), mshadow::kNCHW)
        << "QuantizedConvOp only supports NCHW for now";
    }
    if (param_.stride.ndim() == 0U) param_.stride = mshadow::Shape2(1, 1);
    if (param_.dilate.ndim() == 0U) param_.dilate = mshadow::Shape2(1, 1);
    if (param_.pad.ndim() == 0U)    param_.pad = mshadow::Shape2(0, 0);
    N = 0, H = 2, W = 3, C = 1;
    src_type_ = mshadow::DataType<SrcType>::kCudnnFlag;
    dst_type_ = mshadow::DataType<DstType>::kCudnnFlag;
    cmp_type_ = mshadow::DataType<CmpType>::kCudnnFlag;
    algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    format_ = CUDNN_TENSOR_NHWC;
    InitDescriptors(in_shape, out_shape);
    GetTempSize(ctx);
  }

  ~QuantizedCuDNNConvOp() {
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc_));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    CHECK_EQ(param_.kernel.ndim(), 2U)
      << "QuantizedCuDNNConvOp only supports 2D convolution for now";
    using namespace mshadow;
    CHECK_EQ(in_data.size(), param_.no_bias? 6U : 9U);
    CHECK_EQ(out_data.size(), 3U);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->dnn_handle_ownership_, Stream<gpu>::OwnHandle);

    const TBlob& data   = in_data[0];
    const TBlob& filter = in_data[1];
    const TBlob& out    = out_data[0];
    const TShape& dshape = data.shape_;
    const TShape& fshape = filter.shape_;
    const TShape& oshape = out.shape_;

    // allocate workspace
    const int dev_id = ctx.run_ctx.ctx.dev_id;
    const int dev_mask = gpu::kDevMask;
    if (!param_.layout.has_value() || param_.layout.value() == mshadow::kNCHW) {
      const size_t data_size = dshape.Size();
      const size_t weight_size = fshape.Size();
      const size_t output_size = oshape.Size();
      size_t total_temp_bytes = (workspace_ + data_size + weight_size) * sizeof(SrcType)
                              + output_size * (sizeof(DstType) + sizeof(int32_t));
      Tensor<gpu, 1, char> temp_space =
        ctx.requested[0].get_space_typed<gpu, 1, char>(mshadow::Shape1(total_temp_bytes), s);
      char* temp_dptr = temp_space.dptr_;
      TBlob data_(reinterpret_cast<SrcType*>(temp_dptr),
                  TShape({dshape[N], dshape[H], dshape[W], dshape[C]}),
                  dev_mask, DataType<SrcType>::kFlag, dev_id);
      temp_dptr += data_size * sizeof(SrcType);
      TBlob filter_(reinterpret_cast<SrcType*>(temp_dptr),
                    TShape({fshape[N], fshape[H], fshape[W], fshape[C]}),
                    dev_mask, DataType<SrcType>::kFlag, dev_id);
      temp_dptr += weight_size * sizeof(SrcType);

      // input:  [NCHW] => [NHWC](batch, in_height, in_width, in_channels)
      // filter: [NCHW] => [NHWC](out_channels, filter_height, filter_width, in_channels)
      TransposeImpl<gpu>(ctx.run_ctx, data,   data_,   TShape({N, H, W, C}));
      TransposeImpl<gpu>(ctx.run_ctx, filter, filter_, TShape({N, H, W, C}));
      TBlob out_(reinterpret_cast<DstType*>(temp_dptr),
                 TShape({oshape[N], oshape[H], oshape[W], oshape[C]}),
                 dev_mask, DataType<DstType>::kFlag, dev_id);
      temp_dptr += output_size * sizeof(DstType);
      TBlob out_tcast(reinterpret_cast<int32_t*>(temp_dptr),
                      TShape({oshape[N], oshape[H], oshape[W], oshape[C]}),
                      dev_mask, DataType<int32_t>::kFlag, dev_id);
      temp_dptr += output_size * sizeof(int32_t);
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
                                         temp_dptr,
                                         workspace_byte_,
                                         &beta_,
                                         out_desc_,
                                         out_.dptr_));

      Tensor<gpu, 1, DstType> out_tensor = out_.FlatTo1D<gpu, DstType>(s);
      Tensor<gpu, 1, int32_t> out_tcast_tensor = out_tcast.FlatTo1D<gpu, int32_t>(s);
      Assign(out_tcast_tensor, kWriteTo, mshadow::expr::tcast<int32_t>(out_tensor));
      // output: [NHWC](batch, out_height, out_width, out_channels) => [NCHW]
      TransposeImpl<gpu>(ctx.run_ctx, out_tcast, out, TShape({0, 3, 1, 2}));
    } else {
      LOG(FATAL) << "quantized_conv only supports NCHW for now";
    }

    // calculate the min/max range for out_data as it's a multiplication
    // of in_data[0] and in_data[1]. Need to rescale the min/max range of out_data
    // based on the min/max ranges of in_data[0] and in_data[1].
    const size_t num_inputs = param_.no_bias ? 2 : 3;
    mxnet_op::Kernel<QuantizationRangeForMultiplicationStruct, gpu>::Launch(s, 1,
      out_data[1].dptr<float>(), out_data[2].dptr<float>(),
       in_data[num_inputs].dptr<float>(),  in_data[num_inputs+1].dptr<float>(),
       in_data[num_inputs+2].dptr<float>(),  in_data[num_inputs+3].dptr<float>());

    if (!param_.no_bias) {
      if (param_.layout.has_value()) {
        CHECK_EQ(param_.layout.value(), mshadow::kNCHW)
          << "quantized_conv only supports NCHW when there is a bias";
      }
      const TBlob& bias = in_data[2];
      mxnet_op::Kernel<QuantizedBiasAddKernel, gpu>::Launch(s, out.Size(),
          bias.Size(), out.dptr<int32_t>(), bias.dptr<int8_t>(),
          out_data[1].dptr<float>(), out_data[2].dptr<float>(),
          in_data[7].dptr<float>(),  in_data[8].dptr<float>(),
          oshape[2] * oshape[3]);
    }
  }

  void InitDescriptors(const std::vector<TShape>& in_shape,
                       const std::vector<TShape>& out_shape) {
    const TShape& dshape =  in_shape[0];
    const TShape& kshape =  in_shape[1];
    const TShape& oshape = out_shape[0];
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
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(s->dnn_handle_,
                                                       data_desc_,
                                                       filter_desc_,
                                                       conv_desc_,
                                                       out_desc_,
                                                       algo_,
                                                       &workspace_byte_));
    workspace_ = workspace_byte_ / sizeof(SrcType) + 1;
  }

 private:
  ConvolutionParam param_;
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
};  // class QuantizedCuDNNConvOp
#endif  // MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 6 && CUDA_VERSION >= 8000

void QuantizedConvForwardGPU(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  CHECK_EQ(param.kernel.ndim(), 2U)
    << "QuantizedConvForward<gpu> only supports 2D convolution for now";
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 6 && CUDA_VERSION >= 8000
  typedef QuantizedCuDNNConvOp<int8_t, float, int32_t> QuantizedConvOpInt8;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local QuantizedConvOpInt8 op;
#else
  static MX_THREAD_LOCAL QuantizedConvOpInt8 op;
#endif  // DMLC_CXX11_THREAD_LOCAL
  op.Init(param, ctx, {inputs[0].shape_, inputs[1].shape_}, {outputs[0].shape_});
  op.Forward(ctx, inputs, req, outputs);
#else
  LOG(FATAL) << "QuantizedConvForward<gpu> only supports cudnnConvolutionForward "
                "with CUDNN >= 6.0 and CUDA >= 8.0";
#endif  // MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 6 && CUDA_VERSION >= 8000
}

NNVM_REGISTER_OP(_contrib_quantized_conv)
.set_attr<FCompute>("FCompute<gpu>", QuantizedConvForwardGPU);

}  // namespace op
}  // namespace mxnet
