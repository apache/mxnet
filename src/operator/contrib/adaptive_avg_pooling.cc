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
 * \file adaptive_avg_pooling.cc
 * \brief adaptive average pooling operator
 * \author Hang Zhang
 */
#include "adaptive_avg_pooling-inl.h"
// #include "elemwise_op_common.h"
#include "../elemwise_op_common.h"
#if MXNET_USE_ONEDNN == 1
#include "../nn/dnnl/dnnl_base-inl.h"
#include "../nn/dnnl/dnnl_pooling-inl.h"
#endif  // MXNET_USE_ONEDNN

#define START_IND(a, b, c) static_cast<int>(std::floor(static_cast<float>(a * c) / b))
#define END_IND(a, b, c)   static_cast<int>(std::ceil(static_cast<float>((a + 1) * c) / b))
#define DIV_ROUND_UP(a, b) ((a + (b - 1)) / b)

namespace mxnet {
namespace op {

using namespace mshadow;

template <typename real>
static void SpatialAdaptiveAveragePooling_updateOutput_frame(real* input_p,
                                                             real* output_p,
                                                             int64_t sizeD,
                                                             int64_t isizeH,
                                                             int64_t isizeW,
                                                             int64_t osizeH,
                                                             int64_t osizeW,
                                                             int64_t istrideD,
                                                             int64_t istrideH,
                                                             int64_t istrideW) {
  int64_t d;
#pragma omp parallel for private(d) \
    num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (d = 0; d < sizeD; d++) {
    /* loop over output */
    int64_t oh, ow, ih, iw;
    int outOffset = d * osizeH * osizeW;
    for (oh = 0; oh < osizeH; oh++) {
      int istartH      = START_IND(oh, osizeH, isizeH);
      int startOffsetH = istartH * istrideH;
      int outOffsetH   = oh * osizeW;
      int iendH        = END_IND(oh, osizeH, isizeH);
      int kH           = iendH - istartH;

      for (ow = 0; ow < osizeW; ow++) {
        int istartW = START_IND(ow, osizeW, isizeW);
        int iendW   = END_IND(ow, osizeW, isizeW);
        int kW      = iendW - istartW;

        /* local pointers */
        real* ip = input_p + d * istrideD + startOffsetH + istartW * istrideW;
        real* op = output_p + outOffset + outOffsetH + ow;

        /* compute local average: */
        real sum = 0;
        for (ih = 0; ih < kH; ih++) {
          int ihOffset = ih * istrideH;
          for (iw = 0; iw < kW; iw++) {
            real val = *(ip + ihOffset + iw * istrideW);
            sum += val;
          }
        }

        /* set output to local average */
        *op = sum / kW / kH;
      }
    }
  }
}

template <typename real>
static void SpatialAdaptiveAveragePooling_updateGradInput_frame(real* gradInput_p,
                                                                real* gradOutput_p,
                                                                int64_t sizeD,
                                                                int64_t isizeH,
                                                                int64_t isizeW,
                                                                int64_t osizeH,
                                                                int64_t osizeW) {
  int64_t d;
#pragma omp parallel for private(d) \
    num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (d = 0; d < sizeD; d++) {
    real* gradInput_p_d  = gradInput_p + d * isizeW * isizeH;
    real* gradOutput_p_d = gradOutput_p + d * osizeW * osizeH;

    /* calculate average */
    int64_t oh, ow;
    for (oh = 0; oh < osizeH; oh++) {
      int istartH = START_IND(oh, osizeH, isizeH);
      int iendH   = END_IND(oh, osizeH, isizeH);
      int kH      = iendH - istartH;

      for (ow = 0; ow < osizeW; ow++) {
        int istartW = START_IND(ow, osizeW, isizeW);
        int iendW   = END_IND(ow, osizeW, isizeW);
        int kW      = iendW - istartW;

        real grad_delta = gradOutput_p_d[oh * osizeW + ow] / kH / kW;

        int ih, iw;
        for (ih = istartH; ih < iendH; ih++) {
          for (iw = istartW; iw < iendW; iw++) {
            /* update gradient */
            gradInput_p_d[ih * isizeW + iw] += grad_delta;
          }
        }
      }
    }
  }
}

template <typename xpu, typename DType, typename AccReal>
void AdaptiveAvgPoolUpdateOutput(mshadow::Stream<cpu>* s,
                                 const std::vector<TBlob>& input,
                                 const std::vector<TBlob>& output) {
  Tensor<xpu, 4, DType> itensor = input[0].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> otensor = output[0].get<xpu, 4, DType>(s);

  DType* input_data  = itensor.dptr_;
  DType* output_data = otensor.dptr_;

  int64_t sizeB  = itensor.size(0);
  int64_t sizeD  = itensor.size(1);
  int64_t isizeH = itensor.size(2);
  int64_t isizeW = itensor.size(3);

  int64_t istrideB = get_stride<xpu, 4, DType>(itensor, 0);
  int64_t istrideD = get_stride<xpu, 4, DType>(itensor, 1);
  int64_t istrideH = get_stride<xpu, 4, DType>(itensor, 2);
  int64_t istrideW = get_stride<xpu, 4, DType>(itensor, 3);

  int64_t osizeH = otensor.size(2);
  int64_t osizeW = otensor.size(3);

  int64_t b;
#pragma omp parallel for private(b) \
    num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (b = 0; b < sizeB; b++) {
    SpatialAdaptiveAveragePooling_updateOutput_frame<DType>(
        input_data + b * istrideB,
        output_data + b * sizeD * osizeH * osizeW,
        sizeD,
        isizeH,
        isizeW,
        osizeH,
        osizeW,
        istrideD,
        istrideH,
        istrideW);
  }
}

#if MXNET_USE_ONEDNN == 1
// Support for https://oneapi-src.github.io/oneDNN/v2.6/dev_guide_pooling.html
bool SupportDNNLAveragePooling(const NDArray& input, const NDArray& output) {
  for (int64_t idx = 2; idx < input.shape().ndim(); ++idx) {
    const int s1 = input.shape()[idx];
    const int s2 = output.shape()[idx];
    if (s2 == 0) {
      return false;
    }
    if (s1 % s2 != 0) {
      return false;
    }
  }
  const int IH         = input.shape()[2];
  const int IW         = input.shape()[3];
  const int OH         = output.shape()[2];
  const int OW         = output.shape()[3];
  const int strides_H  = ((IH << 1) / OH) - (IH / OH);
  const int strides_W  = ((IW << 1) / OW) - (IW / OW);
  const int kernel_H   = DIV_ROUND_UP((IH << 1) / OH, 1) - (IH / OH);
  const int kernel_W   = DIV_ROUND_UP((IW << 1) / OW, 1) - (IW / OW);
  const int pad_l_top  = (strides_H * (OH - 1) + kernel_H - IH) / 2;
  const int pad_l_left = (strides_W * (OW - 1) + kernel_W - IW) / 2;

  return SupportDNNL<3, 5, DNNLTypeMode::AllTypes>(input) && pad_l_top == 0 && pad_l_left == 0;
}

void AdaptiveAvgPoolOpBackwardExCPU(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<NDArray>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);

  if (SupportDNNLAveragePooling(outputs[0], inputs[0])) {
    DNNL_OPCHECK_INIT(true, outputs.size(), inputs, outputs);
    DNNLRun(DNNLPoolingGradCompute, attrs, ctx, inputs, req, outputs);
    DNNL_OPCHECK_RUN(AdaptiveAvgPoolOpBackward<cpu>, attrs, ctx, inputs, req, outputs);
    return;
  }
  FallBackCompute(AdaptiveAvgPoolOpBackward<cpu>, attrs, ctx, inputs, req, outputs);
}

inline static bool BackwardAdaptivePoolingStorageType(const nnvm::NodeAttrs& attrs,
                                                      const int dev_mask,
                                                      DispatchMode* dispatch_mode,
                                                      std::vector<int>* in_attrs,
                                                      std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);

  // support_dnnl is set to true, because at this point there is no way
  // to check if DNNLAdaptivePooling is supported
  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

void AdaptiveAvgPoolComputeExCPU(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<NDArray>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  /*
  oneDNN doesn't support adaptive pooling.
  Fallback is needed when padding is not equal 0;
  */
  if (SupportDNNLAveragePooling(inputs[0], outputs[0])) {
    DNNL_OPCHECK_INIT(false, 1, inputs, outputs);
    DNNLRun(DNNLPoolingCompute, attrs, ctx, inputs, req, outputs);
    DNNL_OPCHECK_RUN(PoolingCompute<cpu>, attrs, ctx, inputs, req, outputs);
    return;
  }
  FallBackCompute(AdaptiveAvgPoolOpForward<cpu>, attrs, ctx, inputs, req, outputs);
}

inline static bool AdaptivePoolingStorageType(const nnvm::NodeAttrs& attrs,
                                              const int dev_mask,
                                              DispatchMode* dispatch_mode,
                                              std::vector<int>* in_attrs,
                                              std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);

  // support_dnnl is set to true, because at this point there is no way
  // to check if DNNLAdaptivePooling is supported
  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}
#endif

template <typename xpu, typename DType, typename AccReal>
void AdaptiveAvgPoolUpdateGradInput(mshadow::Stream<cpu>* s,
                                    const std::vector<TBlob>& input,
                                    const std::vector<TBlob>& output) {
  Tensor<xpu, 4, DType> gradOut = input[0].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> gradIn  = output[0].get<xpu, 4, DType>(s);

  DType* gradOutput_data = gradOut.dptr_;
  DType* gradInput_data  = gradIn.dptr_;

  int64_t sizeB  = gradIn.size(0);
  int64_t sizeD  = gradIn.size(1);
  int64_t isizeH = gradIn.size(2);
  int64_t isizeW = gradIn.size(3);

  int64_t osizeH = gradOut.size(2);
  int64_t osizeW = gradOut.size(3);

  int64_t b;
#pragma omp parallel for private(b) \
    num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (b = 0; b < sizeB; b++) {
    SpatialAdaptiveAveragePooling_updateGradInput_frame<DType>(
        gradInput_data + b * sizeD * isizeH * isizeW,
        gradOutput_data + b * sizeD * osizeH * osizeW,
        sizeD,
        isizeH,
        isizeW,
        osizeH,
        osizeW);
  }
}

NNVM_REGISTER_OP(_contrib_AdaptiveAvgPooling2D)
    .describe(R"code(
Applies a 2D adaptive average pooling over a 4D input with the shape of (NCHW).
The pooling kernel and stride sizes are automatically chosen for desired output sizes.

- If a single integer is provided for output_size, the output size is \
  (N x C x output_size x output_size) for any input (NCHW).

- If a tuple of integers (height, width) are provided for output_size, the output size is \
  (N x C x height x width) for any input (NCHW).

)code" ADD_FILELINE)
    .set_attr_parser(PoolingParamParser)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<mxnet::FInferShape>("FInferShape", AdaptiveAvgPoolOpInferShape)
    .set_attr<FCompute>("FCompute<cpu>", AdaptiveAvgPoolOpForward<cpu>)
    .set_attr<nnvm::FGradient>("FGradient",
                               ElemwiseGradUseNone{"_backward_contrib_AdaptiveAvgPooling2D"})
#if MXNET_USE_ONEDNN == 1
    .set_attr<FInferStorageType>("FInferStorageType", AdaptivePoolingStorageType)
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", AdaptiveAvgPoolComputeExCPU)
#endif
    .add_argument("data", "NDArray-or-Symbol", "Input data")
    .add_arguments(PoolingParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_contrib_AdaptiveAvgPooling2D)
    .set_attr_parser(PoolingParamParser)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
#if MXNET_USE_ONEDNN == 1
    .set_attr<FInferStorageType>("FInferStorageType", BackwardAdaptivePoolingStorageType)
    // Different backend requires different FInplaceOption
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      const PoolingParam& param =
                                          nnvm::get<PoolingParam>(attrs.parsed);
                                      if (DNNLRequireWorkspace(param) && param.IsAdaptivePooling())
                                        return std::vector<std::pair<int, int>>{{1, 0}};
                                      return std::vector<std::pair<int, int>>();
                                    })
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", AdaptiveAvgPoolOpBackwardExCPU)
#else
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int>>();
                                    })
#endif
    .set_attr<FCompute>("FCompute<cpu>", AdaptiveAvgPoolOpBackward<cpu>);

}  // namespace op
}  // namespace mxnet
