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
 * Copyright (c) 2018 by Contributors
 * \file adaptive_avg_pooling.cc
 * \brief adaptive average pooling operator
 * \author Hang Zhang
*/
#include "adaptive_avg_pooling-inl.h"
// #include "elemwise_op_common.h"
#include "../elemwise_op_common.h"

#define START_IND(a, b, c) static_cast<int>(std::floor(static_cast<float>(a * c) / b))
#define END_IND(a, b, c) static_cast<int>(std::ceil(static_cast<float>((a + 1) * c) / b))

namespace mxnet {
namespace op {

using namespace mshadow;

template<typename real>
static void SpatialAdaptiveAveragePooling_updateOutput_frame(
          real *input_p,
          real *output_p,
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
    int outOffset = d*osizeH*osizeW;
    for (oh = 0; oh < osizeH; oh++) {
      int istartH = START_IND(oh, osizeH, isizeH);
      int startOffsetH = istartH * istrideH;
      int outOffsetH = oh * osizeW;
      int iendH   = END_IND(oh, osizeH, isizeH);
      int kH = iendH - istartH;

      for (ow = 0; ow < osizeW; ow++) {
        int istartW = START_IND(ow, osizeW, isizeW);
        int iendW   = END_IND(ow, osizeW, isizeW);
        int kW = iendW - istartW;

        /* local pointers */
        real *ip = input_p   + d*istrideD + startOffsetH + istartW*istrideW;
        real *op = output_p  + outOffset + outOffsetH + ow;

        /* compute local average: */
        real sum = 0;
        for (ih = 0; ih < kH; ih++) {
          int ihOffset = ih*istrideH;
          for (iw = 0; iw < kW; iw++) {
            real val = *(ip + ihOffset + iw*istrideW);
            sum += val;
          }
        }

        /* set output to local average */
        *op = sum / kW / kH;
      }
    }
  }
}

template<typename real>
static void SpatialAdaptiveAveragePooling_updateGradInput_frame(
          real *gradInput_p,
          real *gradOutput_p,
          int64_t sizeD,
          int64_t isizeH,
          int64_t isizeW,
          int64_t osizeH,
          int64_t osizeW) {
  int64_t d;
#pragma omp parallel for private(d) \
num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (d = 0; d < sizeD; d++) {
    real *gradInput_p_d = gradInput_p + d*isizeW*isizeH;
    real *gradOutput_p_d = gradOutput_p + d*osizeW*osizeH;

    /* calculate average */
    int64_t oh, ow;
    for (oh = 0; oh < osizeH; oh++) {
      int istartH = START_IND(oh, osizeH, isizeH);
      int iendH   = END_IND(oh, osizeH, isizeH);
      int kH = iendH - istartH;

      for (ow = 0; ow < osizeW; ow++) {
        int istartW = START_IND(ow, osizeW, isizeW);
        int iendW   = END_IND(ow, osizeW, isizeW);
        int kW = iendW - istartW;

        real grad_delta = gradOutput_p_d[oh*osizeW +ow] / kH / kW;

        int ih, iw;
        for (ih = istartH; ih < iendH; ih++) {
          for (iw = istartW; iw < iendW; iw++) {
            /* update gradient */
            gradInput_p_d[ih*isizeW + iw] += grad_delta;
          }
        }
      }
    }
  }
}


template<typename xpu, typename DType, typename AccReal>
void AdaptiveAvgPoolUpdateOutput(mshadow::Stream<cpu> *s,
                                           const std::vector<TBlob> &input,
                                           const std::vector<TBlob> &output) {
  Tensor<xpu, 4, DType> itensor = input[0].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> otensor = output[0].get<xpu, 4, DType>(s);

  DType *input_data = itensor.dptr_;
  DType *output_data = otensor.dptr_;

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
      input_data+b*istrideB, output_data+b*sizeD*osizeH*osizeW,
      sizeD,
      isizeH, isizeW,
      osizeH, osizeW,
      istrideD,
      istrideH, istrideW);
  }
}


template<typename xpu, typename DType, typename AccReal>
void AdaptiveAvgPoolUpdateGradInput(mshadow::Stream<cpu> *s,
                                              const std::vector<TBlob> &input,
                                              const std::vector<TBlob> &output) {
  Tensor<xpu, 4, DType> gradOut = input[0].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> gradIn = output[0].get<xpu, 4, DType>(s);

  DType *gradOutput_data = gradOut.dptr_;
  DType *gradInput_data = gradIn.dptr_;

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
      gradInput_data+b*sizeD*isizeH*isizeW, gradOutput_data+b*sizeD*osizeH*osizeW,
      sizeD,
      isizeH, isizeW,
      osizeH, osizeW);
  }
}


DMLC_REGISTER_PARAMETER(AdaptiveAvgPoolParam);

NNVM_REGISTER_OP(_contrib_AdaptiveAvgPooling2D)
.describe(R"code(
Applies a 2D adaptive average pooling over a 4D input with the shape of (NCHW).
The pooling kernel and stride sizes are automatically chosen for desired output sizes.

- If a single integer is provided for output_size, the output size is \
  (N x C x output_size x output_size) for any input (NCHW).

- If a tuple of integers (height, width) are provided for output_size, the output size is \
  (N x C x height x width) for any input (NCHW).

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<AdaptiveAvgPoolParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", AdaptiveAvgPoolOpInferShape)
.set_attr<FCompute>("FCompute<cpu>", AdaptiveAvgPoolOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  ElemwiseGradUseNone{"_backward_contrib_AdaptiveAvgPooling2D"})
.add_argument("data", "NDArray-or-Symbol", "Input data")
.add_arguments(AdaptiveAvgPoolParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_contrib_AdaptiveAvgPooling2D)
.set_attr_parser(ParamParser<AdaptiveAvgPoolParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", AdaptiveAvgPoolOpBackward<cpu>);


}  // namespace op
}  // namespace mxnet
