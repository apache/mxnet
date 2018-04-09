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
 * \file bilinear_resize.cc
 * \brief bilinear resize operator
 * \author Hang Zhang
*/
#include "bilinear_resize-inl.h"
// #include "elemwise_op_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

using namespace mshadow;

template<typename xpu, typename DType, typename AccReal>
void SpatialUpSamplingBilinearUpdateOutput(mshadow::Stream<cpu> *s,
                                           const std::vector<TBlob> &input,
                                           const std::vector<TBlob> &output) {
  Tensor<xpu, 4, DType> itensor = input[0].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> otensor = output[0].get<xpu, 4, DType>(s);
  int nbatch = otensor.size(0);
  int channels = otensor.size(1);
  int outputHeight = otensor.size(2);
  int outputWidth = otensor.size(3);
  int inputHeight = itensor.size(2);
  int inputWidth = itensor.size(3);

  DType *idata = itensor.dptr_;
  DType *odata = otensor.dptr_;
  channels = nbatch * channels;
  // special case: just copy
  if (inputHeight == outputHeight && inputWidth == outputWidth) {
    for (int h2 = 0; h2 < outputHeight; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < outputWidth; ++w2) {
        const int w1 = w2;
        const DType* pos1 = &idata[h1 * inputWidth + w1];
        DType* pos2 = &odata[h2 * outputWidth + w2];
        for (int c = 0; c < channels; ++c) {
          pos2[0] = pos1[0];
          pos1 += inputWidth * inputHeight;
          pos2 += outputWidth * outputHeight;
        }
      }
    }
    return;
  }
  const float rheight =(outputHeight > 1) ? static_cast<float>(inputHeight - 1)/
                       (outputHeight - 1) : 0.f;
  const float rwidth = (outputWidth > 1) ? static_cast<float>(inputWidth - 1) /
                       (outputWidth - 1) : 0.f;
  for (int h2 = 0; h2 < outputHeight; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < inputHeight - 1) ? 1 : 0;
    const DType h1lambda = h1r - h1;
    const DType h0lambda = (DType)1. - h1lambda;
    for (int w2 = 0; w2 < outputWidth; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < inputWidth - 1) ? 1 : 0;
      const DType w1lambda = w1r - w1;
      const DType w0lambda = (DType)1. - w1lambda;
      const DType* pos1 = &idata[h1 * inputWidth + w1];
      DType* pos2 = &odata[h2 * outputWidth + w2];
      for (int c = 0; c < channels; ++c) {
        pos2[0] = h0lambda * (w0lambda * pos1[0]+ w1lambda * pos1[w1p])
                  + h1lambda * (w0lambda * pos1[h1p * inputWidth]
                  + w1lambda * pos1[h1p * inputWidth + w1p]);
        pos1 += inputWidth * inputHeight;
        pos2 += outputWidth * outputHeight;
      }
    }
  }
}


template<typename xpu, typename DType, typename AccReal>
void SpatialUpSamplingBilinearUpdateGradInput(mshadow::Stream<cpu> *s,
                                              const std::vector<TBlob> &input,
                                              const std::vector<TBlob> &output) {
  Tensor<xpu, 4, DType> gradOutput = input[0].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> gradInput = output[0].get<xpu, 4, DType>(s);

  int nbatch = gradInput.size(0);
  int channels = gradInput.size(1);
  int outputHeight = gradOutput.size(2);
  int outputWidth = gradOutput.size(3);
  int inputHeight = gradInput.size(2);
  int inputWidth = gradInput.size(3);

  DType *data1 = gradInput.dptr_;
  DType *data2 = gradOutput.dptr_;
  channels = nbatch * channels;

  // special case: same-size matching grids
  if (inputHeight == outputHeight && inputWidth == outputWidth) {
    for (int h2 = 0; h2 < outputHeight; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < outputWidth; ++w2) {
        const int w1 = w2;
        DType* pos1 = &data1[h1 * inputWidth + w1];
        const DType* pos2 = &data2[h2 * outputWidth + w2];
        for (int c = 0; c < channels; ++c) {
          pos1[0] += pos2[0];
          pos1 += inputWidth * inputHeight;
          pos2 += outputWidth * outputHeight;
        }
      }
    }
    return;
  }
  const float rheight =(outputHeight > 1) ? static_cast<float>(inputHeight - 1)/
                       (outputHeight - 1) : 0.f;
  const float rwidth = (outputWidth > 1) ? static_cast<float>(inputWidth - 1)/
                       (outputWidth - 1) : 0.f;
  for (int h2 = 0; h2 < outputHeight; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < inputHeight - 1) ? 1 : 0;
    const DType h1lambda = h1r - h1;
    const DType h0lambda = (DType)1. - h1lambda;
    for (int w2 = 0; w2 < outputWidth; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < inputWidth - 1) ? 1 : 0;
      const DType w1lambda = w1r - w1;
      const DType w0lambda = (DType)1. - w1lambda;
      DType* pos1 = &data1[h1 * inputWidth + w1];
      const DType* pos2 = &data2[h2 * outputWidth + w2];
      for (int c = 0; c < channels; ++c) {
        pos1[0] += h0lambda * w0lambda * pos2[0];
        pos1[w1p] += h0lambda * w1lambda * pos2[0];
        pos1[h1p * inputWidth] += h1lambda * w0lambda * pos2[0];
        pos1[h1p * inputWidth + w1p] += h1lambda * w1lambda * pos2[0];
        pos1 += inputWidth * inputHeight;
        pos2 += outputWidth * outputHeight;
      }
    }
  }
}


DMLC_REGISTER_PARAMETER(BilinearSampleParam);

NNVM_REGISTER_OP(_contrib_BilinearResize2D)
.describe(R"code(
Perform 2D resizing (upsampling or downsampling) for 4D input using bilinear interpolation.

Expected input is a 4 dimensional NDArray (NCHW) and the output
with the shape of (N x C x height x width). 
The key idea of bilinear interpolation is to perform linear interpolation
first in one direction, and then again in the other direction. See the wikipedia of
`Bilinear interpolation  <https://en.wikipedia.org/wiki/Bilinear_interpolation>`_
for more details.
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<BilinearSampleParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", BilinearSampleOpInferShape)
.set_attr<nnvm::FInferType>("FInferType", BilinearSampleOpInferType)
.set_attr<FInferStorageType>("FInferStorageType", BilinearSampleOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", BilinearSampleOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  ElemwiseGradUseNone{"_backward_contrib_BilinearResize2D"})
.add_argument("data", "NDArray-or-Symbol", "Input data")
.add_arguments(BilinearSampleParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_contrib_BilinearResize2D)
.set_attr_parser(ParamParser<BilinearSampleParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", BilinearSampleOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", BilinearSampleOpBackward<cpu>);


}  // namespace op
}  // namespace mxnet
