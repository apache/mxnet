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
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

using namespace mshadow;

template<typename xpu, typename DType, typename AccReal>
void SpatialUpSamplingBilinearUpdateOutput(mshadow::Stream<cpu> *s,
                                           const std::vector<TBlob> &input,
                                           const std::vector<TBlob> &output,
                                           bool align_corners) {
  Tensor<xpu, 4, DType> itensor = input[0].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> otensor = output[0].get<xpu, 4, DType>(s);
  int nbatch = otensor.size(0);
  int channels = otensor.size(1);
  int outputHeight = otensor.size(2);
  int outputWidth = otensor.size(3);
  int inputHeight = itensor.size(2);
  int inputWidth = itensor.size(3);

  const auto nthreads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();

  DType *idata = itensor.dptr_;
  DType *odata = otensor.dptr_;
  channels = nbatch * channels;
  const int input_elems_per_channel = inputWidth * inputHeight;
  const int output_elems_per_channel = outputWidth * outputHeight;

  // special case: just copy
  if (inputHeight == outputHeight && inputWidth == outputWidth) {
#pragma omp parallel for num_threads(nthreads)
    for (int index = 0; index < output_elems_per_channel; index++) {
      const int h2 = index / outputWidth;
      const int h1 = h2;
      const int w2 = index % outputWidth;
      const int w1 = w2;
      const DType* pos1 = &idata[h1 * inputWidth + w1];
      DType* pos2 = &odata[index];
      for (int c = 0; c < channels; ++c) {
        *pos2 = *pos1;
        pos1 += input_elems_per_channel;
        pos2 += output_elems_per_channel;
      }
    }
    return;
  }
  const float rheight = area_pixel_compute_scale<float>(
    inputHeight, outputHeight, align_corners);
  const float rwidth = area_pixel_compute_scale<float>(
    inputWidth, outputWidth, align_corners);

#pragma omp parallel for num_threads(nthreads)
  for (int index = 0; index < output_elems_per_channel; index++) {
    const int h2 = index / outputWidth;
    const int w2 = index % outputWidth;

  const float h1r = area_pixel_compute_source_index<float>(
    rheight, h2, align_corners, false);
    const int h1 = h1r;
    const int h1p = (h1 < inputHeight - 1) ? 1 : 0;
    const DType h1lambda = h1r - h1;
    const DType h0lambda = (DType)1. - h1lambda;

  const float w1r = area_pixel_compute_source_index<float>(
    rwidth, w2, align_corners, false);
    const int w1 = w1r;
    const int w1p = (w1 < inputWidth - 1) ? 1 : 0;
    const DType w1lambda = w1r - w1;
    const DType w0lambda = (DType)1. - w1lambda;
    const DType* pos1 = &idata[h1 * inputWidth + w1];
    DType* pos2 = &odata[index];

    for (int c = 0; c < channels; ++c) {
      *pos2 = h0lambda * (w0lambda * (*pos1) + w1lambda * *(pos1 + w1p))
            + h1lambda * (w0lambda * *(pos1 + h1p * inputWidth)
            + w1lambda * *(pos1 + h1p * inputWidth + w1p));
      pos1 += input_elems_per_channel;
      pos2 += output_elems_per_channel;
    }
  }
}

template<typename xpu, typename DType, typename AccReal>
void SpatialUpSamplingBilinearUpdateGradInput(mshadow::Stream<cpu> *s,
                                              const std::vector<TBlob> &input,
                                              const std::vector<TBlob> &output,
                                              bool modeLike,
                                              bool align_corners) {
  Tensor<xpu, 4, DType> gradOutput = input[0].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> gradInput = output[0].get<xpu, 4, DType>(s);

  int nbatch = gradInput.size(0);
  int channels = gradInput.size(1);
  int outputHeight = gradOutput.size(2);
  int outputWidth = gradOutput.size(3);
  int inputHeight = gradInput.size(2);
  int inputWidth = gradInput.size(3);

  const auto nthreads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();

  DType *dataInput = gradInput.dptr_;
  DType *dataOutput = gradOutput.dptr_;
  channels = nbatch * channels;
  const int input_elems_per_channel = inputWidth * inputHeight;
  const int output_elems_per_channel = outputWidth * outputHeight;

  // special case: same-size matching grids
  if (inputHeight == outputHeight && inputWidth == outputWidth) {
#pragma omp parallel for num_threads(nthreads)
    for (int index = 0; index < output_elems_per_channel; index++) {
      const int h2 = index / outputWidth;
      const int h1 = h2;
      const int w2 = index % outputWidth;
      const int w1 = w2;
      DType* pos1 = &dataInput[h1 * inputWidth + w1];
      const DType* pos2 = &dataOutput[index];
      for (int c = 0; c < channels; ++c) {
        *pos1 += *pos2;
        pos1 += input_elems_per_channel;
        pos2 += output_elems_per_channel;
      }
    }
    return;
  }
  const float rheight = area_pixel_compute_scale<float>(
    inputHeight, outputHeight, align_corners);
  const float rwidth = area_pixel_compute_scale<float>(
    inputWidth, outputWidth, align_corners);
#pragma omp parallel for num_threads(nthreads)
  for (int index = 0; index < output_elems_per_channel; index++) {
    const int h2 = index / outputWidth;
    const int w2 = index % outputWidth;

    const float h1r = area_pixel_compute_source_index<float>(
        rheight, h2, align_corners, false);
    const int h1 = h1r;
    const int h1p = (h1 < inputHeight - 1) ? 1 : 0;
    const DType h1lambda = h1r - h1;
    const DType h0lambda = (DType)1. - h1lambda;

    const float w1r = area_pixel_compute_source_index<float>(
        rwidth, w2, align_corners, false);
    const int w1 = w1r;
    const int w1p = (w1 < inputWidth - 1) ? 1 : 0;
    const DType w1lambda = w1r - w1;
    const DType w0lambda = (DType)1. - w1lambda;

    DType* posInput = &dataInput[h1 * inputWidth + w1];
    const DType* posOutput = &dataOutput[index];
    for (int c = 0; c < channels; ++c) {
      #pragma omp critical
      {
        *posInput += h0lambda * w0lambda * (*posOutput);
        *(posInput + w1p) += h0lambda * w1lambda * (*posOutput);
        *(posInput + h1p * inputWidth) += h1lambda * w0lambda * (*posOutput);
        *(posInput + h1p * inputWidth + w1p) += h1lambda * w1lambda * (*posOutput);
      }
      posInput += input_elems_per_channel;
      posOutput += output_elems_per_channel;
    }
  }

  if (modeLike) {
    Tensor<xpu, 4, DType> gradInputLike = output[1].get<xpu, 4, DType>(s);
    int inputHeightLike = gradInputLike.size(2);
    int inputWidthLike = gradInputLike.size(3);
    DType *dataInputLike = gradInputLike.dptr_;
    int channelsLike = nbatch * gradInputLike.size(1);

    const int inputLike_elems_per_channel = inputHeightLike * inputWidthLike;
#pragma omp parallel for num_threads(nthreads)
    for (int index = 0; index < inputLike_elems_per_channel; index++) {
      DType *posInput = &dataInputLike[index];
      for (int c = 0; c < channelsLike; ++c) {
        *posInput = 0;
        posInput += inputLike_elems_per_channel;
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
.set_num_inputs(BilinearSampleOpNumInputs)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames", BilinearSampleOpInputNames)
.set_attr<mxnet::FInferShape>("FInferShape", BilinearSampleOpInferShape)
.set_attr<FCompute>("FCompute<cpu>", BilinearSampleOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  ElemwiseGradUseNone{"_backward_contrib_BilinearResize2D"})
.add_argument("data", "NDArray-or-Symbol", "Input data")
.add_argument("like", "NDArray-or-Symbol", "Resize data to it's shape")
.add_arguments(BilinearSampleParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_contrib_BilinearResize2D)
.set_attr_parser(ParamParser<BilinearSampleParam>)
.set_num_inputs(1)
.set_num_outputs(BilinearSampleOpNumBackwardOutputs)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", BilinearSampleOpBackward<cpu>);


}  // namespace op
}  // namespace mxnet
