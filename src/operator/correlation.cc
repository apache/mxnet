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
 * Copyright (c) 2015 by Contributors
 * \file correlation.cc
 * \brief correlation op
 * \author Xu Dong
*/
#include "./correlation-inl.h"
#include "./mshadow_op.h"

namespace mshadow {
template<typename Dtype>
void AddPad(const Tensor<cpu, 4, Dtype> &original,
            const Tensor<cpu, 4, Dtype> &out,
            int pad_size)
{ for (index_t nbatch = 0 ; nbatch < original.size(0) ; nbatch++)
  for (index_t channel = 0 ; channel < original.size(1) ; channel++)
    for (index_t h = 0 ; h < original.size(2) ; h++)
      for (index_t w = 0 ; w < original.size(3) ; w++)
         out[nbatch][h+pad_size][w+pad_size][channel] = original[nbatch][channel][h][w];
}
template<typename Dtype>
inline void CorrelationForward(const Tensor<cpu, 4, Dtype> &out,
                               const Tensor<cpu, 4, Dtype> &data1,
                               const Tensor<cpu, 4, Dtype> &data2,
                               const Tensor<cpu, 4, Dtype> &tmp1,
                               const Tensor<cpu, 4, Dtype> &tmp2,
                               int top_channels_, int top_height_, int top_width_,
                               int pad_size_, bool is_multiply,
                               int max_displacement_, int kernel_size_,
                               int neighborhood_grid_radius_, int neighborhood_grid_width_,
                               int  kernel_radius_, int stride1_, int stride2_) {
  const index_t bnum = data1.size(0);
  const int bchannels = data1.size(1);
  const int sumelems = kernel_size_ * kernel_size_ * bchannels;
  AddPad<Dtype>(data1, tmp1, pad_size_);
  index_t top_channels_unsigned_ = static_cast<index_t>(top_channels_);
  AddPad<Dtype>(data2, tmp2, pad_size_);
  for (index_t i = 0 ; i < static_cast<index_t>(top_height_) ; i++)
      for (index_t j = 0 ; j < static_cast<index_t>(top_width_); j++)
        for (index_t nbatch = 0 ; nbatch < bnum ; nbatch++) {
            int x1 = j*stride1_+max_displacement_;
            int y1 = i*stride1_+max_displacement_;
            for (index_t top_channel = 0 ; top_channel < top_channels_unsigned_ ; top_channel++) {
              int s2o = (top_channel % neighborhood_grid_width_ -\
                         neighborhood_grid_radius_) * stride2_;
              int s2p = (top_channel / neighborhood_grid_width_ -\
                         neighborhood_grid_radius_) * stride2_;
              int x2 = x1 + s2o;
              int y2 = y1 + s2p;
              for (index_t h = 0; h < static_cast<index_t>(kernel_size_); h++)
                for (index_t w = 0; w < static_cast<index_t>(kernel_size_); w++)
                  for (index_t channel = 0; channel < static_cast<index_t>(bchannels); channel++) {
                    if (is_multiply == true)
                        out[nbatch][top_channel][i][j] += \
                        tmp1[nbatch][y1+h][x1+w][channel]*tmp2[nbatch][y2+h][x2+w][channel];
                    else
                        out[nbatch][top_channel][i][j] += \
                        fabsf(tmp1[nbatch][y1+h][x1+w][channel]-tmp2[nbatch][y2+h][x2+w][channel]);
                  }
              out[nbatch][top_channel][i][j] /= sumelems;
            }
        }
}
template<typename Dtype>
inline void CorrelationBackward(const Tensor<cpu, 4, Dtype> &out_grad,
                                const Tensor<cpu, 4, Dtype> &in_grad1,
                                const Tensor<cpu, 4, Dtype> &in_grad2,
                                const Tensor<cpu, 4, Dtype> &tmp1,
                                const Tensor<cpu, 4, Dtype> &tmp2,
                                int top_channels_, int top_height_,
                                int top_width_, int pad_size_,
                                bool is_multiply, int max_displacement_,
                                int kernel_size_, int neighborhood_grid_radius_,
                                int neighborhood_grid_width_,
                                int  kernel_radius_, int stride1_,
                                int stride2_, int num,
                                int channels, int height, int width
                            ) {
  const float sumelems = kernel_size_ * kernel_size_ * channels;
  for (index_t i = 0 ; i < static_cast<index_t>(top_height_) ; i++)
     for (index_t j = 0 ; j < static_cast<index_t>(top_width_); j++)
        for (index_t nbatch = 0 ; nbatch < static_cast<index_t>(num) ; nbatch++) {
            int x1 = j*stride1_+max_displacement_;
            int y1 = i*stride1_+max_displacement_;
            for (int top_channel = 0 ; top_channel < top_channels_ ; top_channel++) {
              int s2o = (top_channel % neighborhood_grid_width_ - \
              neighborhood_grid_radius_) * stride2_;
              int s2p = (top_channel / neighborhood_grid_width_ - \
              neighborhood_grid_radius_) * stride2_;
              int x2 = x1 + s2o;
              int y2 = y1 + s2p;
              for (int h = 0; h < kernel_size_; h++)
                for (int w = 0; w < kernel_size_; w++)
                  for (int channel = 0 ; channel < channels; channel++) {
                    if (is_multiply == true) {
                      if ((y1 +  h - pad_size_ >= 0) && (x1 + w - pad_size_ >= 0) && \
                      (y1 + h < height +pad_size_) && (x1 + w < width + pad_size_)) {
                        in_grad1[nbatch][channel][y1+h-pad_size_][x1+w-pad_size_] += \
                        out_grad[nbatch][top_channel][i][j] * \
                        tmp2[nbatch][y2+h][x2+w][channel]/sumelems;
                       }
                       if ((y2 +  h - pad_size_ >= 0) && (x2 + w -pad_size_ >=0) && \
                       (y2 + h < height +pad_size_) && (x2 + w < width + pad_size_)) {
                       in_grad2[nbatch][channel][y2+h-pad_size_][x2+w-pad_size_] += \
                       out_grad[nbatch][top_channel][i][j] * \
                       tmp1[nbatch][y1+h][x1+w][channel]/sumelems;
                       }
                    } else {
                      if ((y1 +  h - pad_size_ >= 0) && (x1 + w -pad_size_ >=0) && \
                      (y1 + h < height + pad_size_) && (x1 + w < width + pad_size_)) {
                        Dtype sign  = (tmp1[nbatch][y1+h][x1+w][channel] >= \
                        tmp2[nbatch][y2+h][x2+w][channel])? Dtype(1.0) : Dtype(-1.0);
                        in_grad1[nbatch][channel][y1+h-pad_size_][x1+w-pad_size_] +=\
                        out_grad[nbatch][top_channel][i][j]*sign/sumelems;
                      }
                      if ((y2 +  h - pad_size_ >= 0) && (x2 + w - pad_size_ >=0) && \
                      (y2 + h < height + pad_size_) && (x2 + w < width + pad_size_)) {
                        Dtype sign  = (tmp1[nbatch][y1+h][x1+w][channel] >= \
                        tmp2[nbatch][y2+h][x2+w][channel])? Dtype(-1.0) : Dtype(1.0);
                        in_grad2[nbatch][channel][y2+h-pad_size_][x2+w-pad_size_] +=\
                        out_grad[nbatch][top_channel][i][j]*sign/sumelems;
                       }
                    }
                  }
               }
         }
}
}  // namespace mshadow
namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(CorrelationParam param) {
  return new CorrelationOp<cpu>(param);
}
Operator* CorrelationProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}
DMLC_REGISTER_PARAMETER(CorrelationParam);
MXNET_REGISTER_OP_PROPERTY(Correlation, CorrelationProp)
.add_argument("data1", "NDArray-or-Symbol", "Input data1 to the correlation.")
.add_argument("data2", "NDArray-or-Symbol", "Input data2 to the correlation.")
.add_arguments(CorrelationParam::__FIELDS__())
.describe(R"code(Applies correlation to inputs.

The correlation layer performs multiplicative patch comparisons between two feature maps.

Given two multi-channel feature maps :math:`f_{1}, f_{2}`, with :math:`w`, :math:`h`, and :math:`c` being their width, height, and number of channels,
the correlation layer lets the network compare each patch from :math:`f_{1}` with each patch from :math:`f_{2}`.

For now we consider only a single comparison of two patches. The 'correlation' of two patches centered at :math:`x_{1}` in the first map and
:math:`x_{2}` in the second map is then defined as:

.. math::
   c(x_{1}, x_{2}) = \sum_{o \in [-k,k] \times [-k,k]} <f_{1}(x_{1} + o), f_{2}(x_{2} + o)>

for a square patch of size :math:`K:=2k+1`.

Note that the equation above is identical to one step of a convolution in neural networks, but instead of convolving data with a filter, it convolves data with other
data. For this reason, it has no training weights.

Computing :math:`c(x_{1}, x_{2})` involves :math:`c * K^{2}` multiplications. Comparing all patch combinations involves :math:`w^{2}*h^{2}` such computations.

Given a maximum displacement :math:`d`, for each location :math:`x_{1}` it computes correlations :math:`c(x_{1}, x_{2})` only in a neighborhood of size :math:`D:=2d+1`,
by limiting the range of :math:`x_{2}`. We use strides :math:`s_{1}, s_{2}`, to quantize :math:`x_{1}` globally and to quantize :math:`x_{2}` within the neighborhood
centered around :math:`x_{1}`.

The final output is defined by the following expression:

.. math::
  out[n, q, i, j] = c(x_{i, j}, x_{q})

where :math:`i` and :math:`j` enumerate spatial locations in :math:`f_{1}`, and :math:`q` denotes the :math:`q^{th}` neighborhood of :math:`x_{i,j}`.
)code" ADD_FILELINE);
}  // namespace op
}  // namespace mxnet
