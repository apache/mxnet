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
 * \file rnn.cc
 * \brief
 * \author Sebastian Bodenstein
*/
#include "./rnn-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(RNNParam param, int dtype) {
  Operator *op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new RNNOp<DType>(param);
  });
  return op;
}

Operator *RNNProp::CreateOperatorEx(Context ctx,
                                  std::vector<TShape> *in_shape,
                                  std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(RNNParam);

MXNET_REGISTER_OP_PROPERTY(RNN, RNNProp)
.describe(R"code(Applies recurrent layers to input data. Currently, vanilla RNN, LSTM and GRU are
implemented, with both multi-layer and bidirectional support.

When the input data is of type float32 and the environment variables MXNET_CUDA_ALLOW_TENSOR_CORE
and MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION are set to 1, this operator will try to use
pseudo-float16 precision (float32 math with float16 I/O) precision in order to use
Tensor Cores on suitable NVIDIA GPUs. This can sometimes give significant speedups.

**Vanilla RNN**

Applies a single-gate recurrent layer to input X. Two kinds of activation function are supported:
ReLU and Tanh.

With ReLU activation function:

.. math::
    h_t = relu(W_{ih} * x_t + b_{ih}  +  W_{hh} * h_{(t-1)} + b_{hh})

With Tanh activtion function:

.. math::
    h_t = \tanh(W_{ih} * x_t + b_{ih}  +  W_{hh} * h_{(t-1)} + b_{hh})

Reference paper: Finding structure in time - Elman, 1988.
https://crl.ucsd.edu/~elman/Papers/fsit.pdf

**LSTM**

Long Short-Term Memory - Hochreiter, 1997. http://www.bioinf.jku.at/publications/older/2604.pdf

.. math::
  \begin{array}{ll}
            i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
            f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\
            o_t = \mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
            c_t = f_t * c_{(t-1)} + i_t * g_t \\
            h_t = o_t * \tanh(c_t)
            \end{array}

**GRU**

Gated Recurrent Unit - Cho et al. 2014. http://arxiv.org/abs/1406.1078

The definition of GRU here is slightly different from paper but compatible with CUDNN.

.. math::
  \begin{array}{ll}
            r_t = \mathrm{sigmoid}(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \mathrm{sigmoid}(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)} \\
            \end{array})code")
.add_argument("data", "NDArray-or-Symbol", "Input data to RNN")
.add_argument("parameters", "NDArray-or-Symbol",
              "Vector of all RNN trainable parameters concatenated")
.add_argument("state", "NDArray-or-Symbol", "initial hidden state of the RNN")
.add_argument("state_cell", "NDArray-or-Symbol",
              "initial cell state for LSTM networks (only for LSTM)")
.add_arguments(RNNParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
