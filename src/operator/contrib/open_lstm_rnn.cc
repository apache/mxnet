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
 * \file open_lstm_rnn.cc
 * \brief LSTM RNN Open-Source CUDA Implementation
 * \author Bojian (Jack) Zheng, Gennady Pekhimenko
 */
#include "./open_lstm_rnn-inl.h"

namespace mxnet {
namespace op {

template <>
Operator *CreateOp<cpu>(OpenLSTMRNNParam param, int dtype) {
  LOG(FATAL) << "OpenLSTMRNN is only available for gpu at the moment.";

  Operator * op = NULL;

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType,
                           {op = new OpenLSTMRNNOp<cpu, DType>(param);});

  return op;
}

Operator *OpenLSTMRNNProp::CreateOperatorEx(Context ctx,
                                            std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(OpenLSTMRNNParam);

MXNET_REGISTER_OP_PROPERTY(OpenLSTMRNN, OpenLSTMRNNProp)
.describe(R"code(Applies a LSTM recurrent layer to input, with multi-layer BUT NOT bidirectional support.
**LSTM**
Long Short-Term Memory - Hochreiter, 1997.
.. math::
  \begin{array}{ll}
            i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
            f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\
            o_t = \mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
            c_t = f_t * c_{(t-1)} + i_t * g_t \\
            h_t = o_t * \tanh(c_t)
            \end{array})code")
.add_argument("data", "NDArray-or-Symbol", "Input data to EcoRNN")
.add_argument("init_hidden", "NDArray-or-Symbol", "Initial Hidden State")
.add_argument("init_cell"  , "NDArray-or-Symbol", "Initial Cell State")
.add_argument("i2h_weight" , "NDArray-or-Symbol",  "Input-to-Hidden Weight")
.add_argument("i2h_bias"   , "NDArray-or-Symbol",  "Input-to-Hidden Bias")
.add_argument("h2h_weight" , "NDArray-or-Symbol", "Hidden-to-Hidden Weight")
.add_argument("h2h_bias"   , "NDArray-or-Symbol", "Hidden-to-Hidden Bias")
.add_arguments(OpenLSTMRNNParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
