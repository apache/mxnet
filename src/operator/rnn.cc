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
  LOG(FATAL) << "RNN is only available for gpu at the moment.";
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new RNNOp<cpu, DType>(param);
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
.describe("Applies a recurrent layer to input.")
.add_argument("data", "NDArray-or-Symbol", "Input data to RNN")
.add_argument("parameters", "NDArray-or-Symbol",
              "Vector of all RNN trainable parameters concatenated")
.add_argument("state", "NDArray-or-Symbol", "initial hidden state of the RNN")
.add_argument("state_cell", "NDArray-or-Symbol",
              "initial cell state for LSTM networks (only for LSTM)")
.add_arguments(RNNParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
