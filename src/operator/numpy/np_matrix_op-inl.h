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
 *  Copyright (c) 2019 by Contributors
 * \file np_matrix_op-inl.h
 * \brief Function definition of matrix related operators
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_

#include <vector>
#include "../tensor/matrix_op-inl.h"
#include "../nn/concat-inl.h"

namespace mxnet {
namespace op {

struct NumpyVstackParam : public dmlc::Parameter<NumpyVstackParam> {
  int num_args;
  DMLC_DECLARE_PARAMETER(NumpyVstackParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
    .describe("Number of inputs to be vstacked.");
  }
};

template<typename xpu>
void NumpyVstackForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow_op;

  const NumpyVstackParam& param = nnvm::get<NumpyVstackParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), param.num_args);
  CHECK_EQ(outputs.size(), 1);
  CHECK_EQ(req.size(), 1);

  // reshape if necessary
  std::vector<TBlob> data(param.num_args);
  for (int i = 0; i < param.num_args; i++) {
    if (inputs[i].shape_.ndim() == 0 || inputs[i].shape_.ndim() == 1) {
      TShape shape = Shape2(1, inputs[i].shape_.Size());
      data[i] = inputs[i].reshape(shape);
    } else {
      data[i] = inputs[i];
    }
  }

  // initialize ConcatOp
  ConcatParam cparam;
  cparam.num_args = param.num_args;
  cparam.dim = 0;
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    ConcatOp<xpu, DType> op;
    op.Init(cparam);
    op.Forward(ctx, data, req, outputs);
  });
}

template<typename xpu>
void NumpyVstackBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow_op;

  const NumpyVstackParam& param = nnvm::get<NumpyVstackParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(outputs.size(), param.num_args);
  CHECK_EQ(req.size(), param.num_args);

  // reshape if necessary
  std::vector<TBlob> data(param.num_args);
  for (int i = 0; i < param.num_args; i++) {
    if (outputs[i].shape_.ndim() == 0 || outputs[i].shape_.ndim() == 1) {
      TShape shape = Shape2(1, outputs[i].shape_.Size());
      data[i] = outputs[i].reshape(shape);
    } else {
      data[i] = outputs[i];
    }
  }

  // initialize ConcatOp
  ConcatParam cparam;
  cparam.num_args = param.num_args;
  cparam.dim = 0;
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    ConcatOp<xpu, DType> op;
    op.Init(cparam);
    op.Backward(ctx, inputs[0], req, data);
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_
