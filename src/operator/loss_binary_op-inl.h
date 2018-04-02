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
 *  Copyright (c) 2015 by Contributors
 * \file loss_binary_op-inl.h
 * \brief Loss functions
 */
#ifndef MXNET_OPERATOR_LOSS_BINARY_OP_INL_H_
#define MXNET_OPERATOR_LOSS_BINARY_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "./mshadow_op.h"
#include "./elemwise_op_common.h"

namespace mxnet {
namespace op {

// return a shape of scalar
inline bool SoftmaxCrossEntropyShape(const nnvm::NodeAttrs& attrs,
                                     std::vector<TShape> *in_attrs,
                                     std::vector<TShape> *out_attrs) {
  CHECK_EQ((*in_attrs)[0].ndim(), 2U)
      << "SoftmaxCrossEntropy only accept 2D data";
  CHECK_EQ((*in_attrs)[1].ndim(), 1U)
      << "SoftmaxCrossEntropy only accept 1D label";
  CHECK_EQ((*in_attrs)[0][0], (*in_attrs)[1][0])
      << "SoftmaxCrossEntropy: data label shape mismatch";
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(1));
  return true;
}

template<typename xpu>
void SoftmaxCrossEntropyForward(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(outputs[0].type_flag_, inputs[0].type_flag_)
    << "Binary function only support input/output with the same type";
  CHECK_EQ(outputs[0].type_flag_, inputs[1].type_flag_)
    << "Binary function only support input/output with the same type";
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    mshadow::Tensor<xpu, 1, DType> out = outputs[0].get<xpu, 1, DType>(s);
    mshadow::Tensor<xpu, 1, DType> mlabel = inputs[1].get<xpu, 1, DType>(s);
    mshadow::Tensor<xpu, 2, DType> mdata = inputs[0].get<xpu, 2, DType>(s);
    mshadow::Tensor<xpu, 1, DType> workspace = ctx.requested[0].get_space_typed<xpu, 1, DType>(
        mshadow::Shape1(mdata.shape_.Size() + mlabel.size(0)), s);
    mshadow::Tensor<xpu, 2, DType> temp1(workspace.dptr_, mdata.shape_, s);
    mshadow::Tensor<xpu, 2, DType> temp2(workspace.dptr_ + mdata.shape_.Size(),
        mshadow::Shape2(1, mlabel.size(0)), s);
    // calculate softmax on temp
    // TODO(tqchen): change to SoftmaxLog later
    mshadow::Softmax(temp1, mdata);
    // choose the softmax rows
    mshadow::Tensor<xpu, 1, DType> tdst = temp2[0];
    tdst = F<mshadow_op::negation>(
        F<mshadow_op::log>(
            F<mshadow_op::maximum>(mat_choose_row_element(temp1, mlabel),
                                   scalar<DType>(1e-8f))));
    ASSIGN_DISPATCH(out, req[0], sumall_except_dim<0>(temp2));
  });
}

template<typename xpu>
void SoftmaxCrossEntropyBackward(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(req[1], kNullOp)
      << "SoftmaxCrossEntropy: Cannot take gradient wrt label";
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    mshadow::Tensor<xpu, 1, DType> mlabel = inputs[2].get<xpu, 1, DType>(s);
    mshadow::Tensor<xpu, 2, DType> mdata = inputs[1].get<xpu, 2, DType>(s);
    mshadow::Tensor<xpu, 2, DType> mdata_grad = outputs[0].get<xpu, 2, DType>(s);
    mshadow::Tensor<xpu, 1, DType> mscale = inputs[0].get<xpu, 1, DType>(s);
    mshadow::Tensor<xpu, 2, DType> temp = ctx.requested[0].get_space_typed<xpu, 2, DType>(
        mdata.shape_, s);
    mshadow::Softmax(temp, mdata);
    mshadow::SoftmaxGrad(temp, temp, mlabel);
    ASSIGN_DISPATCH(mdata_grad, req[0], broadcast_scalar(mscale, temp.shape_) * temp);
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_LOSS_BINARY_OP_INL_H_
