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
 *  Copyright (c) 2018 by Contributors
 * \file transformer-inl.h
 * \brief Function used in cc and cu
 */
#ifndef MXNET_OPERATOR_CONTRIB_TRANSFORMER_INL_H_
#define MXNET_OPERATOR_CONTRIB_TRANSFORMER_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mxnet_op.h"
#include "../mshadow_op.h"


namespace mxnet {
namespace op {

template<typename xpu>
static void DivSqrtDimForward_(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  double sqrt_dim = std::sqrt(static_cast<double>(inputs[0].shape_[inputs[0].ndim() - 1]));
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::div, Req>, xpu>::Launch(
        s, inputs[0].Size(), outputs[0].dptr<DType>(), inputs[0].dptr<DType>(), DType(sqrt_dim));
    });
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_TRANSFORMER_INL_H_
