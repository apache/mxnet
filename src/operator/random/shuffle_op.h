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
 * \file shuffle_op.h
 * \brief Operator to shuffle elements of an NDArray
 */
#ifndef MXNET_OPERATOR_RANDOM_SHUFFLE_OP_H_
#define MXNET_OPERATOR_RANDOM_SHUFFLE_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

template<typename xpu, typename ShuffleImpl>
void ShuffleForward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[0] == kNullOp) {
    return;
  }
  CHECK_NE(req[0], kAddTo) << "Shuffle does not support AddTo";
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TShape input_shape = inputs[0].shape_;
  const size_t n_elements = input_shape[input_shape.ndim() - 1];
  const size_t n_batches = inputs[0].Size() / n_elements;
  const size_t size = inputs[0].Size();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> in = inputs[0].get_with_shape<xpu, 1, DType>(Shape1(size), s);
    Tensor<xpu, 1, DType> out = outputs[0].get_with_shape<xpu, 1, DType>(Shape1(size), s);
    if (req[0] != kWriteInplace) {
      Copy(out, in, s);
    }
    ShuffleImpl::shuffle(ctx, out, n_batches, n_elements);
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_RANDOM_SHUFFLE_OP_H_
