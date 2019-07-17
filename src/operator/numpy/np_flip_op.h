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
 * \file np_flip_op.h
 * \brief Function definition of flip related operators
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_FLIP_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_FLIP_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"
#include "../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {

struct FlipParam : public dmlc::Parameter<FlipParam> {
  mxnet::Tuple<int> axis;
  DMLC_DECLARE_PARAMETER(FlipParam) {
    DMLC_DECLARE_FIELD(axis)
    .describe("The axis which to flip elements.");
  }
};
struct flip0dim {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* out_data,
                                  const DType* in_data) {
    out_data[i] = in_data[i];
  }
};
#define FLIP_MAX_DIM 10
#define FLIP_MIN_DIM -1

template<typename xpu>
void FlipOpForward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const FlipParam& param = nnvm::get<FlipParam>(attrs.parsed);
  mxnet::Tuple<int> axistemp;
  CHECK_EQ(inputs[0].type_flag_, outputs[0].type_flag_);
  CHECK_LT(param.axis.ndim(), FLIP_MAX_DIM);
  CHECK_GE(param.axis.ndim(), FLIP_MIN_DIM);
  if (param.axis.ndim() == FLIP_MIN_DIM) {
    if (inputs[0].shape_.ndim() == 0) {
      Stream<xpu> *s = ctx.get_stream<xpu>();
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        Kernel<flip0dim, xpu>::Launch(s, inputs[0].Size(),
        outputs[0].dptr<DType>(), inputs[0].dptr<DType>());
      });
      return;
    }
    std::vector<int> temp;
    for (int i = 0; i < inputs[0].shape_.ndim(); i++) {
      temp.push_back(i);
    }
    axistemp.assign(temp.begin(), temp.end());
  } else {
    axistemp = param.axis;
  }
  Stream<xpu> *s = ctx.get_stream<xpu>();

  const mxnet::TShape& ishape = inputs[0].shape_;
  if (ishape.ProdShape(0, ishape.ndim()) == 0) {
    // zero shape
    return;
  }
  std::vector<index_t> stride_(axistemp.ndim());
  std::vector<index_t>  trailing_(axistemp.ndim());
  index_t flip_index = 0;
  for (int axis : axistemp) {
    CHECK_LT(axis, ishape.ndim());
    stride_[flip_index] = ishape[axis];
    trailing_[flip_index] = 1;
    for (int i2 = axis + 1; i2 < ishape.ndim(); ++i2) {
      trailing_[flip_index] *= ishape[i2];
    }
    flip_index++;
  }

#ifdef __CUDACC__
  mshadow::Tensor<xpu, 1, uint8_t> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, uint8_t>(
      mshadow::Shape1(flip_index * sizeof(index_t) * 2), s);

  auto stride_workspace = workspace.dptr_;
  auto trailing_workspace = workspace.dptr_ + flip_index * sizeof(index_t);

  cudaMemcpyAsync(stride_workspace, thrust::raw_pointer_cast(stride_.data()),
                  stride_.size() * sizeof(index_t),
                  cudaMemcpyHostToDevice, mshadow::Stream<gpu>::GetStream(s));
  cudaMemcpyAsync(trailing_workspace, thrust::raw_pointer_cast(trailing_.data()),
                  trailing_.size() * sizeof(index_t),
                  cudaMemcpyHostToDevice, mshadow::Stream<gpu>::GetStream(s));

#endif

#ifdef __CUDACC__
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Kernel<reverse, xpu>::Launch(s, inputs[0].Size(), flip_index,
      inputs[0].dptr<DType>(), outputs[0].dptr<DType>(),
      reinterpret_cast<index_t*>(stride_workspace), reinterpret_cast<index_t*>(trailing_workspace));
  });
#else
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Kernel<reverse, xpu>::Launch(s, inputs[0].Size(), flip_index,
      inputs[0].dptr<DType>(), outputs[0].dptr<DType>(),
      stride_.data(), trailing_.data());
  });
#endif
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NUMPY_NP_FLIP_OP_H_
