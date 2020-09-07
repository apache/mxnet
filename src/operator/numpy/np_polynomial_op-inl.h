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
 * \file np_polynomial_op.h
 * \brief Functions for dealing with polynomials.
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_POLYNOMIAL_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_POLYNOMIAL_OP_INL_H_

#include <mxnet/base.h>
#include <string>
#include <vector>
#include <type_traits>
#include "../mxnet_op.h"
#include "../../common/utils.h"
#include "../tensor/elemwise_binary_broadcast_op.h"


namespace mxnet {
namespace op {

inline bool NumpyPolyvalShape(const nnvm::NodeAttrs& attrs,
                              mxnet::ShapeVector *in_attrs,
                              mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& p_shape = in_attrs->at(0);
  const mxnet::TShape& x_shape = in_attrs->at(1);
  const mxnet::TShape& v_shape = out_attrs->at(0);
  CHECK_EQ(p_shape.ndim(), 1U) << "ValueError: p has to be an 1-D array.";
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, x_shape);
  SHAPE_ASSIGN_CHECK(*in_attrs, 1, v_shape);
  return shape_is_known(*in_attrs) && shape_is_known(*out_attrs);
}

template<int req>
struct polyval_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* out_data,
                                  const DType* p_data,
                                  const DType* x_data,
                                  const index_t p_size) {
    DType val = 0;
    for (index_t j = 0; j < p_size; j++) {
        val = val * x_data[i] + p_data[j];
    }
    KERNEL_ASSIGN(out_data[i], req, val);
  }
};

template<typename xpu>
void NumpyPolyvalForward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mxnet;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& p_data = inputs[0];
  const TBlob& x_data = inputs[1];
  const TBlob& out_data = outputs[0];
  const size_t p_size = p_data.Size();
  using namespace mxnet_op;

  MSHADOW_TYPE_SWITCH(x_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<polyval_forward<req_type>, xpu>::Launch(
          s, out_data.Size(), out_data.dptr<DType>(),
          p_data.dptr<DType>(), x_data.dptr<DType>(), p_size);
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_POLYNOMIAL_OP_INL_H_
