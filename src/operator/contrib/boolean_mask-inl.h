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
 * \file boolean_mask-inl.h
*/

#ifndef MXNET_OPERATOR_CONTRIB_BOOLEAN_MASK_INL_H_
#define MXNET_OPERATOR_CONTRIB_BOOLEAN_MASK_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/ndarray.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include "../operator_common.h"
#include "../mxnet_op.h"
#include "../tensor/init_op.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct BooleanMaskParam : public dmlc::Parameter<BooleanMaskParam> {
  int axis;
  DMLC_DECLARE_PARAMETER(BooleanMaskParam) {
    DMLC_DECLARE_FIELD(axis).set_default(0)
    .describe("An integer that represents the axis in NDArray to mask from.");
  }
};

template<typename xpu>
inline void BooleanMaskForward(const nnvm::NodeAttrs& attrs,
                               const OpContext &ctx,
                               const std::vector<NDArray> &inputs,
                               const std::vector<OpReqType> &req,
                               const std::vector<NDArray> &outputs) {
  // TODO(@junrushao1994): This implementation is a proof-of-concept,
  // hence very slow actually. Performance should be improved in the future.
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  const BooleanMaskParam& param = nnvm::get<BooleanMaskParam>(attrs.parsed);
  const int axis = param.axis;
  const NDArray &data = inputs[0];
  const NDArray &idx = inputs[1];
  const NDArray &out = outputs[0];
  CHECK_EQ(axis, 0) << "Not supported yet";
  CHECK_EQ(data.shape()[axis], idx.shape()[0]);
  CHECK_EQ(idx.shape().ndim(), 1U);
  // count the number of 1s in `idx`, so that we could know the output dimension
  size_t valid_num = 0;
  MSHADOW_TYPE_SWITCH(idx.dtype(), DType, {
    DType* idx_dptr = idx.data().dptr<DType>();
    int length = idx.shape()[0];
    for (int i = 0; i < length; i++) {
      if (idx_dptr[i]) {
        ++valid_num;
      }
    }
  });
  // set the output shape forcefully
  TShape s = data.shape();
  s[axis] = valid_num;
  const_cast<NDArray &>(out).Init(s);
  // do the copy
  MSHADOW_TYPE_SWITCH(idx.dtype(), DType, {
    DType* idx_dptr = idx.data().dptr<DType>();
    int length = idx.shape()[0];
    mshadow::Stream<xpu> *stream = ctx.get_stream<xpu>();
    for (int i = 0, j = 0; i < length; ++i) {
      if (idx_dptr[i]) {
        NDArray src = data.At(i);
        NDArray dst = out.At(j++);
        CHECK(src.shape() == dst.shape());
        mxnet_op::copy(stream, dst.data(), src.data());
      }
    }
  });
}

template<typename xpu>
inline void BooleanMaskBackward(const nnvm::NodeAttrs& attrs,
                                const OpContext &ctx,
                                const std::vector<NDArray> &inputs,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);
  // inputs: {ograd, data, idx}
  // outputs: {igrad_data, igrad_idx}
  const NDArray& ograd = inputs[0];
  const NDArray& idx = inputs[2];
  const NDArray& igrad_data = outputs[0];
  MSHADOW_TYPE_SWITCH(idx.dtype(), DType, {
    DType* idx_dptr = idx.data().dptr<DType>();
    int length = idx.shape()[0];
    mshadow::Stream<xpu> *stream = ctx.get_stream<xpu>();
    Fill<false>(stream, igrad_data.data(), req[0], 0);
    for (int i = 0, j = 0; i < length; ++i) {
      if (idx_dptr[i]) {
        NDArray src = ograd.At(j++);
        NDArray dst = igrad_data.At(i);
        CHECK(src.shape() == dst.shape());
        mxnet_op::copy(stream, dst.data(), src.data());
      }
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_BOOLEAN_MASK_INL_H_
