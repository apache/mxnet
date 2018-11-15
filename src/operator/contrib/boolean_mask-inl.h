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
#include "../operator_common.h"
#include "../mxnet_op.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

struct BooleanMaskParam : public dmlc::Parameter<BooleanMaskParam> {
  int axis;
  DMLC_DECLARE_PARAMETER(BooleanMaskParam) {
    DMLC_DECLARE_FIELD(axis).set_default(0)
    .describe("An integer that represents the axis in NDArray to mask from.");
  }
};

template <typename xpu>
inline void BooleanMaskForward(const nnvm::NodeAttrs& attrs,
                               const OpContext &ctx,
                               const std::vector<NDArray> &inputs,
                               const std::vector<OpReqType> &req,
                               const std::vector<NDArray> &outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  const BooleanMaskParam& param = nnvm::get<BooleanMaskParam>(attrs.parsed);
  CHECK_EQ(param.axis, 0);
  CHECK_EQ(inputs[0].shape()[param.axis], inputs[1].shape()[0]);
  CHECK_EQ(inputs[1].shape().ndim(), 1U);
  size_t valid_num = 0;
  const TBlob &idx = inputs[1].data();
  MSHADOW_TYPE_SWITCH(inputs[1].dtype(), DType, {
    for (int i = 0; i < inputs[1].shape()[0]; i++) {
      if (idx.dptr<DType>()[i])
        valid_num++;
    }
  });
  TShape s = inputs[0].shape();
  s[0] = valid_num;
  const_cast<NDArray &>(outputs[0]).Init(s);
  size_t j = 0;
  size_t ele_size = mshadow::mshadow_sizeof(inputs[0].dtype());
  MSHADOW_TYPE_SWITCH(inputs[1].dtype(), DType, {
  for (int i = 0; i < inputs[1].shape()[0]; i++) {
    if (idx.dptr<DType>()[i]) {
      NDArray src = inputs[0].At(i);
      NDArray dst = outputs[0].At(j);
      CHECK(src.shape() == dst.shape());
      memcpy(dst.data().dptr_, src.data().dptr_, src.shape().Size() * ele_size);
      j++;
    }
  }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_BOOLEAN_MASK_INL_H_
