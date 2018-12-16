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
 * \file sarange-inl.h
*/

#ifndef MXNET_OPERATOR_CONTRIB_SARANGE_INL_H_
#define MXNET_OPERATOR_CONTRIB_SARANGE_INL_H_

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

struct SArangeParam : public dmlc::Parameter<SArangeParam> {
  std::string ctx;
  int dtype;
  DMLC_DECLARE_PARAMETER(SArangeParam) {
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              "Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype).set_default(mshadow::kFloat32)
    MXNET_ADD_ALL_TYPES
    .describe("Target data type.");
  }
};

template<typename xpu>
inline void SArangeForward(const nnvm::NodeAttrs& attrs,
                           const OpContext &ctx,
                           const std::vector<NDArray> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<NDArray> &outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const NDArray &_size = inputs[0];
  const NDArray &out = outputs[0];
  const int size = [&]() {
    MSHADOW_TYPE_SWITCH(_size.dtype(), DType, {
      DType* dptr = _size.data().dptr<DType>();
      int length = _size.shape()[0];
      CHECK_EQ(length, 1);
      return static_cast<int>(dptr[0]);
    });
    return -1;
  }();
  TShape shape{size};
  const_cast<NDArray &>(out).Init(shape);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(out.dtype(), DType, {
    mxnet_op::Kernel<range_fwd, xpu>::Launch(s,
                                             out.data().Size(),
                                             /*repeat=*/1,
                                             /*start=*/static_cast<DType>(0),
                                             /*step=*/static_cast<DType>(1),
                                             req[0],
                                             out.data().dptr<DType>());
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_SARANGE_INL_H_
