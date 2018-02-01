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
 *  Copyright (c) 2017 by Contributors
 * \file compute_acc_hits-inl.h
 * \brief implementation of compute_accidental_hits operator
 */
#ifndef MXNET_OPERATOR_CONTRIB_COMPUTE_ACC_HITS_INL_H_
#define MXNET_OPERATOR_CONTRIB_COMPUTE_ACC_HITS_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../elemwise_op_common.h"
#include "../operator_common.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

template<typename xpu>
void AccidentalHitComputeCsrImpl(mshadow::Stream<xpu> *s,
                                 const TBlob& label,
                                 const TBlob& sample,
                                 const OpReqType req,
                                 const NDArray& output);

template<typename xpu>
void AccidentalHitComputeEx(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<NDArray>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  if (common::ContainsOnlyStorage(inputs, kDefaultStorage) &&
      outputs[0].storage_type() == kCSRStorage) {
    AccidentalHitComputeCsrImpl(s, inputs[0].data(), inputs[1].data(), req[0],
                                outputs[0]);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

inline bool AccidentalHitShape(const nnvm::NodeAttrs& attrs,
                               std::vector<TShape> *in_attrs,
                               std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  for (size_t i = 0; i < 2; ++i) {
    CHECK_EQ(in_attrs->at(i).ndim(), 1);
  }
  TShape out_attr{in_attrs->at(0)[0], in_attrs->at(1)[0]};
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_attr);
  return true;
}

inline bool AccidentalHitStorageType(const nnvm::NodeAttrs& attrs,
                                     const int dev_mask,
                                     DispatchMode* dispatch_mode,
                                     std::vector<int>* in_attrs,
                                     std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  auto& out_stype = out_attrs->at(0);
  bool dispatched = false;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage) &&
      dev_mask == Context::kCPU) {
    // dns, dns -> csr
    dispatched = storage_type_assign(&out_stype, kCSRStorage, dispatch_mode,
                                     DispatchMode::kFComputeEx);
  }
  return dispatched;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_COMPUTE_ACC_HITS_INL_H_
