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
 * \file allclose-inl.h
 * \brief Operator implementing numpy.allclose function.
 * \author Andrei Ivanov
 */
#ifndef MXNET_OPERATOR_CONTRIB_ALLCLOSE_OP_INL_H_
#define MXNET_OPERATOR_CONTRIB_ALLCLOSE_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

struct AllCloseParam : public dmlc::Parameter<AllCloseParam> {
  float rtol, atol;
  bool equal_nan;
  DMLC_DECLARE_PARAMETER(AllCloseParam) {
    DMLC_DECLARE_FIELD(rtol)
      .set_default(1e-05)
      .describe("Relative tolerance.");
    DMLC_DECLARE_FIELD(atol)
      .set_default(1e-08)
      .describe("Absolute tolerance.");
    DMLC_DECLARE_FIELD(equal_nan)
      .set_default(true)
      .describe("Whether to compare NaN’s as equal. If True, NaN’s in a will be considered equal "
                "to NaN’s in b in the output array.");
  }
};

inline bool AllCloseShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape>* in_attrs,
                          std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U) << "Input:[array1, array2]";
  CHECK_EQ(out_attrs->size(), 1U);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape());
  return in_attrs->at(0) == in_attrs->at(1);
}

inline bool AllCloseType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  // The output will be boolean stored as an integer
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt32);
  return (*out_attrs)[0] != -1;
}

using namespace mshadow_op::isnan_typed;

template<int req>
struct allclose_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, int *out_data, const DType* in_a, const DType* in_b,
                                  const float rtol, const float atol, bool equal_nan) {
      const DType a = in_a[i], b = in_b[i];
      const bool val = (IsNan(a) || IsNan(b))? equal_nan :
                       a == b ||
                       (a > b?  a - b : b - a) <= atol + (b > 0? rtol * b :  (-rtol) * b);
      KERNEL_ASSIGN(out_data[i], req, val? 1 : 0);
  }
};

template<typename xpu>
size_t GetAdditionalMemorySize(const int num_items);

template<typename xpu>
void AllCloseAction(mshadow::Stream<xpu> *s,
                    int *workspaceMemory,
                    size_t extraStorageBytes,
                    const TBlob& in_array0,
                    const TBlob& in_array1,
                    const std::vector<OpReqType>& req,
                    const AllCloseParam& param,
                    int *outPntr);

template<typename xpu>
void AllClose(const nnvm::NodeAttrs& attrs,
              const OpContext& ctx,
              const std::vector<TBlob>& inputs,
              const std::vector<OpReqType>& req,
              const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  const TBlob& in_array0 = inputs[0];
  const TBlob& in_array1 = inputs[1];

  const AllCloseParam& param = nnvm::get<AllCloseParam>(attrs.parsed);

  auto s = ctx.get_stream<xpu>();
  const int num_items = in_array0.Size();

  // Get length of the additional memory (which is used only by DeviceReduce::Min(...) on gpu)
  const size_t extraStorageBytes = GetAdditionalMemorySize<xpu>(num_items);
  const size_t workspace_total_bytes_ = num_items * sizeof(float) + extraStorageBytes;
  mshadow::Tensor<xpu, 1, uint8_t> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, uint8_t>(
      mshadow::Shape1(workspace_total_bytes_), s);

  auto workspaceMem = reinterpret_cast<int *>(workspace.dptr_);
  AllCloseAction(s, workspaceMem, extraStorageBytes, in_array0, in_array1,
                 req, param, outputs[0].dptr<int>());
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_ALLCLOSE_OP_INL_H_
