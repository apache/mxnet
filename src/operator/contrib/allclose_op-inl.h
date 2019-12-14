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

// Intermediate and Output data types could be integers OR unsigned characters
#define USE_INTEGER   0
#if USE_INTEGER
  #define INTERM_DATA_TYPE int32_t
  #define OUT_DATA_TYPE    mshadow::kInt32
#else
  #define INTERM_DATA_TYPE uint8_t
  #define OUT_DATA_TYPE    mshadow::kUint8
#endif

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
      .describe("Whether to compare NaN's as equal. If True, NaN's in A will be considered equal "
                "to NaN's in B in the output array.");
  }
};

inline bool AllCloseShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape>* in_attrs,
                          std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U) << "Input:[array1, array2]";
  CHECK_EQ(out_attrs->size(), 1U);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(0, -1));
  return in_attrs->at(0) == in_attrs->at(1);
}

inline bool AllCloseType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  // The output will be boolean stored as an OUT_DATA_TYPE format
  TYPE_ASSIGN_CHECK(*out_attrs, 0, OUT_DATA_TYPE);
  return (*out_attrs)[0] != -1;
}

using mshadow::isnan_typed::IsNan;

template<int req>
struct allclose_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, INTERM_DATA_TYPE *out_data,
                                  const DType *in_a, const DType *in_b,
                                  const float rtol, const float atol, bool equal_nan) {
      const DType a = in_a[i], b = in_b[i];
      bool val;
      if (IsNan(a) || IsNan(b))
        val = equal_nan && IsNan(a) == IsNan(b);
      else
        val = a == b || (a > b?  a - b : b - a) <= atol + (b > 0? rtol * b :  (-rtol) * b);

      KERNEL_ASSIGN(out_data[i], req, val? 1 : 0);
  }
};

template<typename xpu>
size_t GetAdditionalMemoryLogical(mshadow::Stream<xpu> *s, const int num_items);

template<typename xpu>
INTERM_DATA_TYPE *GetAdditionalMemoryLogical(const OpContext& ctx,
                                             int num_items, size_t *pExtraStorageBytes) {
// Get length of the additional memory (which is used only by DeviceReduce::Min(...) on gpu)
  *pExtraStorageBytes = GetAdditionalMemoryLogical<xpu>(ctx.get_stream<xpu>(), num_items);
  const size_t workspace_total_bytes_ = num_items * sizeof(INTERM_DATA_TYPE) + *pExtraStorageBytes;
  mshadow::Tensor<xpu, 1, uint8_t> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, uint8_t>(
      mshadow::Shape1(workspace_total_bytes_), ctx.get_stream<xpu>());

  return reinterpret_cast<INTERM_DATA_TYPE *>(workspace.dptr_);
}

template<typename xpu>
void GetResultLogical(mshadow::Stream<xpu> *s, INTERM_DATA_TYPE *workMem, size_t extraStorageBytes,
                      int num_items, INTERM_DATA_TYPE *outPntr);

template<typename xpu>
void AllClose(const nnvm::NodeAttrs& attrs,
              const OpContext& ctx,
              const std::vector<TBlob>& inputs,
              const std::vector<OpReqType>& req,
              const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  const TBlob& in0 = inputs[0];
  const TBlob& in1 = inputs[1];
  const int num_items = in0.Size();

  size_t extraStorageBytes;
  auto workspaceMem = GetAdditionalMemoryLogical<xpu>(ctx, num_items, &extraStorageBytes);
  auto s = ctx.get_stream<xpu>();
  const AllCloseParam& param = nnvm::get<AllCloseParam>(attrs.parsed);
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(in0.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<allclose_forward<req_type>, xpu>::Launch(
        s, num_items, workspaceMem, in0.dptr<DType>(), in1.dptr<DType>(),
        param.rtol, param.atol, param.equal_nan);
    });
  });

  auto *pOut = outputs[0].dptr<INTERM_DATA_TYPE>();
  GetResultLogical<xpu>(s, workspaceMem, extraStorageBytes, num_items, pOut);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_ALLCLOSE_OP_INL_H_
