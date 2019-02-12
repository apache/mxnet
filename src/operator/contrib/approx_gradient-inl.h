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
 * \file approx_gradient-inl.h
 * \brief Operator implementing calculation of numerical gradient approximation.
 * \author Andrei Ivanov
 */
#ifndef MXNET_OPERATOR_CONTRIB_APPROX_GRADIENT_INL_H_
#define MXNET_OPERATOR_CONTRIB_APPROX_GRADIENT_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

struct ApproxGradientParam : public dmlc::Parameter<ApproxGradientParam> {
  int index;
  float eps;
  bool batched_mode;
  DMLC_DECLARE_PARAMETER(ApproxGradientParam) {
    DMLC_DECLARE_FIELD(index).set_lower_bound(0)
      .describe(R"code(Index of gradient vector coordinate to be calculated OR
the batch size when batched_mode is TRUE.
      )code");
    DMLC_DECLARE_FIELD(eps).set_default(1e-05)
      .describe("Epsilon value.");
    DMLC_DECLARE_FIELD(batched_mode).set_default(false)
    .describe("Whether the batch mode was used.");
  }
};

#define CHECK_ATTRS(in_attrs, out_attrs)                                        \
      CHECK_EQ((in_attrs)->size(), 3U) << "Input:[array1, array2, gradVector]"; \
      CHECK_EQ((out_attrs)->size(), 1U);

inline bool ApproxGradientShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape>* in_attrs,
                          std::vector<TShape>* out_attrs) {
  CHECK_ATTRS(in_attrs, out_attrs);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape());
  return in_attrs->at(0) == in_attrs->at(1);
}

inline bool ApproxGradientType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  CHECK_ATTRS(in_attrs, out_attrs);

  // The output will be same type as input vector
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat32);
  return (*out_attrs)[0] != -1;
}

template<int req>
struct vector_increment {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, float *out_data, const DType *in_a,
                                  const DType *in_b, const float eps) {
      KERNEL_ASSIGN(out_data[i], req, (in_a[i] - in_b[i]) / eps);
  }
};

template<int req>
struct approx_gradient {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, float *pCoordValue) {
    KERNEL_ASSIGN(out[i], req, *pCoordValue);
  }
};

template<int req>
struct assign_gradient {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, float *workSpaceMemory, int num_items) {
    float sum = 0.f;
    workSpaceMemory += i * num_items;
    while (num_items-- > 0)
      sum += workSpaceMemory[num_items];

    KERNEL_ASSIGN(out[i], req, sum);
  }
};

template<typename xpu>
size_t GetAdditionalMemorySizeA(const int num_items);

template<typename xpu>
void ApproxGradientAction(mshadow::Stream<xpu> *s,
                    float *workspaceMemory,
                    size_t extraStorageBytes,
                    const TBlob& in_array0,
                    const TBlob& in_array1,
                    const TBlob& gradCoord,
                    const std::vector<OpReqType>& req,
                    const ApproxGradientParam& param);

template<typename xpu>
void ApproxGradient(const nnvm::NodeAttrs& attrs,
                const OpContext& ctx,
                const std::vector<TBlob>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& outputs) {
  CHECK_ATTRS(&inputs, &outputs);
  CHECK_EQ(req.size(), 1U);

  const TBlob& in_array0 = inputs[0];
  const int num_items = in_array0.Size();

  const ApproxGradientParam& param = nnvm::get<ApproxGradientParam>(attrs.parsed);
  const auto batched_mode = param.batched_mode;
  const auto index = param.index;

  // Get length of the additional memory (which is used only by DeviceReduce::Sum(...) on gpu)
  const size_t extraStorageBytes = GetAdditionalMemorySizeA<xpu>(num_items);

  // Allocate memory for intermediate results and coordinate of the gradient
  const size_t workspace_total_bytes = batched_mode /*|| extraStorageBytes == 0*/?
                        (num_items + index) * sizeof(float) + extraStorageBytes * index :
                        (num_items + 1) * sizeof(float) + extraStorageBytes;

  auto s = ctx.get_stream<xpu>();
  mshadow::Tensor<xpu, 1, uint8_t> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, uint8_t>(
      mshadow::Shape1(workspace_total_bytes), s);

  auto workspaceMem = reinterpret_cast<float *>(workspace.dptr_);
  ApproxGradientAction(s, workspaceMem, extraStorageBytes,
                       in_array0, inputs[1], inputs[2], req, param);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_APPROX_GRADIENT_INL_H_
