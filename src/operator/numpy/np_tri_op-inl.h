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
 * Copyright (c) 2019 by Contributors
 * \file np_tri_op-inl.h
 * \brief Function definition of the tri op
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_TRI_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_TRI_OP_INL_H_

#include <dmlc/parameter.h>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../../api/operator/op_utils.h"

namespace mxnet {
namespace op {

struct TriParam : public dmlc::Parameter<TriParam> {
  nnvm::dim_t N;
  dmlc::optional<nnvm::dim_t> M;
  int k;
  int dtype;
  std::string ctx;
  DMLC_DECLARE_PARAMETER(TriParam) {
    DMLC_DECLARE_FIELD(N)
      .describe("Number of rows in the array.");
    DMLC_DECLARE_FIELD(M)
      .set_default(dmlc::optional<nnvm::dim_t>())
      .describe("Number of columns in the array. "
                "By default, M is taken equal to N.");
    DMLC_DECLARE_FIELD(k)
      .set_default(0)
      .describe("The sub-diagonal at and below which the array is filled. "
                "k = 0 is the main diagonal, while k < 0 is below it, "
                "and k > 0 is above. The default is 0.");
    DMLC_DECLARE_FIELD(dtype)
    MXNET_ADD_ALL_TYPES_WITH_BOOL
    .set_default(mshadow::kFloat32);
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              " Only used for imperative calls.");
  }

  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream N_s, M_s, k_s, dtype_s;
    N_s << N;
    M_s << M;
    k_s << k;
    dtype_s << dtype;
    (*dict)["N"] = N_s.str();
    (*dict)["M"] = M_s.str();
    (*dict)["k"] = k_s.str();
    (*dict)["dtype"] = MXNetTypeWithBool2String(dtype);
  }
};

struct tri_fwd {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out,
                                  mshadow::Shape<2> oshape, int k) {
    using namespace mxnet_op;

    const index_t row_id = i / oshape[1];
    const index_t col_id = i % oshape[1];
    if (col_id > (row_id + k)) {
      out[i] = static_cast<DType>(0);
    } else {
      out[i] = static_cast<DType>(1);
    }
  }
};

template<typename xpu>
void TriOpForward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 0U);
  CHECK_EQ(outputs.size(), 1U);

  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& out_data = outputs[0];
  const TriParam& param = nnvm::get<TriParam>(attrs.parsed);
  const mxnet::TShape& oshape = out_data.shape_;

  MSHADOW_TYPE_SWITCH_WITH_BOOL(out_data.type_flag_, DType, {
    Kernel<tri_fwd, xpu>::Launch(
        s, out_data.Size(), out_data.dptr<DType>(),
        Shape2(oshape[0], oshape[1]), param.k);
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_TRI_OP_INL_H_
