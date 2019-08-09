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
 * \file np_init_op.h
 * \brief Function definition of numpy init op
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_INIT_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_INIT_OP_H_

#include <algorithm>
#include <vector>
#include <string>
#include "../tensor/init_op.h"
#include "../tensor/elemwise_unary_op.h"


namespace mxnet {
namespace op {

struct IndicesOpParam : public dmlc::Parameter<IndicesOpParam> {
  mxnet::TShape dimensions;
  int dtype;
  std::string ctx;
  DMLC_DECLARE_PARAMETER(IndicesOpParam) {
    DMLC_DECLARE_FIELD(dimensions)
    .describe("The shape of the grid.");
    DMLC_DECLARE_FIELD(dtype).set_default(mshadow::kInt32)
      MXNET_ADD_ALL_TYPES
      .describe("Target data type.");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              "Only used for imperative calls.");
  }
};

template<int req>
struct indices_fwd {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out,
                                  const nnvm::dim_t value,
                                  const nnvm::dim_t N,
                                  const nnvm::dim_t dim_i,
                                  const nnvm::dim_t j,
                                  const nnvm::dim_t k,
                                  const nnvm::dim_t t) {
    KERNEL_ASSIGN(out[dim_i*N+N/(t*value)*j+i+k*N/t], req, static_cast<DType>(j));
  }
};

template<int req>
struct identity {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out_data, const int n) {
    using namespace mxnet_op;

    const index_t row_id = i / n;
    const index_t col_id = i % n;
    if (row_id == col_id) {
      KERNEL_ASSIGN(out_data[i], req, static_cast<DType>(1));
    } else {
      KERNEL_ASSIGN(out_data[i], req, static_cast<DType>(0));
    }
  }
};

template<typename xpu>
void IndicesCompute(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 0U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const IndicesOpParam& param = nnvm::get<IndicesOpParam>(attrs.parsed);
  const TBlob& out_data = outputs[0];
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  dim_t indim = param.dimensions.ndim();
  dim_t t = 1;
  dim_t N = out_data.Size()/indim;
  dim_t value = 0;
  if (out_data.Size() == 0) return;
  if (req[0] != kNullOp) {
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
          for (int i = 0; i < indim; ++i) {
            value = param.dimensions[i];
            for (int k = 0; k < t; ++k) {
              for (int j = 0; j < param.dimensions[i]; ++j) {
                Kernel<indices_fwd<req_type>, xpu>::Launch(s, N/(param.dimensions[i] * t),
                    out_data.dptr<DType>(), value, N, i, j, k, t);
              }
            }
            t = t * param.dimensions[i];
          }
      });
    });
  }
}

template<typename xpu>
void IdentityCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 0U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& out_data = outputs[0];
  int n = out_data.shape_[0];
  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<identity<req_type>, xpu>::Launch(
          s, out_data.Size(), out_data.dptr<DType>(), n);
    });
  });
}

struct LogspaceParam : public dmlc::Parameter<LogspaceParam> {
  double start;
  double stop;
  int num;
  bool endpoint;
  double base;
  std::string ctx;
  int dtype;
  DMLC_DECLARE_PARAMETER(LogspaceParam) {
    DMLC_DECLARE_FIELD(start)
    .describe("The starting value of the sequence.");
    DMLC_DECLARE_FIELD(stop)
    .describe("The ending value of the sequence");
    DMLC_DECLARE_FIELD(num)
    .describe("Number of samples to generate. Must be non-negative.");
    DMLC_DECLARE_FIELD(endpoint)
    .set_default(true)
    .describe("If True, stop is the last sample. Otherwise, it is not included.");
    DMLC_DECLARE_FIELD(base)
    .set_default(10.0)
    .describe("The base of the log space. The step size between the elements in "
    "ln(samples) / ln(base) (or log_base(samples)) is uniform.");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
    "Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype).set_default(mshadow::kFloat32)
    MXNET_ADD_ALL_TYPES
    .describe("Target data type.");
  }
};

struct logspace_fwd {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, double start, double stop, double base,
                                  double step, int req, DType* out) {
    KERNEL_ASSIGN(out[i], req,
                  static_cast<DType>(math::pow(base, static_cast<double>(start + step * i))));
  }
};

template<typename xpu>
void LogspaceCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const LogspaceParam& param = nnvm::get<LogspaceParam>(attrs.parsed);
  if (param.num == 0) return;
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      int step_num = param.endpoint ? param.num - 1 : param.num;
      double step = step_num > 0 ? (param.stop - param.start) / step_num : 0.0f;
      Kernel<logspace_fwd, xpu>::Launch(s, outputs[0].Size(), param.start, param.stop, param.base,
          step, req[0], outputs[0].dptr<DType>());
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_INIT_OP_H_
