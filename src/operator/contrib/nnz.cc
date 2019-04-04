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
 *  Copyright (c) 2018 by Contributors
 * \file nnz.cc
 * \brief CPU Implementation of nnz operator
 */
#include <mxnet/operator_util.h>
#include <vector>
#include <limits>
#include <algorithm>
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

struct NNZParam : public dmlc::Parameter<NNZParam> {
  dmlc::optional<int> axis;
  DMLC_DECLARE_PARAMETER(NNZParam) {
    DMLC_DECLARE_FIELD(axis)
    .set_default(dmlc::optional<int>())
    .describe("Select between the number of values across the whole matrix, "
              "in each column, or in each row.");
  }
};

static bool NNZType(const nnvm::NodeAttrs& attrs,
                    std::vector<int> *in_attrs,
                    std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  // infer int64 for count
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt64);
  return true;
}

inline bool NNZShape(const nnvm::NodeAttrs& attrs,
                     std::vector<TShape> *in_attrs,
                     std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  // csr_matrix is 2-D
  CHECK_EQ(in_attrs->at(0).ndim(), 2);
  const NNZParam& param = nnvm::get<NNZParam>(attrs.parsed);
  // whole matrix
  if (!param.axis.has_value()) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape1(1));
  } else if (param.axis.value() == 0) {
    // columns
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape1(in_attrs->at(0)[1]));
  } else if (param.axis.value() == 1) {
    // rows
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape1(in_attrs->at(0)[0]));
  } else {
    LOG(FATAL) << "Unexpected value for axis(" << param.axis.value()
      << "). Candidates are None, 0, and 1";
  }
  return true;
}

template<typename xpu>
void NNZComputeCsrImpl(const NNZParam& param,
                       const OpContext& ctx,
                       const NDArray& input,
                       const OpReqType req,
                       const TBlob& output);

struct CsrNNZRowKernel {
  /*!
   * \brief Map function for general case of take grad
   * \param tid           global thread id
   * \param out           ptr to output
   * \param indptr        ptr to source csr indptr
   */
  template<typename IType, typename DType>
  MSHADOW_XINLINE static void Map(int tid, DType* out, const IType* indptr) {
    out[tid] = static_cast<DType>(indptr[tid + 1] - indptr[tid]);
  }
};

template<>
void NNZComputeCsrImpl<cpu>(const NNZParam& param,
                            const OpContext& ctx,
                            const NDArray& input,
                            const OpReqType req,
                            const TBlob& output) {
  using namespace csr;
  CHECK_EQ(req, kWriteTo);
  mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
  if (!input.storage_initialized()) {
    Fill<false>(s, output, kWriteTo, 0);
    return;
  }
  MSHADOW_TYPE_SWITCH(output.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(input.aux_type(kIndPtr), IType, {
      DType* out_ptr = output.dptr<DType>();
      const IType* indptr = input.aux_data(kIndPtr).dptr<IType>();
      const nnvm::dim_t num_rows = input.shape()[0];
      if (!param.axis.has_value()) {
        // whole matrix
        out_ptr[0] = indptr[num_rows];
      } else if (param.axis.value() == 0) {
        // column
        LOG(FATAL) << "getnnz with axis = 0 is not supported yet";
      } else if (param.axis.value() == 1) {
        // row
        mxnet_op::Kernel<CsrNNZRowKernel, cpu>::Launch(s, num_rows, out_ptr, indptr);
      }
    });
  });
}

template<typename xpu>
void NNZComputeEx(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<NDArray>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const auto in_stype = inputs[0].storage_type();
  const auto out_stype = outputs[0].storage_type();
  const NNZParam& param = nnvm::get<NNZParam>(attrs.parsed);
  if (in_stype == kCSRStorage && out_stype == kDefaultStorage) {
    NNZComputeCsrImpl<xpu>(param, ctx, inputs[0], req[0], outputs[0].data());
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

bool NNZStorageType(const nnvm::NodeAttrs& attrs,
                    const int dev_mask,
                    DispatchMode* dispatch_mode,
                    std::vector<int> *in_attrs,
                    std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  bool dispatched = false;
  const auto in_stype = in_attrs->at(0);
  auto& out_stype = out_attrs->at(0);
  // only support csr for now
  if (!dispatched && in_stype == kCSRStorage) {
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
  return dispatched;
}

DMLC_REGISTER_PARAMETER(NNZParam);

NNVM_REGISTER_OP(_contrib_getnnz)
.describe(R"code(Number of stored values for a sparse tensor, including explicit zeros.

This operator only supports CSR matrix on CPU.

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NNZParam>)
.set_attr<nnvm::FInferShape>("FInferShape", NNZShape)
.set_attr<nnvm::FInferType>("FInferType", NNZType)
.set_attr<FInferStorageType>("FInferStorageType", NNZStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", NNZComputeEx<cpu>)
.add_argument("data", "NDArray-or-Symbol", "Input")
.add_arguments(NNZParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
