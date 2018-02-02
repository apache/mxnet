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
 * \file regression_ouput-inl.h
 * \brief Regression output operator.
*/
#ifndef MXNET_OPERATOR_REGRESSION_OUTPUT_INL_H_
#define MXNET_OPERATOR_REGRESSION_OUTPUT_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <utility>
#include "./mshadow_op.h"
#include "./mxnet_op.h"
#include "./operator_common.h"
#include "./tensor/elemwise_binary_op.h"

namespace mxnet {
namespace op {

/*!
 * \brief regression namespace
 */
namespace reg_enum {
enum RegressionOutputOpInputs {kData, kLabel};
enum RegressionOutputOutputs {kOut};
}  // reg_enum

struct RegressionOutputParam : public dmlc::Parameter<RegressionOutputParam> {
  float grad_scale;
  DMLC_DECLARE_PARAMETER(RegressionOutputParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scale the gradient by a float factor");
  };
};

inline bool RegressionOpShape(const nnvm::NodeAttrs& attrs,
                              std::vector<TShape> *in_attrs,
                              std::vector<TShape> *out_attrs) {
  using namespace mshadow;
  CHECK_EQ(in_attrs->size(), 2U) << "Input:[data, label]";
  const TShape &dshape = in_attrs->at(0);
  if (dshape.ndim() == 0) return false;
  auto &lshape = (*in_attrs)[1];
  if (lshape.ndim() == 0) {
    // special treatment for 1D output, to allow 1D label by default.
    // Think about change convention later
    if (dshape.ndim() == 2 && dshape[1] == 1) {
      lshape = Shape1(dshape[0]);
    } else {
      lshape = dshape;
    }
  } else if (lshape[0] != dshape[0] || lshape.Size() != dshape.Size()) {
    std::ostringstream os;
    os << "Shape inconsistent, Provided=" << lshape << ','
       << " inferred shape=" << dshape;
    throw ::mxnet::op::InferShapeError(os.str(), 1);
  }
  out_attrs->clear();
  out_attrs->push_back(dshape);
  return true;
}

template<int label_pos>
inline bool RegressionInferStorageType(const nnvm::NodeAttrs& attrs,
                                       const int dev_mask,
                                       DispatchMode* dispatch_mode,
                                       std::vector<int>* in_attrs,
                                       std::vector<int>* out_attrs) {
  const auto label_stype = in_attrs->at(label_pos);
  auto& out_stype = out_attrs->at(0);
  bool dispatched = false;
  if (!dispatched && label_stype == kDefaultStorage) {
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }

  if (!dispatched && label_stype == kCSRStorage) {
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }

  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  // In backward pass, although we don't care about gradients of label,
  // a storage type should be assigned to it.
  if (out_attrs->size() > 1) type_assign(&out_attrs->at(1), kDefaultStorage);

  return dispatched;
}

template<typename xpu, typename ForwardOp>
inline void RegressionForwardImpl(mshadow::Stream<xpu> *s, const OpReqType req,
                                  const TBlob &data, const TBlob &out) {
  MSHADOW_REAL_TYPE_SWITCH(data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req, Req, {
      const DType* in_data = data.dptr<DType>();
      DType* out_data = out.dptr<DType>();
      using namespace mxnet_op;
      Kernel<op_with_req<ForwardOp, Req>, xpu>::Launch(
        s, out.Size(), out_data, in_data);
    });
  });
}

template<typename xpu, typename ForwardOp>
void RegressionForward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  RegressionForwardImpl<xpu, ForwardOp>(s, req[reg_enum::kOut],
    inputs[reg_enum::kData], outputs[reg_enum::kOut]);
}

template<typename xpu, typename ForwardOp>
void RegressionForwardEx(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<NDArray>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  RegressionForwardImpl<xpu, ForwardOp>(s, req[reg_enum::kOut],
    inputs[reg_enum::kData].data(), outputs[reg_enum::kOut].data());
}

template<typename xpu, typename BackwardOp>
void RegressionBackward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2);
  CHECK_EQ(outputs.size(), 2);
  const RegressionOutputParam& param = nnvm::get<RegressionOutputParam>(attrs.parsed);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  // inputs are in_label, out_data
  // outputs are data_grad, label_grad
  MSHADOW_REAL_TYPE_SWITCH(inputs[1].type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      const DType* in_label = inputs[0].dptr<DType>();
      const DType* out_data = inputs[1].dptr<DType>();
      DType* data_grad = outputs[0].dptr<DType>();
      const real_t num_output = inputs[0].Size()/inputs[0].shape_[0];
      using namespace mxnet_op;
      Kernel<op_with_req<BackwardOp, Req>, xpu>::Launch(
        s, outputs[0].Size(), data_grad, out_data, in_label);
      Kernel<op_with_req<mshadow_op::mul, Req>, xpu>::Launch(
        s, outputs[0].Size(), data_grad, data_grad,
        static_cast<DType>(param.grad_scale/num_output));
    });
  });
}


template<typename xpu, typename BackwardOp>
inline void RegressionBackwardCSRImpl(mshadow::Stream<xpu> *s,
                                      const RegressionOutputParam& param,
                                      const OpReqType req,
                                      const NDArray &data, const NDArray &label,
                                      const NDArray &data_grad, const bool sparse_kernel) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace csr;
  const TShape dshape = data.shape();
  const nnvm::dim_t num_rows = dshape[0];
  const nnvm::dim_t row_length = dshape[1];
  MSHADOW_IDX_TYPE_SWITCH(label.aux_type(kIndPtr), RType, {
    MSHADOW_IDX_TYPE_SWITCH(label.aux_type(kIdx), IType, {
      MSHADOW_REAL_TYPE_SWITCH(label.dtype(), DType, {
        MXNET_ASSIGN_REQ_SWITCH(req, Req, {
          const RType* label_indptr = label.aux_data(kIndPtr).dptr<RType>();
          const IType* label_idx = label.aux_data(kIdx).dptr<IType>();
          const DType* label_data = label.data().dptr<DType>();
          const DType* data_ptr = data.data().dptr<DType>();
          DType* grad_ptr = data_grad.data().dptr<DType>();
          if (sparse_kernel) {
            Kernel<op_with_req<mshadow_op::identity, Req>, xpu>::Launch(s,
              dshape.Size(), grad_ptr, data_ptr);
            Kernel<DnsCsrSparseKernel<BackwardOp, Req>, xpu>::Launch(s, num_rows,
              grad_ptr, data_ptr, label_data, label_idx, label_indptr, row_length);
          } else {
            Kernel<DnsCsrKernel<BackwardOp, Req>, xpu>::Launch(s, num_rows,
              grad_ptr, data_ptr, label_data, label_idx, label_indptr, row_length);
          }
          Kernel<op_with_req<mshadow_op::mul, Req>, xpu>::Launch(s, dshape.Size(),
            grad_ptr, grad_ptr, static_cast<DType>(param.grad_scale/row_length));
        });
      });
    });
  });
}


template<typename xpu, typename BackwardOP, bool sparse_kernel>
void RegressionBackwardEx(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 2U);
  const RegressionOutputParam& param = nnvm::get<RegressionOutputParam>(attrs.parsed);
  auto label_stype = inputs[0].storage_type();
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  if (label_stype == kCSRStorage) {
    RegressionBackwardCSRImpl<xpu, BackwardOP>(s, param, req[0], inputs[1], inputs[0],
      outputs[0], sparse_kernel);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

struct RegressionOpGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> heads;
    heads.push_back(n->inputs[reg_enum::kLabel]);
    heads.emplace_back(nnvm::NodeEntry{n, reg_enum::kOut, 0});
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_REGRESSION_OUTPUT_INL_H_
