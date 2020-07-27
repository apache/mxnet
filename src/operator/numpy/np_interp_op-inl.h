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
 * Copyright (c) 2020 by Contributors
 * \file np_interp_op-inl.h
*/

#ifndef MXNET_OPERATOR_NUMPY_NP_INTERP_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_INTERP_OP_INL_H_

#include <vector>
#include <string>
#include <unordered_map>
#include "../tensor/ordering_op-inl.h"
#include "../tensor/matrix_op-inl.h"
#include "../tensor/elemwise_binary_scalar_op.h"
#include "../../common/utils.h"
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "np_broadcast_reduce_op.h"

namespace mxnet {
namespace op {

struct NumpyInterpParam : public dmlc::Parameter<NumpyInterpParam> {
  dmlc::optional<double> left;
  dmlc::optional<double> right;
  dmlc::optional<double> period;
  double x_scalar;
  bool x_is_scalar;
  DMLC_DECLARE_PARAMETER(NumpyInterpParam) {
    DMLC_DECLARE_FIELD(left)
      .set_default(dmlc::optional<double>())
      .describe("Value to return for x < xp[0], default is fp[0].");
    DMLC_DECLARE_FIELD(right)
      .set_default(dmlc::optional<double>())
      .describe("Value to return for x > xp[-1], default is fp[-1].");
    DMLC_DECLARE_FIELD(period)
      .set_default(dmlc::optional<double>())
      .describe("A period for the x-coordinates. This parameter allows"
                "the proper interpolation of angular x-coordinates. Parameters"
                "left and right are ignored if period is specified.");
    DMLC_DECLARE_FIELD(x_scalar).set_default(0.0)
      .describe("x is a scalar input");
    DMLC_DECLARE_FIELD(x_is_scalar).set_default(false)
      .describe("Flag that determines whether input is a scalar");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream left_s, right_s, period_s, x_scalar_s, x_is_scalar_s;
    left_s << left;
    right_s << right;
    period_s << period;
    x_scalar_s << x_scalar;
    x_is_scalar_s << x_is_scalar;
    (*dict)["left"] = left_s.str();
    (*dict)["right"] = right_s.str();
    (*dict)["period"] = period_s.str();
    (*dict)["x_scalar"] = x_scalar_s.str();
    (*dict)["x_is_scalar"] = x_is_scalar_s.str();
  }
};

struct interp {
  MSHADOW_XINLINE static void Map(int i,
                                  double* out,
                                  const double* x,
                                  const double* xp,
                                  const double* fp,
                                  const int dsize,
                                  const double left,
                                  const double right,
                                  const bool has_left,
                                  const bool has_right) {
    double x_value = x[i];
    double xp_low = xp[0];
    double xp_above = xp[dsize-1];
    double lval = has_left ? left : fp[0];
    double rval = has_right ? right : fp[dsize-1];

    if (x_value > xp_above) {
      out[i] = rval;
    } else if (x_value < xp_low) {
      out[i] = lval;
    } else {
      int imin = 0;
      int imax = dsize;
      int imid;
      while (imin < imax) {
        imid = static_cast<int>((imax + imin) / 2);
        if (x_value >= xp[imid]) {
          imin = imid + 1;
        } else {
          imax = imid;
        }
      }  // biserction search

      int j = imin;
      if (j == dsize) {
        out[i] = fp[dsize-1];
      } else if (x_value == xp[j-1]) {
        out[i] = fp[j-1];  // void potential non-finite interpolation
      } else {
        double xp_below = xp[j-1];
        double xp_above = xp[j];
        double weight_above = (x_value - xp_below) / (xp_above - xp_below);
        double weigth_below = 1 - weight_above;
        double x1 = fp[j-1] * weigth_below;
        double x2 = fp[j] * weight_above;
        out[i] = x1 + x2;
      }
    }
  }
};

struct interp_period {
  MSHADOW_XINLINE static void Map(int i,
                                  double* out,
                                  const double* x,
                                  const double* xp,
                                  const double* fp,
                                  const index_t* idx,
                                  const int dsize,
                                  const double period) {
    double x_value = x[i];
    int imin = 0;
    int imax = dsize;
    int imid;
    while (imin < imax) {
      imid = static_cast<int>((imax + imin) / 2);
      if (x_value >= xp[idx[imid]]) {
        imin = imid + 1;
      } else {
        imax = imid;
      }
    }  // biserction search

    int j = imin;
    double xp_below, xp_above;
    double fp1, fp2;
    if (j == 0) {
      xp_below = xp[idx[dsize-1]] - period;
      xp_above = xp[idx[0]];
      fp1 = fp[idx[dsize-1]];
      fp2 = fp[idx[0]];
    } else if (j == dsize) {
      xp_below = xp[idx[dsize-1]];
      xp_above = xp[idx[0]] + period;
      fp1 = fp[idx[dsize-1]];
      fp2 = fp[idx[0]];
    } else {
      xp_below = xp[idx[j-1]];
      xp_above = xp[idx[j]];
      fp1 = fp[idx[j-1]];
      fp2 = fp[idx[j]];
    }
    double weight_above = (x_value - xp_below) / (xp_above - xp_below);
    double weigth_below = 1 - weight_above;
    double x1 = fp1 * weigth_below;
    double x2 = fp2 * weight_above;
    out[i] = x1 + x2;
  }
};

template<typename xpu, typename OP>
void NumpyInterpForward(const nnvm::NodeAttrs& attrs,
                        const OpContext &ctx,
                        const std::vector<TBlob> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &outputs) {
  if (req[0] == kNullOp) return;
  using namespace mxnet;
  using namespace mxnet_op;
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_GE(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  Stream<xpu> *s = ctx.get_stream<xpu>();
  const NumpyInterpParam& param = nnvm::get<NumpyInterpParam>(attrs.parsed);
  dmlc::optional<double> left = param.left;
  dmlc::optional<double> right = param.right;
  dmlc::optional<double> period = param.period;
  bool x_is_scalar = param.x_is_scalar;

  TBlob xp = inputs[0];
  const TBlob &fp = inputs[1];
  const TBlob &out = outputs[0];
  bool has_left = left.has_value() ? true : false;
  bool has_right = right.has_value() ? true : false;
  double left_value = left.has_value() ? left.value() : 0.0;
  double right_value = right.has_value() ? right.value() : 0.0;

  CHECK_GE(xp.Size(), 1U) <<"ValueError: array of sample points is empty";

  TopKParam topk_param = TopKParam();
  topk_param.axis = dmlc::optional<int>(-1);
  topk_param.is_ascend = true;
  topk_param.k = 0;
  topk_param.ret_typ = topk_enum::kReturnIndices;

  size_t topk_temp_size;  // Used by Sort
  size_t topk_workspace_size = TopKWorkspaceSize<xpu, double>(xp, topk_param, &topk_temp_size);
  size_t size_x = x_is_scalar ? 8 : 0;
  size_t size_norm_x = x_is_scalar ? 8 : inputs[2].Size() * sizeof(double);
  size_t size_norm_xp = xp.Size() * sizeof(double);
  size_t size_norm = period.has_value()? size_norm_x + size_norm_xp : 0;
  size_t size_idx = period.has_value()? xp.Size() * sizeof(index_t) : 0;
  size_t workspace_size =
    topk_workspace_size + size_x + size_norm + size_idx;

  Tensor<xpu, 1, char> temp_mem =
    ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);

  char* workspace_curr_ptr = temp_mem.dptr_;

  TBlob x, idx;
  if (x_is_scalar) {
    double x_scalar = param.x_scalar;
    Tensor<cpu, 1, double> host_x(&x_scalar, Shape1(1), ctx.get_stream<cpu>());
    Tensor<xpu, 1, double> device_x(reinterpret_cast<double*>(workspace_curr_ptr),
                                    Shape1(1), ctx.get_stream<xpu>());
    Copy(device_x, host_x, ctx.get_stream<xpu>());
    x = TBlob(device_x.dptr_, TShape(0, 1), xpu::kDevMask);
    workspace_curr_ptr += 8;
  } else {
    x = inputs[2];
  }   // handle input x is a scalar

  // normalize the input data by periodic boundaries.
  if (period.has_value()) {
    double* norm_xp_ptr;
    double* norm_x_ptr;
    double period_value = period.value();
    index_t* idx_ptr;
    CHECK_NE(period_value, 0.0)<< "period must be a non-zero value";

    norm_xp_ptr = reinterpret_cast<double*>(workspace_curr_ptr);
    norm_x_ptr = reinterpret_cast<double*>(workspace_curr_ptr + size_norm_xp);
    idx_ptr = reinterpret_cast<index_t*>(workspace_curr_ptr + size_norm_xp + size_norm_x);

    TBlob norm_x = TBlob(norm_x_ptr, x.shape_, xpu::kDevMask);
    TBlob norm_xp = TBlob(norm_xp_ptr, xp.shape_, xpu::kDevMask);
    const OpReqType ReqType = kWriteTo;
    Kernel<op_with_req<OP, ReqType>, xpu>::Launch(
      s, x.Size(), norm_x.dptr<double>(), x.dptr<double>(), period_value);
    Kernel<op_with_req<OP, ReqType>, xpu>::Launch(
      s, xp.Size(), norm_xp.dptr<double>(), xp.dptr<double>(), period_value);

    workspace_curr_ptr += size_x + size_norm + size_idx;
    idx = TBlob(idx_ptr, xp.shape_, xpu::kDevMask);
    std::vector<OpReqType> req_TopK = {kWriteTo};
    std::vector<TBlob> ret = {idx};

    TopKImplwithWorkspace<xpu, double, index_t>(ctx.run_ctx, req_TopK, norm_xp, ret, topk_param,
                                                workspace_curr_ptr, topk_temp_size, s);
    Kernel<interp_period, xpu>::Launch(
      s, norm_x.Size(), out.dptr<double>(), norm_x.dptr<double>(), norm_xp.dptr<double>(),
      fp.dptr<double>(), idx.dptr<index_t>(), norm_xp.Size(), period_value);
  } else {
    Kernel<interp, xpu>::Launch(
      s, x.Size(), out.dptr<double>(), x.dptr<double>(), xp.dptr<double>(), fp.dptr<double>(),
      xp.Size(), left_value, right_value, has_left, has_right);
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_INTERP_OP_INL_H_
