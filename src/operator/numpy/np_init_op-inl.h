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
 * \file np_init_op-inl.h
 * \brief Function definition of numpy init op
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_INIT_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_INIT_OP_INL_H_

#include <dmlc/parameter.h>
#include <mxnet/operator_util.h>
#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

MSHADOW_XINLINE div_t my_div(int x, int y) {
  div_t result;
  result.quot = x / y;
  result.rem = x % y;
  return result;
}

struct NumpyDiagflatParam : public dmlc::Parameter<NumpyDiagflatParam> {
  int k;
  DMLC_DECLARE_PARAMETER(NumpyDiagflatParam) {
    DMLC_DECLARE_FIELD(k)
        .set_default(0)
        .describe("Diagonal to set."
                  "0, the default, corresponds to the \"main\" diagonal."
                  "a positive (negative) k giving the number of"
                  "the diagonal above (below) the main.");
  }
};

template<int req>
struct numpy_diagflat {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  DType *out_data,
                                  const DType *in_data,
                                  dim_t diag_len,
                                  int k) {
    using namespace mxnet_op;
    using namespace mshadow;

    if (diag_len == 0) {
      return;
    }

    // recover the original diagonal len
    auto orig_diag_len = diag_len - abs(k);

    div_t divmod;
    if (k >= 0) {
      divmod = my_div(static_cast<int>(i - k), static_cast<int>(diag_len + 1));
    } else {
      divmod = my_div(static_cast<int>(i + k * diag_len),
                   static_cast<int>(diag_len + 1));
    }
    DType to_write;
    // if the coord lies on the shifted diagonal and actually lies in the matrix
    if (divmod.rem == 0 && divmod.quot >= 0 && divmod.quot < orig_diag_len) {
      auto in_idx = divmod.quot;
      to_write = in_data[in_idx];
    } else {
      to_write = 0;
    }
    KERNEL_ASSIGN(out_data[i], req, to_write);
  }
};

template<typename xpu>
void NumpyDiagflatOpForward(const nnvm::NodeAttrs &attrs,
                            const OpContext &ctx,
                            const std::vector<TBlob> &inputs,
                            const std::vector<OpReqType> &req,
                            const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);  // only one input
  CHECK_EQ(outputs.size(), 1U);  // only one output
  CHECK_EQ(req.size(), 1U);  // only one req
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob &in_data = inputs[0];
  const TBlob &out_data = outputs[0];
  // get the diagonal length
  const mxnet::TShape &out_shape = outputs[0].shape_;
  CHECK_EQ(out_shape.ndim(), 2);
  auto &diag_len = *out_shape.data();
  // get k
  const NumpyDiagflatParam &param = nnvm::get<NumpyDiagflatParam>(attrs.parsed);

  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<numpy_diagflat<req_type>, xpu>::Launch(s,
                                                      out_data.Size(),
                                                      out_data.dptr<DType>(),
                                                      in_data.dptr<DType>(),
                                                      diag_len,
                                                      param.k);
    });
  });
}

template<int req>
struct numpy_diagflat_backward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  DType *in_grad,
                                  const DType *out_grad,
                                  dim_t diag_len,
                                  int k) {
    using namespace mxnet_op;
    using namespace mshadow;

    if (diag_len == 0) {
      return;
    }

    // recover the original diagonal len
    auto orig_diag_len = diag_len - abs(k);

    div_t divmod;
    if (k >= 0) {
      divmod = my_div(static_cast<int>(i - k), static_cast<int>(diag_len + 1));
    } else {
      divmod = my_div(static_cast<int>(i + k * diag_len),
                   static_cast<int>(diag_len + 1));
    }
    // if the coord lies on the shifted diagonal and actually lies in the matrix
    if (divmod.rem == 0 && divmod.quot >= 0 && divmod.quot < orig_diag_len) {
      auto in_idx = divmod.quot;
      KERNEL_ASSIGN(in_grad[in_idx], req, out_grad[i]);
    }
  }
};

template<typename xpu>
void NumpyDiagflatOpBackward(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);  // only use out grad
  CHECK_EQ(outputs.size(), 1U);  // only use input grad
  CHECK_EQ(req.size(), 1U);  // only one req
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob &out_grad = inputs[0];
  const TBlob &in_grad = outputs[0];

  const mxnet::TShape &out_shape = inputs[0].shape_;
  CHECK_EQ(out_shape.ndim(), 2);
  auto &diag_len = *out_shape.data();

  const NumpyDiagflatParam &param = nnvm::get<NumpyDiagflatParam>(attrs.parsed);

  MSHADOW_TYPE_SWITCH(out_grad.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<numpy_diagflat_backward<req_type>, xpu>::Launch(
            s,
            out_grad.Size(),
            in_grad.dptr<DType>(),
            out_grad.dptr<DType>(),
            diag_len,
            param.k);
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_INIT_OP_INL_H_
