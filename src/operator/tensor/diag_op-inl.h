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
* Copyright (c) 2015 by Contributors
* \file diag_op-inl.h
* \brief CPU Implementation of the diag op
* \author Istvan Fehervari
*/

#ifndef MXNET_OPERATOR_TENSOR_DIAG_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_DIAG_OP_INL_H_

#include <dmlc/parameter.h>
#include <vector>
#include <algorithm>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct DiagParam : public dmlc::Parameter<DiagParam> {
  dmlc::optional<int> k;
  DMLC_DECLARE_PARAMETER(DiagParam) {
    DMLC_DECLARE_FIELD(k)
    .set_default(dmlc::optional<int>(0))
    .describe("Diagonal in question. The default is 0. "
              "Use k>0 for diagonals above the main diagonal, "
              "and k<0 for diagonals below the main diagonal. "
              "If input has shape (S0 S1) k must be between -S0 and S1");
  }
};

inline TShape DiagShapeImpl(const TShape& ishape, const nnvm::dim_t k) {
  if (ishape.ndim() == 1) {
    auto s = ishape[0] + std::abs(k);
    return TShape({s, s});
  }

  auto h = ishape[0];
  auto w = ishape[1];

  if (k > 0) {
    w -= k;
  } else if (k < 0) {
    h += k;
  }

  auto s = std::min(h, w);
  if (s < 0) {
    s = 0;
  }

  return TShape({s});
}

inline bool DiagOpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_attrs,
                             std::vector<TShape>* out_attrs) {
    CHECK_EQ(in_attrs->size(), 1U);
    CHECK_EQ(out_attrs->size(), 1U);

    const TShape& ishape = (*in_attrs)[0];
    if (ishape.ndim() == 0) return false;
    if (ishape.ndim() > 2) LOG(FATAL) << "Input must be 1- or 2-d.";

    const DiagParam& param = nnvm::get<DiagParam>(attrs.parsed);

    TShape oshape = DiagShapeImpl(ishape, param.k.value());
    if (shape_is_none(oshape)) {
      LOG(FATAL) << "Diagonal does not exist.";
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);

    return out_attrs->at(0).ndim() != 0U;
}

inline bool DiagOpType(const nnvm::NodeAttrs& attrs,
                       std::vector<int> *in_attrs,
                       std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]);
  return (*out_attrs)[0] != -1;
}

template<int req>
struct diag {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* a,
                                  mshadow::Shape<2> ishape, int k) {
    using namespace mxnet_op;
    int j = 0;
    if (k > 0) {
      j = ravel(mshadow::Shape2(i, i + k), ishape);
    } else if (k < 0) {
      j = ravel(mshadow::Shape2(i - k, i), ishape);
    } else {
      j = ravel(mshadow::Shape2(i, i), ishape);
    }

    KERNEL_ASSIGN(out[i], req, a[j]);
  }
};

template<int req>
struct diag_gen {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* a,
                                  mshadow::Shape<2> oshape, int k) {
    using namespace mxnet_op;

    auto j = unravel(i, oshape);
    if (j[1] == (j[0] + k)) {
      auto l = j[0] < j[1] ? j[0] : j[1];
      KERNEL_ASSIGN(out[i], req, a[l]);
    } else {
      KERNEL_ASSIGN(out[i], req, static_cast<DType>(0));
    }
  }
};

template<typename xpu>
void DiagOpForward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteTo);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  const TShape& ishape = inputs[0].shape_;
  const TShape& oshape = outputs[0].shape_;
  const DiagParam& param = nnvm::get<DiagParam>(attrs.parsed);

  if (ishape.ndim() == 2) {
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<diag<req_type>, xpu>::Launch(s, out_data.Size(), out_data.dptr<DType>(),
                            in_data.dptr<DType>(), Shape2(ishape[0], ishape[1]), param.k.value());
      });
    });
  } else {
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<diag_gen<req_type>, xpu>::Launch(s, out_data.Size(), out_data.dptr<DType>(),
                            in_data.dptr<DType>(), Shape2(oshape[0], oshape[1]), param.k.value());
      });
    });
  }
}

template<typename xpu>
void DiagOpBackward(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  const TShape& ishape = inputs[0].shape_;
  const TShape& oshape = outputs[0].shape_;
  const DiagParam& param = nnvm::get<DiagParam>(attrs.parsed);

  if (oshape.ndim() == 2) {
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<diag_gen<req_type>, xpu>::Launch(s, out_data.Size(), out_data.dptr<DType>(),
                            in_data.dptr<DType>(), Shape2(oshape[0], oshape[1]), param.k.value());
      });
    });
  } else {
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<diag<req_type>, xpu>::Launch(s, out_data.Size(), out_data.dptr<DType>(),
                            in_data.dptr<DType>(), Shape2(ishape[0], ishape[1]), param.k.value());
      });
    });
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_DIAG_OP_INL_H_
