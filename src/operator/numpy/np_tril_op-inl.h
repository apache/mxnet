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
 * \file np_tril_op-inl.h
 * \brief Function definition of the tril (lower triangle of an array) op
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_TRIL_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_TRIL_OP_INL_H_

#include <dmlc/parameter.h>
#include <vector>
#include <algorithm>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct TrilParam : public dmlc::Parameter<TrilParam> {
  int k;
  DMLC_DECLARE_PARAMETER(TrilParam) {
    DMLC_DECLARE_FIELD(k)
      .set_default(0)
      .describe("Diagonal in question. The default is 0. "
                "Use k>0 for diagonals above the main diagonal, "
                "and k<0 for diagonals below the main diagonal. "
                "If input has shape (S0 S1) k must be between -S0 and S1");
  }
};

inline bool TrilOpShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_attrs,
                             mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& ishape = (*in_attrs)[0];
  mxnet::TShape oshape;

  if (!mxnet::ndim_is_known(ishape)) {
    return false;
  }

  if (ishape.ndim() == 1) {
    auto s = ishape[0];
    oshape = mxnet::TShape({s, s});
  } else {
    oshape = ishape;
  }

  if (shape_is_none(oshape)) {
    LOG(FATAL) << "Diagonal does not exist.";
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);

  return shape_is_known(out_attrs->at(0));
}

template<int req>
struct tril1Dforward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out, const DType* data,
                                  mshadow::Shape<2> oshape, int k) {
    using namespace mxnet_op;

    const index_t row_id = i / oshape[1];
    const index_t col_id = i % oshape[1];
    if (col_id > (row_id + k)) {
      KERNEL_ASSIGN(out[i], req, static_cast<DType>(0));
    } else {
      KERNEL_ASSIGN(out[i], req, data[col_id]);
    }
  }
};

template<int req>
struct tril1Dbackward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out, const DType* data,
                                  mshadow::Shape<1> oshape, int k) {
    using namespace mxnet_op;
    auto m = oshape[0];
    auto start = (i > k) ? (i - k) : 0;
    DType res = 0;
    for (auto y = start; y < m; y++) {
      res += data[y * m + i];
    }
    KERNEL_ASSIGN(out[i], req, res);
  }
};

template<int req>
struct tril2D {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out, const DType* data,
                                  mshadow::Shape<2> oshape, int k) {
    using namespace mxnet_op;

    const index_t row_id = i / oshape[1];
    const index_t col_id = i % oshape[1];
    if (col_id > (row_id + k)) {
      KERNEL_ASSIGN(out[i], req, static_cast<DType>(0));
    } else {
      KERNEL_ASSIGN(out[i], req, data[i]);
    }
  }
};

template<int req>
struct tril3D {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out, const DType* data,
                                  mshadow::Shape<3> oshape, int k) {
    using namespace mxnet_op;

    const index_t row_id = i % (oshape[1] * oshape[2]) / oshape[2];
    const index_t col_id = i % (oshape[1] * oshape[2]) % oshape[2];
    if (col_id > (row_id + k)) {
      KERNEL_ASSIGN(out[i], req, static_cast<DType>(0));
    } else {
      KERNEL_ASSIGN(out[i], req, data[i]);
    }
  }
};

template<typename xpu, bool back>
void TrilOpProcess(const TBlob& in_data,
                   const TBlob& out_data,
                   index_t dsize,
                   const TrilParam& param,
                   mxnet_op::Stream<xpu> *s,
                   const std::vector<OpReqType>& req) {
  using namespace mxnet_op;
  using namespace mshadow;

  const mxnet::TShape& ishape = in_data.shape_;
  const mxnet::TShape& oshape = out_data.shape_;

  if (ishape.ndim() == 2 && oshape.ndim() == 2) {
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<tril2D<req_type>, xpu>::Launch(
            s, dsize, out_data.dptr<DType>(), in_data.dptr<DType>(),
            Shape2(oshape[0], oshape[1]), param.k);
      });
    });
  } else if (ishape.ndim() > 2) {
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<tril3D<req_type>, xpu>::Launch(
            s, dsize, out_data.dptr<DType>(), in_data.dptr<DType>(),
            oshape.FlatTo3D(oshape.ndim() - 2), param.k);
      });
    });
  } else {
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        if (back) {
          Kernel<tril1Dbackward<req_type>, xpu>::Launch(
              s, dsize, out_data.dptr<DType>(), in_data.dptr<DType>(),
              Shape1(oshape[0]), param.k);
        } else {
          Kernel<tril1Dforward<req_type>, xpu>::Launch(
              s, dsize, out_data.dptr<DType>(), in_data.dptr<DType>(),
              Shape2(oshape[0], oshape[1]), param.k);
        }
      });
    });
  }
}

template<typename xpu>
void TrilOpForward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  const TrilParam& param = nnvm::get<TrilParam>(attrs.parsed);

  TrilOpProcess<xpu, false>(in_data, out_data, out_data.Size(), param, s, req);
}

template<typename xpu>
void TrilOpBackward(const nnvm::NodeAttrs& attrs,
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
  const TrilParam& param = nnvm::get<TrilParam>(attrs.parsed);

  TrilOpProcess<xpu, true>(in_data, out_data, out_data.Size(), param, s, req);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_TRIL_OP_INL_H_
