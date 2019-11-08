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
 * \file diag_op-inl.h
 * \brief Function definition of the numpy-compatible diag op
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_DIAG_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_DIAG_OP_INL_H_

#include "../elemwise_op_common.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../tensor/broadcast_reduce_op.h"
#include <algorithm>
#include <dmlc/parameter.h>
#include <utility>
#include <vector>

namespace mxnet {
namespace op {

struct NumpyDiagParam : public dmlc::Parameter<NumpyDiagParam> {
  int k;
  DMLC_DECLARE_PARAMETER(NumpyDiagParam) {
    DMLC_DECLARE_FIELD(k).set_default(0).describe(
        "Diagonal in question. The default is 0. "
        "Use k>0 for diagonals above the main diagonal, "
        "and k<0 for diagonals below the main diagonal. ");
  }
};

inline mxnet::TShape NumpyDiagShapeImpl(const mxnet::TShape &ishape,
                                        const int k) {
  CHECK_LE(ishape.ndim(), 2) << "Input must be 1- or 2-d";

  if (ishape.ndim() == 1) {
    auto s = ishape[0] + std::abs(k);
    return mxnet::TShape({s, s});
  }

  auto h = ishape[0];
  auto w = ishape[1];

  if (k > 0) {
    w -= k;
  } else if (k < 0) {
    h += k;
  }
  dim_t a = 0;
  auto s = std::max(std::min(h, w), a);
  // s is the length of diagonal with k as the offset

  int32_t n_dim = ishape.ndim() - 1;
  mxnet::TShape oshape(n_dim, -1);
  oshape[n_dim - 1] = s;
  return oshape;
}

inline bool NumpyDiagOpShape(const nnvm::NodeAttrs &attrs,
                             mxnet::ShapeVector *in_attrs,
                             mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape &ishape = (*in_attrs)[0];
  if (!mxnet::ndim_is_known(ishape)) {
    return false;
  }

  const NumpyDiagParam &param = nnvm::get<NumpyDiagParam>(attrs.parsed);
  mxnet::TShape oshape = NumpyDiagShapeImpl(ishape, param.k);

  if (shape_is_none(oshape)) {
    LOG(FATAL) << "Diagonal does not exist.";
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);

  return shape_is_known(out_attrs->at(0));
}

inline bool NumpyDiagOpType(const nnvm::NodeAttrs &attrs,
                            std::vector<int> *in_attrs,
                            std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]);
  return (*out_attrs)[0] != -1;
}

template <int ndim, int req, bool back> struct diag {
  template <typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType *a,
                                  index_t stride, index_t offset) {
    using namespace mxnet_op;
    index_t j = offset + stride * i;

    if (back) {
      KERNEL_ASSIGN(out[j], req, a[i]);
    } else {
      KERNEL_ASSIGN(out[i], req, a[j]);
    }
  }
};

template <int req, bool back> struct diag_gen {
  template <typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType *a,
                                  mshadow::Shape<2> oshape, int k) {
    using namespace mxnet_op;

    auto j = unravel(i, oshape);
    if (j[1] == (j[0] + k)) {
      auto l = j[0] < j[1] ? j[0] : j[1];
      if (back) {
        KERNEL_ASSIGN(out[l], req, a[i]);
      } else {
        KERNEL_ASSIGN(out[i], req, a[l]);
      }
    } else if (!back) {
      KERNEL_ASSIGN(out[i], req, static_cast<DType>(0));
    }
  }
};

template <typename xpu, bool back>
void NumpyDiagOpProcess(const TBlob &in_data, const TBlob &out_data,
                        const mxnet::TShape &ishape,
                        const mxnet::TShape &oshape, index_t dsize,
                        const NumpyDiagParam &param, mxnet_op::Stream<xpu> *s,
                        const std::vector<OpReqType> &req) {
  using namespace mxnet_op;
  using namespace mshadow;
  if (ishape.ndim() > 1) {
    index_t stride1 = ishape[1], stride2 = 1;
    // stride1 + stride2 is the stride for
    // iterating over the diagonal in question

    // the extra index offset introduced by k
    index_t offset;
    int k = param.k;
    if (k > 0) {
      offset = stride2 * k;
    } else if (k < 0) {
      offset = stride1 * -k;
    } else {
      offset = 0;
    }

    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        if (back && req[0] != kAddTo && req[0] != kNullOp) {
          out_data.FlatTo1D<xpu, DType>(s) = 0;
        }

        Kernel<diag<2, req_type, back>, xpu>::Launch(
            s, dsize, out_data.dptr<DType>(), in_data.dptr<DType>(),
            stride1 + stride2, offset);
      });
    });
  } else {
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<diag_gen<req_type, back>, xpu>::Launch(
            s, dsize, out_data.dptr<DType>(), in_data.dptr<DType>(),
            Shape2(oshape[0], oshape[1]), param.k);
      });
    });
  }
}

template <typename xpu>
void NumpyDiagOpForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                        const std::vector<TBlob> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteTo);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob &in_data = inputs[0];
  const TBlob &out_data = outputs[0];
  const mxnet::TShape &ishape = inputs[0].shape_;
  const mxnet::TShape &oshape = outputs[0].shape_;
  const NumpyDiagParam &param = nnvm::get<NumpyDiagParam>(attrs.parsed);

  NumpyDiagOpProcess<xpu, false>(in_data, out_data, ishape, oshape,
                                 out_data.Size(), param, s, req);
}

template <typename xpu>
void NumpyDiagOpBackward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                         const std::vector<TBlob> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  const TBlob &in_data = inputs[0];
  const TBlob &out_data = outputs[0];
  const mxnet::TShape &ishape = inputs[0].shape_;
  const mxnet::TShape &oshape = outputs[0].shape_;
  const NumpyDiagParam &param = nnvm::get<NumpyDiagParam>(attrs.parsed);

  NumpyDiagOpProcess<xpu, true>(in_data, out_data, oshape, ishape,
                                in_data.Size(), param, s, req);
}

} // namespace op
} // namespace mxnet

#endif // MXNET_OPERATOR_NUMPY_NP_DIAG_OP_INL_H_
