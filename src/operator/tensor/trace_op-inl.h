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
* Copyright (c) 2018 by Contributors
* \file trace_op-inl.h
* \brief CPU Implementation of the trace op
* \author Sam Skalicky
*/

#ifndef MXNET_OPERATOR_TENSOR_TRACE_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_TRACE_OP_INL_H_

#include <dmlc/parameter.h>
#include <vector>
#include <algorithm>
#include <utility>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {

struct TraceParam : public dmlc::Parameter<TraceParam> {
  int k;
  int32_t axis1;
  int32_t axis2;
  DMLC_DECLARE_PARAMETER(TraceParam) {
    DMLC_DECLARE_FIELD(k)
      .set_default(0)
      .describe("Diagonal offset. The default is 0. "
                "Use k>0 for diagonals above the main diageonal, "
                "and k<0 for diagonals below the main diagoonal. "
                "If input has shape (S0 S1) k must be between -S0 and S1");
    DMLC_DECLARE_FIELD(axis1)
      .set_default(0)
      .describe("The first axis of the sub-arrays of interest. "
                "Ignored when the input is a 1-D array.");
    DMLC_DECLARE_FIELD(axis2)
      .set_default(1)
      .describe("The second axis of the sub-arrays of interest. "
                "Ignored when the input is a 1-D array.");
  }
};

inline TShape TraceShapeImpl(const TShape& ishape, const int k,
                            const uint32_t axis1, const uint32_t axis2) {
  uint32_t n_dim, idim = ishape.ndim();
  if (idim > 2)  // for +3D we remove the two axis along the diagonal
    n_dim = idim - 2;
  else  // if its 2D then the output will be a single result, so 1-dim 1 element output
    n_dim = 1;

  TShape oshape(n_dim);

  // add all axes except the two specified along the diagonal
  uint32_t idx = 0;
  for (uint32_t i = 0; i < idim; ++i) {
    if (i != axis1 && i != axis2) {
      oshape[idx++] = ishape[i];
    }
  }

  return oshape;
}

inline bool TraceOpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_attrs,
                             std::vector<TShape>* out_attrs) {
    CHECK_EQ(in_attrs->size(), 1U);
    CHECK_EQ(out_attrs->size(), 1U);

    const TShape& ishape = (*in_attrs)[0];

    uint32_t idim = ishape.ndim();
    if (idim < 2) {
      // trace is undefined for 1D arrays
      return false;
    }

    const TraceParam& param = nnvm::get<TraceParam>(attrs.parsed);

    uint32_t x1 = CheckAxis(param.axis1, idim);
    uint32_t x2 = CheckAxis(param.axis2, idim);

    TShape oshape = TraceShapeImpl(ishape,
                                  param.k,
                                  x1,
                                  x2);

    if (shape_is_none(oshape)) {
      LOG(FATAL) << "diagonal does not exist.";
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);

    return out_attrs->at(0).ndim() != 0U;
}

inline bool TraceOpType(const nnvm::NodeAttrs& attrs,
                       std::vector<int> *in_attrs,
                       std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  // propagate types from input to output (or vice-versa)
  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]);
  return (*out_attrs)[0] != -1;
}

template<int ndim, int req, bool back>
struct trace {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out, const DType* a,
                                  mshadow::Shape<ndim> oshape,
                                  mshadow::Shape<ndim> ishape,
                                  index_t stride, index_t offset,
                                  index_t base) {
    using namespace mxnet_op;

    index_t idx = i / base;
    index_t j = ravel(unravel(idx, oshape), ishape) + offset + stride * (i - idx * base);
    out[idx] += a[j];
  }
};

template<typename xpu, bool back>
void TraceOpProcess(const TBlob& in_data,
                   const TBlob& out_data,
                   const TShape& ishape,
                   const TShape& oshape,
                   index_t dsize,
                   const TraceParam& param,
                   mxnet_op::Stream<xpu> *s,
                   const std::vector<OpReqType>& req) {
  using namespace mxnet_op;
  using namespace mshadow;

  // input : (leading + i, body + i, trailing)
  uint32_t x1 = CheckAxis(param.axis1, ishape.ndim());
  uint32_t x2 = CheckAxis(param.axis2, ishape.ndim());

  CHECK_NE(x1, x2) << "axis1 and axis2 cannot refer to the the same axis " << x1;

  uint32_t idim = ishape.ndim();
  uint32_t minx = x1, maxx = x2;
  if (minx > maxx) {
    std::swap(minx, maxx);
  }

  // merges contiguous axes that are not separated
  // by axis1 or axis2 since they can be directly
  // mapped to the output and there is no need
  // to distinguish them
  // (After this the input will have no more than
  // three axes, hence improving the ravel and
  // unravel efficiency)

  index_t oleading = 1,
    obody = 1,
    otrailing = 1;

  for (uint32_t i = 0; i < minx; ++i) {
    oleading *= ishape[i];
  }
  for (uint32_t i = minx + 1; i < maxx; ++i) {
    obody *= ishape[i];
  }
  for (uint32_t i = maxx + 1; i < idim; ++i) {
    otrailing *= ishape[i];
  }

  index_t ileading = oleading,
    ibody = obody * ishape[minx],
    itrailing = otrailing * ishape[maxx];

  index_t stride1 = itrailing * obody,
    stride2 = otrailing;
  // stride1 + stride2 is the stride for
  // iterating over the diagonal in question

  if (x1 == maxx) {
    std::swap(stride1, stride2);
  }

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

  auto h = ishape[x1];
  auto w = ishape[x2];

  if (k > 0) {
    w -= k;
  } else if (k < 0) {
    h += k;
  }

  auto diag_size = std::min(h, w);
  if (diag_size < 0) {
    diag_size = 0;
  }

  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
    std::memset(out_data.dptr<DType>(), 0, sizeof(DType)*out_data.Size());
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      if (ileading == 1) {
        Kernel<trace<2, req_type, back>, xpu>::Launch(s, dsize*diag_size,
          out_data.dptr<DType>(), in_data.dptr<DType>(),
          Shape2(obody, otrailing), Shape2(ibody, itrailing),
          stride1 + stride2, offset, diag_size);
      } else {
        Kernel<trace<3, req_type, back>, xpu>::Launch(s, dsize*diag_size,
          out_data.dptr<DType>(), in_data.dptr<DType>(),
          Shape3(oleading, obody, otrailing), Shape3(ileading, ibody, itrailing),
          stride1 + stride2, offset, diag_size);
      }
    });
  });
}

template<typename xpu>
void TraceOpForward(const nnvm::NodeAttrs& attrs,
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
  const TraceParam& param = nnvm::get<TraceParam>(attrs.parsed);

  TraceOpProcess<xpu, false>(in_data, out_data, ishape, oshape, out_data.Size(), param, s, req);
}

template<typename xpu>
void TraceOpBackward(const nnvm::NodeAttrs& attrs,
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
  const TraceParam& param = nnvm::get<TraceParam>(attrs.parsed);

  TraceOpProcess<xpu, true>(in_data, out_data, oshape, ishape, in_data.Size(), param, s, req);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_TRACE_OP_INL_H_
