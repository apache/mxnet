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
* \author Istvan Fehervari, Zhijingcheng Yu
*/

#ifndef MXNET_OPERATOR_TENSOR_DIAG_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_DIAG_OP_INL_H_

#include <dmlc/parameter.h>
#include <vector>
#include <algorithm>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {

struct DiagParam : public dmlc::Parameter<DiagParam> {
  dmlc::optional<int> k;
  dmlc::optional<int> axis1;
  dmlc::optional<int> axis2;
  DMLC_DECLARE_PARAMETER(DiagParam) {
    DMLC_DECLARE_FIELD(k)
    .set_default(dmlc::optional<int>(0))
    .describe("Diagonal in question. The default is 0. "
              "Use k>0 for diagonals above the main diagonal, "
              "and k<0 for diagonals below the main diagonal. "
              "If input has shape (S0 S1) k must be between -S0 and S1");
    DMLC_DECLARE_FIELD(axis1)
    .set_default(dmlc::optional<int>(0))
    .describe("The first axis of the submatrix in question. Ignored when the input is a 1-D array.");
    DMLC_DECLARE_FIELD(axis2)
    .set_default(dmlc::optional<int>(1))
    .describe("The second axis of the submatrix in question. Ignored when the input is a 1-D array.");
  }
};

inline TShape DiagShapeImpl(const TShape& ishape, const nnvm::dim_t k, 
                            const nnvm::dim_t axis1, const nnvm::dim_t axis2) {
  if (ishape.ndim() == 1) {
    auto s = ishape[0] + std::abs(k);
    return TShape({s, s});
  }

  auto h = ishape[axis1];
  auto w = ishape[axis2];

  if (k > 0) {
    w -= k;
  } else if (k < 0) {
    h += k;
  }

  auto s = std::min(h, w);
  if (s < 0) {
    s = 0;
  }

  auto x1 = CheckAxis(axis1, ishape.ndim());
  auto x2 = CheckAxis(axis2, ishape.ndim());

  CHECK_NE(x1, x2) << "axis1 and axis2 cannot refer the the same axis " << x1;

  if (x1 > x2) {
    std::swap(x1, x2);
  }

  int n_dim = static_cast<int>(ishape.ndim()) - 1;
  TShape oshape(n_dim);

  // remove axis1 and axis2 and append the new axis to the end
  for (int i = 0; i < x1; i ++)
    oshape[i] = ishape[i];
  for (int i = x1; i < x2; i ++)
    oshape[i] = ishape[i + 1];
  for (int i = x2; i < n_dim; i ++)
    oshape[i] = ishape[i + 2];
  oshape[n_dim - 1] = s;

  return oshape;
}

inline bool DiagOpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_attrs,
                             std::vector<TShape>* out_attrs) {
    CHECK_EQ(in_attrs->size(), 1U);
    CHECK_EQ(out_attrs->size(), 1U);

    const TShape& ishape = (*in_attrs)[0];
    if (ishape.ndim() == 0) return false;

    const DiagParam& param = nnvm::get<DiagParam>(attrs.parsed);

    TShape oshape = DiagShapeImpl(ishape,
                                  param.k.value(), 
                                  param.axis1.value(),
                                  param.axis2.value());
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

template<int ndim, int req, bool back>
struct diag {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* a,
                                  mshadow::Shape<ndim> oshape,
                                  mshadow::Shape<ndim> ishape,
                                  int stride, int offset,
                                  int base) {
    using namespace mxnet_op;
    int idx = i / base;
    int j = ravel(unravel(idx, oshape), ishape) + offset + stride * (i - idx * base);
    if (back){
      KERNEL_ASSIGN(out[j], req, a[i]);
    } else{
      KERNEL_ASSIGN(out[i], req, a[j]);
    }
  }
};

template<int req, bool back>
struct diag_gen {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* a,
                                  mshadow::Shape<2> oshape, int k) {
    using namespace mxnet_op;

    auto j = unravel(i, oshape);
    if (j[1] == (j[0] + k)) {
      auto l = j[0] < j[1] ? j[0] : j[1];
      if (back){
        KERNEL_ASSIGN(out[l], req, a[i]);
      } else{
        KERNEL_ASSIGN(out[i], req, a[l]);
      }
    } else if (!back) {
      KERNEL_ASSIGN(out[i], req, static_cast<DType>(0));
    }
  }
};

template<typename xpu, bool back>
void DiagOpProcess(const TBlob& in_data, 
                   const TBlob& out_data,
                   const TShape& ishape,
                   const TShape& oshape,
                   int dsize,
                   const DiagParam& param,
                   mxnet_op::Stream<xpu> *s,
                   const std::vector<OpReqType>& req){
  using namespace mxnet_op;
  using namespace mshadow;
  if (ishape.ndim() > 1) {
    // input : (leading + i, body + i, trailing)
    int x1 = CheckAxis(param.axis1.value(), ishape.ndim());
    int x2 = CheckAxis(param.axis2.value(), ishape.ndim());

    int idim = ishape.ndim(), odim = oshape.ndim();

    int minx = x1, maxx = x2;
    if (minx > maxx)
      std::swap(minx, maxx);

    int oleading = 1, obody = 1, otrailing = 1;
    for (int i = 0; i < minx; i ++)
      oleading *= ishape[i];
    for (int i = minx + 1; i < maxx; i ++)
      obody *= ishape[i];
    for (int i = maxx + 1; i < idim; i ++)
      otrailing *= ishape[i];


    int ileading = oleading, 
        ibody = obody * ishape[minx],
        itrailing = otrailing * ishape[maxx];
    
    int stride1 = itrailing * obody,
        stride2 = otrailing;

    if (x1 == maxx)
      std::swap(stride1, stride2);
    int offset, k = param.k.value();
    if (k > 0) {
      offset = stride2 * k;
    } else if(k < 0) {
      offset = stride1 * -k;
    } else {
      offset = 0;
    }

    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        if(ileading == 1){
          Kernel<diag<2, req_type, back>, xpu>::Launch(s, dsize, out_data.dptr<DType>(),
                              in_data.dptr<DType>(), Shape2(obody, otrailing),
                              Shape2(ibody, itrailing),
                              stride1 + stride2, offset, oshape[odim - 1]);
        } else{
          Kernel<diag<3, req_type, back>, xpu>::Launch(s, dsize, out_data.dptr<DType>(),
                              in_data.dptr<DType>(), Shape3(oleading, obody, otrailing),
                              Shape3(ileading, ibody, itrailing),
                              stride1 + stride2, offset, oshape[odim - 1]);
        }
      });
    });
  } else {
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<diag_gen<req_type, back>, xpu>::Launch(s, dsize, out_data.dptr<DType>(),
                            in_data.dptr<DType>(), Shape2(oshape[0], oshape[1]), 
                            param.k.value());
      });
    });
  }
}

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

  DiagOpProcess<xpu, false>(in_data, out_data, ishape, oshape, out_data.Size(), param, s, req);
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


  DiagOpProcess<xpu, true>(in_data, out_data, oshape, ishape, in_data.Size(), param, s, req);
}


} // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_DIAG_OP_INL_H_
