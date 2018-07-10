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
* \brief
* \author Istvan Fehervari
*/

#ifndef MXNET_OPERATOR_DIAG_INL_H_
#define MXNET_OPERATOR_DIAG_INL_H_

#include <dmlc/parameter.h>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include <vector>

namespace mxnet {
namespace op {

struct DiagParam : public dmlc::Parameter<DiagParam> {
    uint32_t k;
    DMLC_DECLARE_PARAMETER(DiagParam) {
            DMLC_DECLARE_FIELD(k)
                    .set_default(0)
                    .describe("Diagonal in question. Only k=0 is supported"
                              "The default is 0. Use k>0 for diagonals above the main diagonal, and k<0 for diagonals below the main diagonal.");
    }
};

inline TShape DiagShapeImpl(const TShape& ishape) {
  if (ishape.ndim() == 1) {
    return TShape({ishape[0], ishape[0]});
  }
  
  return TShape({std::min(ishape[0], ishape[1])});
}

inline bool DiagOpShape(const nnvm::NodeAttrs& attrs, // contains k
                             std::vector<TShape>* in_attrs, // in shapes
                             std::vector<TShape>* out_attrs) { // out shapes
    CHECK_EQ(in_attrs->size(), 1U); // only one input data
    CHECK_EQ(out_attrs->size(), 1U); // only one output data

    const TShape& ishape = (*in_attrs)[0];
    if (ishape.ndim() == 0) return false;
    if (ishape.ndim() > 2) LOG(FATAL)
      << "Input must be 1- or 2-d.";

    const DiagParam& param = nnvm::get<DiagParam>(attrs.parsed);
    if (param.k != 0) LOG(FATAL)
       << "k != 0 is not supported by diag yet.";
    TShape oshape = DiagShapeImpl(ishape);
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
    
    return out_attrs->at(0).ndim() != 0U && out_attrs->at(0).Size() != 0U;
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

struct diag {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* a,
                                  mshadow::Shape<2> ishape) {
    using namespace mxnet_op;

    int j = ravel(mshadow::Shape2(i,i), ishape);
    out[i] = a[j];
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
  //const DiagParam& param = nnvm::get<DiagParam>(attrs.parsed); needed for k
  
  if (ishape.ndim() == 2) {
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      Kernel<diag, xpu>::Launch(s, out_data.Size(), out_data.dptr<DType>(),
                                in_data.dptr<DType>(), Shape2(ishape[0], ishape[1]));
    });
  } else {
    // TODO 1 dim input
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
  //const DiagParam& param = nnvm::get<DiagParam>(attrs.parsed); needed for k
  
  if (ishape.ndim() == 2) {
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      Kernel<diag, xpu>::Launch(s, out_data.Size(), out_data.dptr<DType>(),
                                in_data.dptr<DType>(), Shape2(ishape[0], ishape[1]));
    });
  } else {
    // TODO 1 dim input
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_DIAG_INL_H_