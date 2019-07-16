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
 * \file indexing_op.h
 * \brief Function definition of numpy indexing operator
 */

#ifndef MXNET_OPERATOR_NUMPY_INDEXING_OP_H_
#define MXNET_OPERATOR_NUMPY_INDEXING_OP_H_

#include <dmlc/parameter.h>
#include <mxnet/operator_util.h>
#include <vector>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../tensor/indexing_op.h"

namespace mxnet {
namespace op {

struct NumpyTakeParam: public dmlc::Parameter<NumpyTakeParam> {
  dmlc::optional<int> axis;
  int mode;
  DMLC_DECLARE_PARAMETER(NumpyTakeParam) {
    DMLC_DECLARE_FIELD(axis)
    .set_default(dmlc::optional<int>())
    .describe("The axis of input array to be taken."
              "For input tensor of rank r, it could be in the range of [-r, r-1]");
    DMLC_DECLARE_FIELD(mode)
    .add_enum("raise", take_::kRaise)
    .add_enum("wrap", take_::kWrap)
    .add_enum("clip", take_::kClip)
    .set_default(take_::kClip)
    .describe("Specify how out-of-bound indices bahave. Default is \"clip\"."
              " \"clip\" means clip to the range. So, if all indices mentioned are too large,"
              " they are replaced by the index that addresses the last element along an axis."
              " \"wrap\" means to wrap around."
              " \"raise\" means to raise an error, not supported yet.");
  }
};

inline bool NumpyTakeOpShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector *in_attrs,
                             mxnet::ShapeVector *out_attrs) {
  using namespace mshadow;
  const mxnet::TShape &arrshape = (*in_attrs)[take_::kArr];
  const mxnet::TShape &idxshape = (*in_attrs)[take_::kIdx];
  if (!shape_is_known(idxshape)) {
    LOG(FATAL) << "Shape of indices is unknown...";
    return false;
  }
  const NumpyTakeParam& param = nnvm::get<NumpyTakeParam>(attrs.parsed);
  out_attrs->clear();
  if (param.axis.has_value()) {
    CHECK(param.axis.value() >= -1 * arrshape.ndim() && param.axis.value() < arrshape.ndim())
      << "If axis is not None, axis should be in the range of [-r, r-1] where r is the rank of input tensor";
    const index_t actual_axis = param.axis.value() + ((param.axis.value() < 0) ? arrshape.ndim() : 0);
    mxnet::TShape oshape(idxshape.ndim() + arrshape.ndim() - 1, -1);
    for (index_t i = 0; i < idxshape.ndim(); ++i) {
      oshape[i + actual_axis] = idxshape[i];
    }
    for (index_t i = 0; i < arrshape.ndim(); i++) {
      if (i < actual_axis) {
        oshape[i] = arrshape[i];
      } else if (i > actual_axis) {
        oshape[i + idxshape.ndim() - 1] = arrshape[i];
      }
    }
    out_attrs->push_back(oshape);
    return shape_is_known(oshape);
  } else {
    mxnet::TShape oshape = idxshape;
    out_attrs->push_back(oshape);
    return shape_is_known(oshape);
  }
}

template<typename xpu>
void NumpyTakeOpForward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs);

template<typename xpu>
void NumpyTakeOpBackward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_NE(req[take_::kIdx], kAddTo)
    << "take layer doesn't support gradient of req type kAddTo to index";

  const NumpyTakeParam& param = nnvm::get<NumpyTakeParam>(attrs.parsed);

  // grad_out is the gradient of the outputs in the feed-forward
  // grad_in is the gradient of the inputs in the feed-forward
  Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {  // output data type
    MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType, {  // index data type
      // inputs are specified in the .cc file, which are the gradients from
      // the upper layer and the input index
      // outputs are the gradients of inputs in the feed-forward pass
      const mxnet::TShape& idxshape = inputs[1].shape_;
      const mxnet::TShape& arrshape = outputs[0].shape_;
      const mxnet::TShape& oshape = inputs[0].shape_;

      if (req[take_::kIdx] != kNullOp) {
        mxnet_op::Kernel<mxnet_op::set_zero, xpu>::Launch(
          s, idxshape.Size(), outputs[take_::kIdx].dptr<IType>());
      }

      if (!param.axis.has_value() || (param.axis.has_value() && param.axis.value() == 0)) {
        int idxndim = idxshape.ndim();
        Tensor<xpu, 1, IType> idx = inputs[1].get_with_shape<xpu, 1, IType>(
            Shape1(idxshape.ProdShape(0, idxndim)), s);
        Tensor<xpu, 2, DType> grad_in;
        if (!param.axis.has_value()) {
          grad_in = outputs[0].get_with_shape<xpu, 2, DType>(Shape2(arrshape.Size(), 1), s);
        } else {
          grad_in = outputs[0].get_with_shape<xpu, 2, DType>(Shape2(arrshape[0], arrshape.ProdShape(1, arrshape.ndim())), s);
        }
        Tensor<xpu, 2, DType> grad_out = inputs[0].get_with_shape<xpu, 2, DType>(
            Shape2(oshape.ProdShape(0, idxndim), oshape.ProdShape(idxndim, oshape.ndim())), s);

        if (req[take_::kArr] == kWriteTo || req[take_::kArr] == kAddTo) {
          if (req[take_::kArr] == kWriteTo) {
            grad_in = scalar<DType>(0.0f);
          }
          if (param.mode == take_::kClip) {
            TakeGradZeroDim<IType, DType, true>(grad_in, idx, grad_out);
          } else {
            TakeGradZeroDim<IType, DType, false>(grad_in, idx, grad_out);
          }
        } else {
          LOG(FATAL) << "wrong req";
        }
      } else {
        const int actual_axis = param.axis.value() + ((param.axis.value() < 0) ? arrshape.ndim() : 0);

        const TBlob& idx = inputs[1];
        const TBlob& arr = outputs[0];
        const TBlob& ograd = inputs[0];

        if (param.mode == take_::kClip) {
          TakeOpBackwardImpl<true>(s, ctx, arr, idx, ograd, actual_axis);
        } else {
          TakeOpBackwardImpl<false>(s, ctx, arr, idx, ograd, actual_axis);
        }
      }
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_INDEXING_OP_H_
