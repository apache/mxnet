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
 *  Copyright (c) 2019 by Contributors
 * \file np_matrix_op-inl.h
 * \brief Function definition of matrix related operators
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_

#include <vector>
#include "../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {

struct NumpyTransposeParam : public dmlc::Parameter<NumpyTransposeParam> {
  mxnet::TShape axes;
  DMLC_DECLARE_PARAMETER(NumpyTransposeParam) {
    DMLC_DECLARE_FIELD(axes).set_default(mxnet::TShape(-1, 0))
    .describe("By default, reverse the dimensions, otherwise permute "
              "the axes according to the values given.");
  }
};

template<typename xpu>
void NumpyTranspose(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  const NumpyTransposeParam& param = nnvm::get<NumpyTransposeParam>(attrs.parsed);
  CHECK_EQ(req[0], kWriteTo) << "Transpose does not support inplace";
  if (ndim_is_known(param.axes)) {
    TransposeImpl<xpu>(ctx.run_ctx, inputs[0], outputs[0], param.axes);
  } else {
    mxnet::TShape axes(inputs[0].ndim(), -1);
    for (int i = 0; i < axes.ndim(); ++i) {
      axes[i] = axes.ndim() - 1 - i;
    }
    TransposeImpl<xpu>(ctx.run_ctx, inputs[0], outputs[0], axes);
  }
}

// 
# TODO(zoeygxy): copied from Wenxu Mao's unmerged code
template<int ndim>
inline void NumpyGetIndexRange(const mxnet::TShape& dshape,
                          const mxnet::Tuple<dmlc::optional<int>>& param_begin,
                          const mxnet::Tuple<dmlc::optional<int>>& param_end,
                          const mxnet::Tuple<dmlc::optional<int>>& param_step,
                          common::StaticArray<index_t, ndim>* begin,
                          common::StaticArray<index_t, ndim>* end,
                          common::StaticArray<index_t, ndim>* step) {
  CHECK_NE(dshape.ndim(), 0U);
  CHECK_LE(param_begin.ndim(), dshape.ndim())
    << "Slicing axis exceeds data dimensions";
  CHECK_LE(param_end.ndim(), dshape.ndim())
    << "Slicing axis exceeds data dimensions";
  CHECK_EQ(param_begin.ndim(), param_end.ndim())
    << "begin and end must have the same length";
  CHECK_EQ(ndim, dshape.ndim())
    << "Static array size=" << ndim
    << " is not equal to data shape ndim=" << dshape.ndim();

  if (mxnet::ndim_is_known(param_step.ndim())) {
    CHECK_EQ(param_step.ndim(), param_begin.ndim())
      << "step and begin must have the same length";
  }

  for (int i = 0; i < param_begin.ndim(); ++i) {
    index_t s = mxnet::ndim_is_known(param_step.ndim())
                 && param_step[i].has_value() ? param_step[i].value() : 1;
    CHECK_NE(s, 0) << "slice op step[" << i << "] cannot be 0";

    index_t b = 0, e = 0;
    const index_t len = dshape[i];
    if (len > 0) {
      b = param_begin[i].has_value() ? param_begin[i].value() : (s < 0 ? len - 1 : 0);
      e = param_end[i].has_value() ? param_end[i].value() : (s < 0 ? -1 : len);

      if (b < 0) {
        b += len;
      }

      if (e < 0 && param_end[i].has_value()) {
        e += len;
      }
    }

    // move the begin and end to correct position for calculating dim size
    b = b < 0 && s > 0 ? 0 : b;
    b = b > len-1 && s < 0 ? len-1 : b;
    // if the start value lead to empty tensor under step s, use -1 for indication
    b = b < 0 || b > len-1 ? -1 : b;
    e = e > -1 ? e : -1;
    e = e > len ? len : e;
    (*begin)[i] = b;
    (*end)[i] = e;
    (*step)[i] = s;
  }

  for (index_t i = param_begin.ndim(); i < dshape.ndim(); ++i) {
    (*begin)[i] = 0;
    (*end)[i] = dshape[i];
    (*step)[i] = 1;
  }
}

inline void NumpySetSliceOpOutputDimSize(const index_t i, const int b,
                                    const int e, const int s,
                                    mxnet::TShape* oshape) {
  if (e != b && b >= 0) {
    if (s > 0) {
      (*oshape)[i] = e > b ? (e - b - 1) / s + 1 : 0;
    } else {
      (*oshape)[i] = e < b ? (b - e - 1) / (-s) + 1 : 0;
    }
  } else {
      (*oshape)[i] = 0;
  }
}

inline bool NumpySliceOpShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector* in_attrs,
                         mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& dshape = (*in_attrs)[0];
  if (!mxnet::ndim_is_known(dshape)) return false;
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  mxnet::TShape oshape = dshape;

  MXNET_NDIM_SWITCH(dshape.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin, end, step;
    NumpyGetIndexRange(dshape, param.begin, param.end, param.step, &begin, &end, &step);
    for (int i = 0; i < param.begin.ndim(); ++i) {
      const int b = begin[i], e = end[i], s = step[i];
      NumpySetSliceOpOutputDimSize(i, b, e, s, &oshape);
    }
  })

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return shape_is_known(oshape);
}

template<typename xpu>
void NumpySliceOpForward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (req[0] == kNullOp) return;
  using namespace mshadow;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  const TBlob& data = inputs[0];
  const TBlob& out = outputs[0];
  if (out.Size() == 0) {
    return;
  }
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  MXNET_NDIM_SWITCH(data.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin, end, step;
    NumpyGetIndexRange(data.shape_, param.begin, param.end, param.step, &begin, &end, &step);
    MSHADOW_TYPE_SWITCH(out.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        size_t num_threads = out.shape_.FlatTo2D()[0];
        if (std::is_same<xpu, gpu>::value) {
          num_threads *= out.shape_.get<ndim>()[ndim - 1];
        }
        mxnet_op::Kernel<slice_forward<ndim, Req, xpu>, xpu>::Launch(s, num_threads,
            out.dptr<DType>(), data.dptr<DType>(),
            data.shape_.get<ndim>(), out.shape_.get<ndim>(), begin, step);
      })
    })
  })
}

template<typename xpu>
void NumpySliceOpBackward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (req[0] == kNullOp) return;
  using namespace mshadow;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  const TBlob& ograd = inputs[0];
  const TBlob& igrad = outputs[0];
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  if (req[0] == kWriteTo) {
    Fill(s, igrad, req[0], 0);
  } else if (req[0] == kWriteInplace) {
    LOG(FATAL) << "_slice_backward does not support kWriteInplace";
  }
  if (ograd.Size() == 0) return;
  MXNET_NDIM_SWITCH(ograd.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin, end, step;
    NumpyGetIndexRange(igrad.shape_, param.begin, param.end, param.step, &begin, &end, &step);
    MSHADOW_TYPE_SWITCH(ograd.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      int num_threads = ograd.shape_.FlatTo2D()[0];
      if (std::is_same<xpu, gpu>::value) {
        num_threads *= ograd.shape_.get<ndim>()[ndim - 1];
      }
      mxnet_op::Kernel<slice_assign<ndim, Req, xpu>, xpu>::Launch(s, num_threads,
          igrad.dptr<DType>(), ograd.dptr<DType>(),
          igrad.shape_.get<ndim>(), ograd.shape_.get<ndim>(), begin, step);
      })
    })
  })
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_
