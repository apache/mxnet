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
#include <string>
#include "../tensor/matrix_op-inl.h"
#include "np_broadcast_reduce_op.h"

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

struct NumpyReshapeParam : public dmlc::Parameter<NumpyReshapeParam> {
  mxnet::TShape newshape;
  std::string order;
  DMLC_DECLARE_PARAMETER(NumpyReshapeParam) {
      DMLC_DECLARE_FIELD(newshape)
          .describe("The new shape should be compatible with the original shape."
                    " If an integer, then the result will be a 1-D array of that length."
                    " One shape dimension can be -1. In this case, the value is inferred"
                    " from the length of the array and remaining dimensions.");
      DMLC_DECLARE_FIELD(order)
      .set_default("C")
      .describe("Read the elements of a using this index order, and place the elements into"
                " the reshaped array using this index order. 'C' means to read/write the elements"
                " using C-like index order, with the last axis index changing fastest, back to the"
                " first axis index changing slowest. Note that currently only C-like order is"
                " supported");
  }
};

struct NumpyXSliceParam : public dmlc::Parameter<NumpyXSliceParam> {
  mxnet::Tuple<dmlc::optional<int>> begin, end;
  mxnet::Tuple<dmlc::optional<int>> step;
  DMLC_DECLARE_PARAMETER(NumpyXSliceParam) {
    DMLC_DECLARE_FIELD(begin)
    .describe("starting indices for the slice operation, supports negative indices.");
    DMLC_DECLARE_FIELD(end)
    .describe("ending indices for the slice operation, supports negative indices.");
    DMLC_DECLARE_FIELD(step)
    .set_default(mxnet::Tuple<dmlc::optional<int>>())
    .describe("step for the slice operation, supports negative values.");
  }
  bool operator==(const NumpyXSliceParam& other) const {
    return this->begin == other.begin &&
           this->end == other.end &&
           this->step == other.step;
  }
};

struct NumpyXReshapeParam : public dmlc::Parameter<NumpyXReshapeParam> {
  mxnet::Tuple<int> newshape;
  std::string order;
  DMLC_DECLARE_PARAMETER(NumpyXReshapeParam) {
      DMLC_DECLARE_FIELD(newshape)
          .set_default(mxnet::Tuple<int>())
          .describe("The new shape should be compatible with the original shape."
                    " If an integer, then the result will be a 1-D array of that length."
                    " One shape dimension can be -1. In this case, the value is inferred"
                    " from the length of the array and remaining dimensions."
                    " -2 to -6 are used for data manipulation"
                    " -2 copy this dimension from the input to the output shape"
                    " -3 will skip current dimension if and only if the current dim size is one"
                    " -4 copy all remain of the input dimensions to the output shape"
                    " -5 use the product of two consecutive dimensions of the input"
                    " shape as the output"
                    " -6 split one dimension of the input into two dimensions passed"
                    " subsequent to -6 in the new shape");
      DMLC_DECLARE_FIELD(order)
      .set_default("C")
      .describe("Read the elements of a using this index order, and place the elements into"
                " the reshaped array using this index order. 'C' means to read/write the elements"
                " using C-like index order, with the last axis index changing fastest, back to the"
                " first axis index changing slowest. Note that currently only C-like order is"
                " supported");
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

template<int ndim>
inline void NumpyXGetIndexRange(const mxnet::TShape& dshape,
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

  if (param_step.ndim() > 0) {
    CHECK_EQ(param_step.ndim(), param_begin.ndim())
      << "step and begin must have the same length";
  }

  for (int i = 0; i < param_begin.ndim(); ++i) {
    index_t s = param_step.ndim() > 0 && param_step[i].has_value()?
                param_step[i].value() : 1;
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

inline void NumpyXSetSliceOpOutputDimSize(const index_t i, const int b,
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

template<typename xpu>
void NumpyXSliceOpForward(const nnvm::NodeAttrs& attrs,
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
  const NumpyXSliceParam& param = nnvm::get<NumpyXSliceParam>(attrs.parsed);
  MXNET_NDIM_SWITCH(data.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin, end, step;
    NumpyXGetIndexRange(data.shape_, param.begin, param.end, param.step, &begin, &end, &step);
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
void NumpyXSliceOpBackward(const nnvm::NodeAttrs& attrs,
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
  const NumpyXSliceParam& param = nnvm::get<NumpyXSliceParam>(attrs.parsed);
  if (req[0] == kWriteTo) {
    Fill(s, igrad, req[0], 0);
  } else if (req[0] == kWriteInplace) {
    LOG(FATAL) << "_slice_backward does not support kWriteInplace";
  }
  if (ograd.Size() == 0) return;
  MXNET_NDIM_SWITCH(ograd.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin, end, step;
    NumpyXGetIndexRange(igrad.shape_, param.begin, param.end, param.step, &begin, &end, &step);
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

namespace std {
template<>
struct hash<mxnet::op::NumpyXSliceParam> {
  size_t operator()(const mxnet::op::NumpyXSliceParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.begin);
    ret = dmlc::HashCombine(ret, val.end);
    ret = dmlc::HashCombine(ret, val.step);
    return ret;
  }
};
}  // namespace std

#endif  // MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_
