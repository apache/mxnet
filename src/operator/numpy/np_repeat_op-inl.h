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
 * \file np_repeat_op-inl.h
 * \brief Function definition of the repeat op
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_REPEAT_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_REPEAT_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <string>
#include <algorithm>
#include <utility>
#include <type_traits>
#include <unordered_map>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../channel_op_common.h"
#include "../mxnet_op.h"
#include "../../common/static_array.h"

namespace mxnet {
namespace op {

struct RepeatsParam : public dmlc::Parameter<RepeatsParam> {
  dmlc::optional<mxnet::Tuple<int>> repeats;
  dmlc::optional<int> axis;
  DMLC_DECLARE_PARAMETER(RepeatsParam) {
    DMLC_DECLARE_FIELD(repeats)
      .describe("The number of repetitions for each element.");
    DMLC_DECLARE_FIELD(axis)
      .set_default(dmlc::optional<int>())
      .describe("The axis along which to repeat values."
                " The negative numbers are interpreted counting from the backward."
                " By default, use the flattened input array,"
                " and return a flat output array.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream repeats_s, axis_s;
    repeats_s << repeats;
    axis_s << axis;
    (*dict)["repeats"] = repeats_s.str();
    (*dict)["axis"] = axis_s.str();
  }
};

inline void GetRepeatsParams(const RepeatsParam& param, const mxnet::TShape& ishape,
                             int* repeats, dmlc::optional<int>* axisOpt, int* axis) {
  *repeats = 0;
  const mxnet::Tuple<int> &repts = param.repeats.value();
  for (int i=0; i < repts.ndim(); i++) {
    CHECK_GE(repts[i], 0) << "repeats cannot be a negative number";
    *repeats += repts[i];
  }
  *axisOpt = param.axis;
  if (static_cast<bool>(*axisOpt)) {
    int ndims = ishape.ndim();
    *axis = axisOpt->value();
    if (*axis < 0) {
      *axis += ndims;
    }
    CHECK(*axis >= 0 && *axis < ndims) << "axis = " << axisOpt->value() << " out of bounds";
  }
}

inline bool RepeatsOpShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector *in_attrs,
                           mxnet::ShapeVector *out_attrs) {
  const RepeatsParam& param = nnvm::get<RepeatsParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& ishape = (*in_attrs)[0];
  int repeats = 0;
  dmlc::optional<int> axisOpt;
  int axis = -1;
  GetRepeatsParams(param, ishape, &repeats, &axisOpt, &axis);
  // If 0 repeats, return an empty 1-dim, 0-size array
  if (0 == repeats) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(1, 0));
    return true;
  }

  // If repeats > 0, multiply the size of the corresponding axis by repeats
  if (static_cast<bool>(axisOpt)) {
    mxnet::TShape shape(ishape.ndim(), -1);
    for (int i = 0; i < ishape.ndim(); ++i) {
      if (i == axis) {
        shape[i] = param.repeats.value().ndim() == 1 ? repeats * ishape[i] : repeats;
      } else {
        shape[i] = ishape[i];
      }
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, shape);
  } else {  // If axis is not input by user, return a flat 1D array of size = repeats
    repeats = param.repeats.value().ndim() == 1 ? ishape.Size() * repeats : repeats;
    mxnet::TShape shape(1, repeats);
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, shape);
  }
  return shape_is_known(out_attrs->at(0));
}

struct repeat_noaxis_fwd {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t i, OType* out, IType* input,
                                  const int* indx) {
    using namespace mxnet_op;
    int ind = 0;
    while (i >= indx[ind]) ind++;
    out[i] = input[ind];
  }
};

struct repeat_axis_fwd {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t i, OType* out, IType* input,
                                  const int* indx, int stride) {
    using namespace mxnet_op;
    int ind_row = i / stride, ind_col = i % stride;
    int ind = 0;
    while (ind_row >= indx[ind]) ind++;
    out[i] = input[ind * stride + ind_col];
  }
};

template<typename xpu>
void NumpyRepeatsOpForward(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  const TBlob& iTBlob = inputs[0];
  const mxnet::TShape& ishape = iTBlob.shape_;
  if (!shape_is_known(ishape)) return;
  Stream<xpu> *s = ctx.get_stream<xpu>();

  int repeats = 0;
  dmlc::optional<int> axisOpt;
  int axis = -1;
  const RepeatsParam& param = nnvm::get<RepeatsParam>(attrs.parsed);
  GetRepeatsParams(param, ishape, &repeats, &axisOpt, &axis);
  if (0 == repeats) return;
  mxnet::Tuple<int> repts = param.repeats.value();
  if (repts.ndim() == 1) {
    int len = static_cast<bool>(axisOpt) ? ishape[axis] : ishape.Size();
    std::vector<int> temp(len, repeats);
    repts = mxnet::Tuple<int>(temp);
  }
  for (int i=1; i < repts.ndim(); i++) {
    repts[i] += repts[i-1];
  }
  size_t total_temp_size = repts.ndim() * sizeof(int);
  Tensor<xpu, 1, char> temp_space =
    ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(total_temp_size), s);
  int* ind = reinterpret_cast<int*>(temp_space.dptr_);

  if (ctx.run_ctx.ctx.dev_mask() == gpu::kDevMask) {
  #if MXNET_USE_CUDA
    cudaMemcpyAsync(ind, repts.begin(), repts.ndim() * sizeof(int),
                    cudaMemcpyHostToDevice, Stream<gpu>::GetStream(ctx.get_stream<gpu>()));
  #else
    LOG(FATAL) << "Illegal attempt to use GPU in a CPU-only build";
  #endif
  } else {
    std::memcpy(ind, repts.begin(), repts.ndim() * sizeof(int));
  }

  if (!param.axis.has_value()) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    const TBlob& in_data = inputs[0];
    const TBlob& out_data = outputs[0];
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
          mxnet_op::Kernel<repeat_noaxis_fwd, xpu>::Launch(
            s, out_data.Size(), out_data.dptr<OType>(),
            in_data.dptr<IType>(), ind);
        });
    });
  } else {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    const TBlob& in_data = inputs[0];
    const TBlob& out_data = outputs[0];
    int stride = 1;
    for (int i = 1; i < ishape.ndim(); i++) {
      stride *= ishape[i];
    }

    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
          mxnet_op::Kernel<repeat_axis_fwd, xpu>::Launch(
            s, out_data.Size(), out_data.dptr<OType>(),
            in_data.dptr<IType>(), ind, stride);
        });
    });
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_REPEAT_OP_INL_H_
