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

#ifndef MXNET_OPERATOR_TENSOR_SLICE_SUM_INL_H_
#define MXNET_OPERATOR_TENSOR_SLICE_SUM_INL_H_

#include <vector>
#include <algorithm>
#include <utility>
#include "../mxnet_op.h"
#include "./broadcast_reduce_op.h"
#include "./init_op.h"
#include "../../common/static_array.h"

namespace mxnet {
namespace op {

struct SliceSumParam : public dmlc::Parameter<SliceSumParam> {
  int begin, end, axis;
  DMLC_DECLARE_PARAMETER(SliceSumParam) {
    DMLC_DECLARE_FIELD(begin)
    .describe("starting indices for the slice operation, supports negative indices.");
    DMLC_DECLARE_FIELD(end)
    .describe("ending indices for the slice operation, supports negative indices.");
    DMLC_DECLARE_FIELD(axis)
    .describe("axis for the slice operation, supports negative values.");
  }
  bool operator==(const SliceSumParam& other) const {
    return this->begin == other.begin &&
           this->end == other.end &&
           this->axis == other.axis;
  }
};

inline bool SliceSumOpShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector* in_attrs,
                         mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& d1_shape = (*in_attrs)[0];
  const mxnet::TShape& d2_shape = (*in_attrs)[1];
  mxnet::TShape oshape = d1_shape;

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return !shape_is_none (d1_shape) && !shape_is_none(d2_shape);
}

template<int ndim, int req>
struct SliceSumKernel;

template<int ndim, int req>
struct SliceSumKernel {
  template<typename DType>
   MSHADOW_XINLINE static void Map(int in1_idx, DType* out_data, const DType* in1_data,
                                  const DType* in2_data,
                                  const common::StaticArray<int, ndim> in1_strides,
                                  const common::StaticArray<int, ndim> in2_strides,
                                  const common::StaticArray<int, ndim> in2_offsets) {
    int in2_idx = 0;
    int idx = in1_idx;
    #pragma unroll
    for (int dim = 0; dim < ndim-1; dim++) {
        int stride = in1_strides[dim];
        int in1_offset = idx / stride;
        in2_idx += (in1_offset + in2_offsets[dim]) * in2_strides[dim];
        idx = idx % stride;
    }
    in2_idx += idx + in2_offsets[ndim-1];
    DType sum = in1_data[in1_idx] + in2_data[in2_idx];
    KERNEL_ASSIGN(out_data[in1_idx], req, sum);
  }
};



template<typename xpu>
void SliceSumImpl(const nnvm::NodeAttrs& attrs,
                      mshadow::Stream<xpu>* s,
                      const TBlob& in1_data,
                      const TBlob& in2_data,
                      const OpReqType req,
                      const TBlob& out_data) {
  const SliceSumParam& param = nnvm::get<SliceSumParam>(attrs.parsed);
  const int ndim_ = in1_data.ndim();
  const int axis = param.axis;
  int64_t begin = param.begin;
  int64_t end = param.end;
  if (end == -1) {
    end = begin + in1_data.shape_[axis];
  }

  using namespace mxnet_op;
  int64_t num_elements = out_data.Size();
  CHECK_EQ (req, kWriteTo);

  CHECK_LT (axis, ndim_);
  CHECK_GE (begin, 0);
  CHECK_GE (end, begin);
  CHECK_LE (end, in2_data.shape_[axis]);
  CHECK_EQ (end-begin, in1_data.shape_[axis]);

  CHECK_EQ (in2_data.ndim(), ndim_);
  for (int i = 0; i < ndim_; i++) {
    if (i != axis) {
      CHECK_EQ (in2_data.shape_[i], in1_data.shape_[i]);
    } 
  }

  MXNET_NDIM_SWITCH(ndim_, ndim, {
      common::StaticArray<int, ndim> in1_strides;
      common::StaticArray<int, ndim> in2_offsets;
      common::StaticArray<int, ndim> in2_strides;
      in1_strides[ndim-1] = 1;
      in2_strides[ndim-1] = 1;
      if (ndim > 1) {
        in1_strides[ndim-2] = in1_data.shape_[ndim-1];
        in2_strides[ndim-2] = in2_data.shape_[ndim-1];
      }

      if (out_data.type_flag_ == mshadow::kFloat16) {
        CHECK_EQ (num_elements % 2, 0);
        CHECK_EQ (in1_data.shape_[ndim-1] % 2, 0);
        CHECK_EQ (in2_data.shape_[ndim-1] % 2,  0);
        num_elements /= 2;
        if (axis == ndim - 1) {
            if (begin >= 0) {
                CHECK_EQ (begin % 2, 0);
                begin /= 2;
            }
        }
        if (ndim > 1) {
          in1_strides[ndim-2] /= 2;
          in2_strides[ndim-2] /= 2;
        }
      } 

      for (int j = ndim-3; j >= 0; j--) {
        in1_strides[j] = in1_strides[j+1] * in1_data.shape_[j+1];
        in2_strides[j] = in2_strides[j+1] * in2_data.shape_[j+1];
      }
      for (int j = ndim-1; j >= 0; j--) {
        if (j == axis) {
            in2_offsets[j] = begin;
        } else {
            in2_offsets[j] = 0;
        }
      }

      MSHADOW_TYPE_SWITCH_WITH_HALF2(out_data.type_flag_, DType, {
        MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
          Kernel<SliceSumKernel<ndim, req_type>, xpu>::Launch(s, num_elements,
              out_data.dptr<DType>(), in1_data.dptr<DType>(), in2_data.dptr<DType>(), 
              in1_strides, in2_strides, in2_offsets);
        })
      })
  })
}

template<typename xpu>
void SliceSumOpForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
  SliceSumImpl(attrs, s, inputs[0], inputs[1], req[0], outputs[0]);
}

template<typename xpu>
void SliceSumOpBackward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req.size(), 2U);
  mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_SLICE_SUM_INL_H_
