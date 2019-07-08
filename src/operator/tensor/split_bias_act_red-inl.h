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

#ifndef MXNET_OPERATOR_TENSOR_SPLIT_BIAS_ACT_RED_INL_H_
#define MXNET_OPERATOR_TENSOR_SPLIT_BIAS_ACT_RED_INL_H_

#include <vector>
#include <algorithm>
#include <utility>
#include "../mxnet_op.h"
#include "../rnn_impl.h"
#include "./broadcast_reduce_op.h"
#include "./init_op.h"
#include "../../common/static_array.h"

typedef mshadow::half::half2_t _half2;
typedef mshadow::half::half_t _half;

#define TYPE_SWITCH_FLOAT_HALF2(type, DType, ...)  \
  switch (type) {                                      \
  case mshadow::kFloat32:                              \
    {                                                  \
      typedef float DType;                             \
      {__VA_ARGS__}                                    \
    }                                                  \
    break;                                             \
  case mshadow::kFloat16:                              \
    {                                                  \
      typedef _half2 DType;                            \
      {__VA_ARGS__}                                    \
    }                                                  \
    break;                                             \
  default:                                             \
    LOG(FATAL) << "This operation only supports "      \
                  "32-bit and 64-bit floating point";  \
  }



namespace mxnet {
namespace op {

struct SplitBiasActRedParam : public dmlc::Parameter<SplitBiasActRedParam> {
  int axis;
  DMLC_DECLARE_PARAMETER(SplitBiasActRedParam) {
    DMLC_DECLARE_FIELD(axis)
    .describe("axis for the slice operation, supports negative values.");
  }
  bool operator==(const SplitBiasActRedParam& other) const {
    return this->axis == other.axis;
  }
};


inline bool SplitBiasActRedShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector* in_attrs,
                         mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& d1_shape = (*in_attrs)[0];
  const mxnet::TShape& d2_shape = (*in_attrs)[1];
  mxnet::TShape oshape = d1_shape;

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return !shape_is_none(oshape) && !shape_is_none(d2_shape);
}

template<typename xpu, int ndim, int req>
struct SplitBiasActRedKernel;

template<typename xpu, typename DType> MSHADOW_XINLINE DType _tanh(DType inp);

template<typename xpu, typename DType> MSHADOW_XINLINE DType _sigmoid(DType inp);


#if defined(__CUDACC__)
template<> MSHADOW_XINLINE float _tanh<gpu, float>(float inp) {
    return tanhf(inp);
}

template<> MSHADOW_XINLINE float _sigmoid<gpu, float>(float inp) {
    return 1.f/(1.f + expf(-inp));
}
#if MSHADOW_CUDA_HALF2
template<> MSHADOW_XINLINE _half2 _tanh<gpu, _half2>(_half2 inp) {
    float inp1 = __low2float(inp.half2_);
    float inp2 = __high2float(inp.half2_);
    float out1 = _tanh<gpu, float>(inp1);
    float out2 = _tanh<gpu, float>(inp2);
    return _half2(__floats2half2_rn(out1, out2));
}

template<> MSHADOW_XINLINE _half2 _sigmoid<gpu, _half2>(_half2 inp) {
    float inp1 = __low2float(inp.half2_);
    float inp2 = __high2float(inp.half2_);
    float out1 = _sigmoid<gpu, float>(inp1);
    float out2 = _sigmoid<gpu, float>(inp2);
    return _half2(__floats2half2_rn(out1, out2));
}

#else
template<> MSHADOW_XINLINE _half2 _tanh<gpu, _half2>(_half2 inp) {
    float inp1 = inp.half_t2[0];
    float inp2 = inp.half_t2[1];
    float out1 = _tanh<gpu, float>(inp1);
    float out2 = _tanh<gpu, float>(inp2);
    return _half2(out1, out2);
}

template<> MSHADOW_XINLINE _half2 _sigmoid<gpu, _half2>(_half2 inp) {
    float inp1 = inp.half_t2[0];
    float inp2 = inp.half_t2[1];
    float out1 = _sigmoid<gpu, float>(inp1);
    float out2 = _sigmoid<gpu, float>(inp2);
    return _half2(out1, out2);
}
#endif

#else
template<> MSHADOW_XINLINE float _tanh<cpu, float>(float inp) {
    return math::tanh(inp);
}

template<> MSHADOW_XINLINE float _sigmoid<cpu, float>(float inp) {
    return 1.f/(1.f + math::exp(-inp));
}

template<> MSHADOW_XINLINE _half2 _tanh<cpu, _half2>(_half2 inp) {
    float inp1 = inp.half_t2[0];
    float inp2 = inp.half_t2[1];
    float out1 = _tanh<cpu, float>(inp1);
    float out2 = _tanh<cpu, float>(inp2);
    return _half2(out1, out2);
}

template<> MSHADOW_XINLINE _half2 _sigmoid<cpu, _half2>(_half2 inp) {
    float inp1 = inp.half_t2[0];
    float inp2 = inp.half_t2[1];
    float out1 = _sigmoid<cpu, float>(inp1);
    float out2 = _sigmoid<cpu, float>(inp2);
    return _half2(out1, out2);
}
#endif

template<typename xpu, int ndim, int req>
struct SplitBiasActRedKernel {
  template<typename DType>
   MSHADOW_XINLINE static void Map(int in1_idx, DType* out_data, const DType* in1_data,
                                  const DType* in2_data,
                                  const common::StaticArray<int, ndim> in1_strides,
                                  const common::StaticArray<int, ndim> in2_strides,
                                  int in2_offset) {
    int in2_idx = 0;
    int idx = in1_idx;
    #pragma unroll
    for (int dim = 0; dim < ndim-1; dim++) {
        int stride = in1_strides[dim];
        in2_idx += (idx / stride) * in2_strides[dim];
        idx = idx % stride;
    }
    in2_idx += idx;
    DType bias = in1_data[in1_idx];
    DType inp1 = in2_data[in2_idx];
    DType inp2 = in2_data[in2_idx + in2_offset];
    DType res = _tanh<xpu, DType>(inp1 + bias)*_sigmoid<xpu, DType>(inp2 + bias);
    KERNEL_ASSIGN(out_data[in1_idx], req, res);
  }
};



template<typename xpu>
void SplitBiasActRedImpl(const nnvm::NodeAttrs& attrs,
                      mshadow::Stream<xpu>* s,
                      const TBlob& in1_data,
                      const TBlob& in2_data,
                      const OpReqType req,
                      const TBlob& out_data) {
  const SplitBiasActRedParam& param = nnvm::get<SplitBiasActRedParam>(attrs.parsed);
  const int axis = param.axis;

  using namespace mxnet_op;
  int64_t num_elements = out_data.Size();
  CHECK_EQ (req, kWriteTo);

  int ndim_ = in1_data.ndim();
  int num_splits = 2;

  CHECK_LT (axis, ndim_);

  CHECK_EQ (in2_data.ndim(), ndim_);
  for (int i = 0; i < ndim_; i++) {
    if (i != axis) {
      CHECK_EQ (in2_data.shape_[i], in1_data.shape_[i]);
    } else {
      CHECK_EQ (in2_data.shape_[i], in1_data.shape_[i]*num_splits);
    }
  }

  MXNET_NDIM_SWITCH(ndim_, ndim, {
      common::StaticArray<int, ndim> in1_strides;
      common::StaticArray<int, ndim> in2_strides;
      in1_strides[ndim-1] = 1;
      in2_strides[ndim-1] = 1;
      if (ndim > 1) {
        in1_strides[ndim-2] = in1_data.shape_[ndim-1];
        in2_strides[ndim-2] = in2_data.shape_[ndim-1];
      }

      int in2_offset = in1_data.shape_[axis];
      if (out_data.type_flag_ == mshadow::kFloat16) {
        CHECK_EQ (num_elements % 2, 0);
        CHECK_EQ (in1_data.shape_[ndim-1] % 2, 0);
        CHECK_EQ (in2_data.shape_[ndim-1] % 2,  0);
        num_elements /= 2;
        if (ndim > 1) {
          in1_strides[ndim-2] /= 2;
          in2_strides[ndim-2] /= 2;
        }
        if (axis == ndim - 1) {
          in2_offset /= 2;
        }
      } 

      for (int j = ndim-3; j >= 0; j--) {
        in1_strides[j] = in1_strides[j+1] * in1_data.shape_[j+1];
        in2_strides[j] = in2_strides[j+1] * in2_data.shape_[j+1];
      }
      in2_offset *= in2_strides[axis];
      
      TYPE_SWITCH_FLOAT_HALF2(out_data.type_flag_, DType, {
        MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
          Kernel<SplitBiasActRedKernel<xpu, ndim, req_type>, xpu>::Launch(s, num_elements,
              out_data.dptr<DType>(), in1_data.dptr<DType>(), in2_data.dptr<DType>(), 
              in1_strides, in2_strides, in2_offset);
        })
      })
  })
}

template<typename xpu>
void SplitBiasActRedForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
  SplitBiasActRedImpl(attrs, s, inputs[0], inputs[1], req[0], outputs[0]);
}

template<typename xpu>
void SplitBiasActRedBackward(const nnvm::NodeAttrs& attrs,
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

#endif  // MXNET_OPERATOR_TENSOR_SPLIT_BIAS_ACT_RED_INL_H_
