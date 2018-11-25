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
 * \file corner_pooling-inl.h
 * \brief corner pooling operator
 * \author Jiajie Tang
*/

#ifndef MXNET_OPERATOR_CONTRIB_CORNER_POOLING_INL_H_
#define MXNET_OPERATOR_CONTRIB_CORNER_POOLING_INL_H_

#include <vector>

#include <dmlc/parameter.h>
#include "../operator_common.h"

namespace mxnet {
namespace op {

namespace corner_pool_enum {
enum PoolingOpInputs {kData};
enum PoolingOpOutputs {kOut};
enum PoolingOpType {kTopPooling, kBottomPooling, kLeftPooling, kRightPooling};
enum PoolingOpPadConventionType {kValid, kFull, kSame};
}

struct CornerPoolingParam : public dmlc::Parameter<CornerPoolingParam> {
  int corner_pooling_type;
  DMLC_DECLARE_PARAMETER(CornerPoolingParam) {
    DMLC_DECLARE_FIELD(corner_pooling_type)
    .add_enum("left", corner_pool_enum::kLeftPooling)
    .add_enum("right", corner_pool_enum::kRightPooling)
    .add_enum("top", corner_pool_enum::kTopPooling)
    .add_enum("bottom", corner_pool_enum::kBottomPooling)
    .describe("CornerPooling type to be applied(left, right, top or bottom).");
  }
};

inline int GetNumOutputs(const CornerPoolingParam &param) {
  return 1;
}

inline int GetNumBackInputs(const CornerPoolingParam &param) {
  return 3;
}


#if MXNET_USE_CUDA
template<typename DType>
inline void corner_pool(mshadow::Stream<gpu> *s,
                        const DType *in_data, const TShape &ishape,
                        const int corner_pooling_type,
                        OpReqType req_type,
                        DType *out_data);

template<typename DType>
inline void corner_pool_grad(mshadow::Stream<gpu> *s,
                             const DType *out_grad,
                             const DType *in_data,
                             const DType *out_data,
                             const TShape &ishape,
                             const int corner_pooling_type,
                             OpReqType req_type,
                             DType *in_grad);
#endif  // MXNET_USE_CUDA

template<typename DType>
inline void corner_pool(mshadow::Stream<cpu> *s,
                        const DType *in_data,
                        const TShape &ishape,
                        const int corner_pooling_type,
                        OpReqType req_type,
                        DType *out_data) {
  using mshadow::red::limits::MinValue;
// const TShape &oshape = ishape;
  CHECK_EQ(req_type, kWriteTo)
      << "Only support req=kWriteTo in corner pooling operations";
  int height = ishape[2], width = ishape[3];
  if (corner_pooling_type == 0 || corner_pooling_type == 1) {
    // top or bottom
    int h_end = 0, h_start = 0, h_step = 0;
    if (corner_pooling_type == 0) {
      h_step = -1;
      h_start = height - 1;
      h_end = -1;
    } else {
      h_step = +1;
      h_start = 0;
      h_end = height;
    }
    const index_t data_offset = width * height;
    for (index_t b{0}; b < ishape[0]; ++b)
      for (index_t c{0}; c < ishape[1]; ++c) {
        for (index_t w{0}; w < width; ++w) {
          DType max_val = MinValue<DType>();
          for (int h{h_start}; h != h_end; h += h_step) {
            const int index = h*width + w;
            max_val = max_val > in_data[index] ? max_val : in_data[index];
            out_data[index] = max_val;
          }
        }
        in_data += data_offset;
        out_data += data_offset;
      }
  } else if (corner_pooling_type == 2 || corner_pooling_type == 3) {
    // left or right
    int w_end = 0, w_start = 0, w_step = 0;
    if (corner_pooling_type == 2) {
      w_step = -1;
      w_start = width - 1;
      w_end = -1;
    } else {
      w_step = +1;
      w_start = 0;
      w_end = width;
    }
    const index_t data_offset = width * height;
    for (index_t b{0}; b < ishape[0]; ++b)
      for (index_t c{0}; c < ishape[1]; ++c) {
        for (index_t h{0}; h < height; ++h) {
          DType max_val = MinValue<DType>();
          for (int w{w_start}; w != w_end; w += w_step) {
            const int index = h*width + w;
            max_val = max_val > in_data[index] ? max_val : in_data[index];
            out_data[index] = max_val;
          }
        }
        in_data += data_offset;
        out_data += data_offset;
      }
  } else {
    LOG(FATAL) << "Unsupported corner pooling type";
  }
}

template<typename DType>
inline void corner_pool_grad(mshadow::Stream<cpu> *s,
                             const DType *out_grad,
                             const DType *in_data,
                             const DType *out_data,
                             const TShape &ishape,
                             const int corner_pooling_type,
                             OpReqType req_type,
                             DType *in_grad) {
  const int height = ishape[2], width = ishape[3];
  const index_t data_offset = width * height;
  if (corner_pooling_type == 0 || corner_pooling_type == 1) {
    // top or bottom
    int h_end = 0, h_start = 0, h_step = 0;
    if (corner_pooling_type == 0) {
      h_step = -1;
      h_start = height - 1;
      h_end = -1;
    } else {
      h_step = +1;
      h_start = 0;
      h_end = height;
    }
    for (index_t b{0}; b < ishape[0]; ++b)
      for (index_t c{0}; c < ishape[1]; ++c) {
        for (index_t w{0}; w < width; ++w) {
          int max_h_idx = h_start;
          for (int h{h_start}; h != h_end; h += h_step) {
            const int index = h*width + w;
            if (out_data[index] != out_data[max_h_idx]) {
              max_h_idx = index;
            }
            in_grad[max_h_idx] += out_grad[index];
          }
        }
        out_data += data_offset;
        in_grad += data_offset;
        out_grad += data_offset;
      }
  } else if (corner_pooling_type == 2 || corner_pooling_type == 3) {
    // left or right
    int w_end = 0, w_start = 0, w_step = 0;
    if (corner_pooling_type == 2) {
      w_step = -1;
      w_start = width - 1;
      w_end = -1;
    } else {
      w_step = +1;
      w_start = 0;
      w_end = width;
    }
    const index_t data_offset = width * height;
    for (index_t b{0}; b < ishape[0]; ++b)
      for (index_t c{0}; c < ishape[1]; ++c) {
        for (index_t h{0}; h < height; ++h) {
          int max_w_idx = w_start;
          for (int w{w_start}; w != w_end; w += w_step) {
            const int index = h*width + w;
            if (out_data[index] != out_data[max_w_idx]) {
              max_w_idx = index;
            }
            in_grad[max_w_idx] += out_grad[index];
          }
        }
        in_grad += data_offset;
        out_grad += data_offset;
        out_data += data_offset;
      }
  }
}

template<typename xpu, typename DType>
class CornerPoolingOp {
 public:
  void Init(CornerPoolingParam p) {
    this->param_ = p;
  }

  void Forward(const OpContext &ctx, const TBlob &in_data,
               const OpReqType &req, const TBlob &out_data) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    corner_pool<DType>(s, in_data.dptr<DType>(), in_data.shape_,
                       param_.corner_pooling_type, req, out_data.dptr<DType>());
  }

  void Backward(const OpContext &ctx, const TBlob &out_grad,
                const TBlob &in_data, const TBlob &out_data,
                const OpReqType &req, const TBlob &in_grad) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    corner_pool_grad<DType>(s, out_grad.dptr<DType>(), in_data.dptr<DType>(),
            out_data.dptr<DType>(), out_grad.shape_,
            param_.corner_pooling_type, req, in_grad.dptr<DType>());
  }

 private:
  CornerPoolingParam param_;
};  // class PoolingOp

template<typename xpu>
void CornerPoolingCompute(const nnvm::NodeAttrs &attrs,
                          const OpContext &ctx,
                          const std::vector<TBlob> &inputs,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &outputs) {
  const CornerPoolingParam &param = nnvm::get<CornerPoolingParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), GetNumOutputs(param));

  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    if (corner_pool_enum::kLeftPooling == param.corner_pooling_type
        || corner_pool_enum::kRightPooling == param.corner_pooling_type
        || corner_pool_enum::kTopPooling == param.corner_pooling_type
        || corner_pool_enum::kBottomPooling == param.corner_pooling_type) {
      CornerPoolingOp<xpu, DType> op;
      op.Init(param);
      op.Forward(ctx, inputs[0], req[0], outputs[0]);
    } else {
      LOG(FATAL) << "unknown corner pooling type";
    }
  });
}

template<typename xpu>
void CornerPoolingGradCompute(const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<TBlob> &outputs) {
  const CornerPoolingParam &param = nnvm::get<CornerPoolingParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), GetNumBackInputs(param));
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  off_t ograd_idx, in_data_idx, out_data_idx;

  ograd_idx = 0;
  in_data_idx = 1;
  out_data_idx = 2;
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    if (corner_pool_enum::kLeftPooling == param.corner_pooling_type
        || corner_pool_enum::kRightPooling == param.corner_pooling_type
        || corner_pool_enum::kTopPooling == param.corner_pooling_type
        || corner_pool_enum::kBottomPooling == param.corner_pooling_type) {
      CornerPoolingOp<xpu, DType> op;
      op.Init(param);
      op.Backward(ctx, inputs[ograd_idx], inputs[in_data_idx],
                  inputs[out_data_idx], req[0], outputs[0]);
    } else {
      LOG(FATAL) << "unknown corner pooling type";
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_CORNER_POOLING_INL_H_
