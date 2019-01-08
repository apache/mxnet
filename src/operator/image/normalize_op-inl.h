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
 * \file normalize_op-inl.h
 * \brief Image normalization operator
*/
#ifndef MXNET_OPERATOR_IMAGE_NORMALIZE_OP_INL_H_
#define MXNET_OPERATOR_IMAGE_NORMALIZE_OP_INL_H_


#include <mxnet/base.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <utility>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {
namespace image {

// Parameter registration for image Normalize operator
struct NormalizeParam : public dmlc::Parameter<NormalizeParam> {
  nnvm::Tuple<float> mean;
  nnvm::Tuple<float> std;
  DMLC_DECLARE_PARAMETER(NormalizeParam) {
    DMLC_DECLARE_FIELD(mean)
    .describe("Sequence of means for each channel.");
    DMLC_DECLARE_FIELD(std)
    .describe("Sequence of standard deviations for each channel.");
  }
};

// Shape and Type inference for image Normalize operator

// Shape inference
inline bool NormalizeOpShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  const NormalizeParam &param = nnvm::get<NormalizeParam>(attrs.parsed);

  const auto& dshape = (*in_attrs)[0];
  if (!dshape.ndim()) return false;

  CHECK((dshape.ndim() == 3) || (dshape.ndim() == 4))
      << "Input tensor must have shape (channels, height, width), or "
      << "(N, channels, height, width), but got " << dshape;

  int32_t nchannels;
  if (dshape.ndim() == 3) {
    nchannels = dshape[0];
    CHECK(nchannels == 3 || nchannels == 1)
      << "The first dimension of input tensor must be the channel dimension with "
      << "either 1 or 3 elements, but got input with shape " << dshape;
  } else if (dshape.ndim() == 4) {
    nchannels = dshape[1];
    CHECK(nchannels == 3 || nchannels == 1)
      << "The second dimension of input tensor must be the channel dimension with "
      << "either 1 or 3 elements, but got input with shape " << dshape;
  }

  CHECK((param.mean.ndim() == 1) || (param.mean.ndim() == nchannels))
      << "Invalid mean for input with shape " << dshape
      << ". mean must have either 1 or " << nchannels
      << " elements, but got " << param.mean;
  CHECK(param.std.ndim() == 1 || param.std.ndim() == nchannels)
      << "Invalid std for input with shape " << dshape
      << ". std must have either 1 or " << nchannels
      << " elements, but got " << param.std;

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape);
  return true;
}

// Type Inference
inline bool NormalizeOpType(const nnvm::NodeAttrs& attrs,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  // Normalized Tensor will be a float
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat32);
  return out_attrs->at(0) != -1;
}

template<int req>
struct normalize_forward {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int j, DType* out_data, const DType* in_data,
                                    const int i, const int length, const int step,
                                    const DType mean, const DType std_dev) {
        KERNEL_ASSIGN(out_data[step + i*length + j], req,
                      (in_data[step + i*length + j] - mean) / std_dev);
    }
};

template<typename xpu>
void NormalizeImpl(const OpContext &ctx,
                          const std::vector<TBlob> &inputs,
                          const std::vector<TBlob> &outputs,
                          const std::vector<OpReqType> &req,
                          const NormalizeParam &param,
                          const int length,
                          const int channel,
                          const int step = 0) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        DType* input = inputs[0].dptr<DType>();
        DType* output = outputs[0].dptr<DType>();

        for (int i = 0; i < channel; ++i) {
            DType mean = param.mean[param.mean.ndim() > 1 ? i : 0];
            DType std_dev = param.std[param.std.ndim() > 1 ? i : 0];
            mxnet_op::Kernel<normalize_forward<req_type>, xpu>::Launch(
                s, length, output, input,
                i, length, step, mean, std_dev);
        }
      });
    });
}

template<typename xpu>
void NormalizeOpForward(const nnvm::NodeAttrs &attrs,
                      const OpContext &ctx,
                      const std::vector<TBlob> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  const NormalizeParam &param = nnvm::get<NormalizeParam>(attrs.parsed);

  // 3D input (c, h, w)
  if (inputs[0].ndim() == 3) {
    const int length = inputs[0].shape_[1] * inputs[0].shape_[2];
    const int channel = inputs[0].shape_[0];
    NormalizeImpl<xpu>(ctx, inputs, outputs, req, param, length, channel);
  } else if (inputs[0].ndim() == 4) {
    // 4D input (n, c, h, w)
    const int batch_size = inputs[0].shape_[0];
    const int length = inputs[0].shape_[2] * inputs[0].shape_[3];
    const int channel = inputs[0].shape_[1];
    const int step = channel * length;

    #pragma omp parallel for
    for (auto n = 0; n < batch_size; ++n) {
      NormalizeImpl<xpu>(ctx, inputs, outputs, req, param, length, channel, n*step);
    }
  }
}

}  // namespace image
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_IMAGE_NORMALIZE_OP_INL_H_
