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
 * \file digitize_op.h
 * \brief Quantize operator a la numpy.digitize.
 */
#ifndef MXNET_OPERATOR_TENSOR_DIGITIZE_H_
#define MXNET_OPERATOR_TENSOR_DIGITIZE_H_

#include <mxnet/base.h>
#include <mxnet/operator_util.h>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"



namespace mxnet {
namespace op {

struct DigitizeParam : public dmlc::Parameter<DigitizeParam> {
  bool right;
  int otype;

  DMLC_DECLARE_PARAMETER(DigitizeParam) {
    DMLC_DECLARE_FIELD(right)
        .set_default(false)
        .describe("Whether the intervals include the right or the left bin edge.");
    DMLC_DECLARE_FIELD(otype)
        .add_enum("uint8", mshadow::kUint8)
        .add_enum("int32", mshadow::kInt32)
        .add_enum("int64", mshadow::kInt64)
        .add_enum("int8", mshadow::kInt8)
        .set_default(mshadow::kInt32)
        .describe("DType of the output.");
  }
};

bool InferShape(const nnvm::NodeAttrs &attrs,
                std::vector<TShape> *in_attrs,
                std::vector<TShape> *out_attrs) {
  using namespace mshadow;

  CHECK_EQ(in_attrs->size(), 2); // Size 2: data and bins
  CHECK_EQ(out_attrs->size(), 1); // Only one output tensor

  auto &data_shape = (*in_attrs)[0];
  auto &bin_shape = (*in_attrs)[1];

  // Only continue if both inputs are defined (ndim > 0), otherwise return 0
  CHECK_GT(data_shape.ndim(), 0) << "Data shape undefined";
  CHECK_GT(bin_shape.ndim(), 0) << "Bin shape undefined";

  CHECK_LE(bin_shape.ndim(), data_shape.ndim())
    << "Bins tensor cannot have a higher dimension than input data";

  // Check if the first n-1 dims of data & bins are the same
  nnvm::dim_t *bin_shape_last = (bin_shape.end() - 1);
  CHECK(std::equal(bin_shape.begin(), bin_shape_last, data_shape.begin()))
    << "First N-1 dimensions of the input data and bins tensors should be the same (N = bins.ndim)";

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, data_shape); // First arg is a shape array

  return true;
}


inline bool DigitizeOpType(const nnvm::NodeAttrs &attrs,
                           std::vector<int> *in_attrs,
                           std::vector<int> *out_attrs) {
  auto &data_type = (*in_attrs)[0];
  auto &bins_type = (*in_attrs)[1];

  CHECK_NE(data_type, -1) << "Input data type undefined";
  CHECK_NE(bins_type, -1) << "Bins type undefined";

  // Verify Have bins & data share the same type to simplify templating
  CHECK_EQ(data_type, bins_type);

  // Assign output_type the param
  const int OType = nnvm::get<DigitizeParam>(attrs.parsed).otype;
  if (OType == -1) { return false; }

  TYPE_ASSIGN_CHECK(*out_attrs, 0, OType);

  return true;
}

  // Verify Have bins & data share the same type to simplify templating
  CHECK_EQ(data_type, bins_type);

template<typename xpu>
struct ForwardKernel {
  template<typename DType, typename OType>
  static MSHADOW_XINLINE void Map(int i,
                                  DType *in_data,
                                  OType *out_data,
                                  DType *bins,
                                  size_t batch_size,
                                  size_t bins_length,
                                  bool right);
};

  TYPE_ASSIGN_CHECK(*out_attrs, 0, OType);

template<>
struct ForwardKernel<cpu> {
  template<typename DType, typename OType>
  static MSHADOW_XINLINE void Map(int i,
                                  DType *in_data,
                                  OType *out_data,
                                  DType *bins,
                                  size_t batch_size,
                                  size_t bins_length,
                                  bool right) {

    const auto data = in_data[i];
    const auto batch = i / batch_size;

    auto elem = right ? std::lower_bound(bins + bins_length * batch,
                                         bins + bins_length * (batch + 1),
                                         data)
                      : std::upper_bound(bins + bins_length * batch,
                                         bins + bins_length * (batch + 1),
                                         data);

    auto index = std::distance(bins, elem);
    out_data[i] = OType(index);
  }
};


template<typename DType>
struct CheckMonotonic {
  static MSHADOW_XINLINE void Map(int i, int bins_length, DType *bins) {
    if ((i + 1) % bins_length != 0) {
      CHECK_LT(bins[i], bins[i + 1]) << "Bins vector is not strictly monotonic and increasing";
    } // TODO: Make sure the next element in bins is actually bins[i+1]
  }

  return true;
}


inline bool DigitizeOpType(const nnvm::NodeAttrs &attrs,
                           std::vector<int> *in_attrs,
                           std::vector<int> *out_attrs) {
  auto &input_type = (*in_attrs)[0];
  auto &output_type = (*out_attrs)[0];

  TYPE_ASSIGN_CHECK(*out_attrs, 0, input_type);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, output_type);
  return out_attrs->at(0) != -1;
}


template<typename xpu, typename DType, typename BType>
struct ForwardKernel {
  static MSHADOW_XINLINE void Map(int i, DType *input_data, DType *out_data, mshadow::Tensor<cpu,
      1, BType>
  &bins, const bool right);
};


template<typename xpu>
void DigitizeOpForward(const nnvm::NodeAttrs &attrs,
                       const OpContext &ctx,
                       const std::vector<TBlob> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &outputs) {
  using namespace mshadow;

  auto s = ctx.get_stream<xpu>();
  const bool right = nnvm::get<DigitizeParam>(attrs.parsed).right;
  const auto &data = inputs[0];
  const auto &bins = inputs[1];

  MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {

    // Verify bins is strictly monotonic
    auto bins_length = bins.shape_[bins.ndim() - 1];
    mxnet_op::Kernel<CheckMonotonic<DType>, xpu>::Launch(s, bins.Size(), bins_length,
                                                         bins.dptr<DType>());

    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
        auto batch_size = data.shape_.ProdShape(bins.ndim() - 1, data.ndim());

        mxnet_op::Kernel<ForwardKernel<xpu>, xpu>::Launch(s,
        outputs[ 0 ].Size(),
        data.dptr<DType>(),
        outputs[ 0 ].dptr<OType>(),
        bins.dptr<DType>(),
        batch_size,
        bins_length,
        right);
    });
  });
}

}  // namespace op
}  // namespace mxnet


#endif  // MXNET_OPERATOR_TENSOR_DIGITIZE_H_
