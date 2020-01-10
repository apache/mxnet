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
 * \file np_bicount_op-inl.h
 * \brief numpy compatible bincount operator
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_BINCOUNT_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_BINCOUNT_OP_INL_H_

#include <mxnet/operator_util.h>
#include <utility>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "np_broadcast_reduce_op.h"

namespace mxnet {
namespace op {

struct NumpyBincountParam : public dmlc::Parameter<NumpyBincountParam> {
  int minlength;
  bool has_weights;
  DMLC_DECLARE_PARAMETER(NumpyBincountParam) {
    DMLC_DECLARE_FIELD(minlength)
    .set_default(0)
    .describe("A minimum number of bins for the output array"
              "If minlength is specified, there will be at least this"
              "number of bins in the output array");
    DMLC_DECLARE_FIELD(has_weights)
    .set_default(false)
    .describe("Determine whether Bincount has weights.");
  }
};

inline bool NumpyBincountType(const nnvm::NodeAttrs& attrs,
                              std::vector<int> *in_attrs,
                              std::vector<int> *out_attrs) {
  const NumpyBincountParam& param = nnvm::get<NumpyBincountParam>(attrs.parsed);
  if (!param.has_weights) {
    return ElemwiseType<1, 1>(attrs, in_attrs, out_attrs) && in_attrs->at(0) != -1;
  } else {
    CHECK_EQ(out_attrs->size(), 1U);
    CHECK_EQ(in_attrs->size(), 2U);
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(1));
    TYPE_ASSIGN_CHECK(*in_attrs, 1, out_attrs->at(0));
    return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
  }
}

inline bool NumpyBincountStorageType(const nnvm::NodeAttrs& attrs,
                                     const int dev_mask,
                                     DispatchMode* dispatch_mode,
                                     std::vector<int> *in_attrs,
                                     std::vector<int> *out_attrs) {
  const NumpyBincountParam& param = nnvm::get<NumpyBincountParam>(attrs.parsed);
  if (param.has_weights) {
    CHECK_EQ(in_attrs->size(), 2U);
  } else {
    CHECK_EQ(in_attrs->size(), 1U);
  }
  CHECK_EQ(out_attrs->size(), 1U);
  for (int &attr : *in_attrs) {
    CHECK_EQ(attr, kDefaultStorage) << "Only default storage is supported";
  }
  for (int &attr : *out_attrs) {
    attr = kDefaultStorage;
  }
  *dispatch_mode = DispatchMode::kFComputeEx;
  return true;
}

template<typename xpu>
void NumpyBincountForwardImpl(const OpContext &ctx,
                              const NDArray &data,
                              const NDArray &weights,
                              const NDArray &out,
                              const size_t &data_n,
                              const int &minlength);

template<typename xpu>
void NumpyBincountForwardImpl(const OpContext &ctx,
                              const NDArray &data,
                              const NDArray &out,
                              const size_t &data_n,
                              const int &minlength);

template<typename xpu>
void NumpyBincountForward(const nnvm::NodeAttrs& attrs,
                          const OpContext &ctx,
                          const std::vector<NDArray> &inputs,
                          const std::vector<OpReqType> &req,
                          const std::vector<NDArray> &outputs) {
  CHECK_GE(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK(req[0] == kWriteTo);
  const NumpyBincountParam& param = nnvm::get<NumpyBincountParam>(attrs.parsed);
  const bool has_weights = param.has_weights;
  const int minlength = param.minlength;
  const NDArray &data = inputs[0];
  const NDArray &out = outputs[0];
  CHECK_LE(data.shape().ndim(), 1U) << "Input only accept 1d array";
  CHECK(!common::is_float(data.dtype())) <<"Input data should be int type";
  size_t N = data.shape().Size();
  if (N == 0) {
    mshadow::Stream<xpu> *stream = ctx.get_stream<xpu>();
    mxnet::TShape s(1, minlength);
    const_cast<NDArray &>(out).Init(s);
    MSHADOW_TYPE_SWITCH(out.dtype(), OType, {
      mxnet_op::Kernel<mxnet_op::set_zero, xpu>::Launch(
        stream, minlength, out.data().dptr<OType>());
    });
  } else {
    if (has_weights) {
      CHECK_EQ(inputs.size(), 2U);
      const NDArray &weights = inputs[1];
      CHECK_EQ(data.shape(), weights.shape()) << "weights should has same size as input";
      NumpyBincountForwardImpl<xpu>(ctx, data, weights, out, N, minlength);
    } else {
      NumpyBincountForwardImpl<xpu>(ctx, data, out, N, minlength);
    }
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_BINCOUNT_OP_INL_H_
