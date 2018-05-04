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
 * Copyright (c) 2017 by Contributors
 * \file multisample_op.h
 * \brief Function definitions of operators for sampling from multiple distributions
 */
#ifndef MXNET_OPERATOR_RANDOM_MULTISAMPLE_OP_H_
#define MXNET_OPERATOR_RANDOM_MULTISAMPLE_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "./sampler.h"


namespace mxnet {
namespace op {

struct MultiSampleParam : public dmlc::Parameter<MultiSampleParam> {
  TShape shape;
  int dtype;
  DMLC_DECLARE_PARAMETER(MultiSampleParam) {
    DMLC_DECLARE_FIELD(shape)
      .set_default(TShape())
      .describe("Shape to be sampled from each random distribution.");
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("None", -1)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .set_default(-1)
    .describe("DType of the output in case this can't be inferred. "
              "Defaults to float32 if not defined (dtype=None).");
  }
};

inline bool MultiSampleOpShape(const nnvm::NodeAttrs& attrs,
                               std::vector<TShape>* in_attrs,
                               std::vector<TShape>* out_attrs) {
  CHECK_GT(in_attrs->size(), 0)
    << "sampling operator takes 1 or 2 arguments (" << in_attrs->size() << " given)";
  CHECK_LT(in_attrs->size(), 3)
    << "sampling operator takes 1 or 2 arguments (" << in_attrs->size() << " given)";
  CHECK_EQ(out_attrs->size(), 1);
  // Get shape to be sampled for each parameter set.
  const MultiSampleParam& param = nnvm::get<MultiSampleParam>(attrs.parsed);
  TShape sshape = param.shape;
  for (size_t i = 0; i < sshape.ndim(); ++i) {
    CHECK_GT(sshape[i], 0) << "shape parameter must be non-zero within each dimension";
  }
  // Examine output shape whether it is already defined.
  TShape tshape((*out_attrs)[0]);
  // The illegal case of tshape.ndim() <= sshape.ndim() will
  // automatically crash when we back-propagate from inputs to outputs.
  if (tshape.ndim() > sshape.ndim()) {
    // Promote down by removing last dimensions which represent the samples.
    tshape = TShape(tshape.begin(), tshape.begin()+(tshape.ndim()-sshape.ndim()));
  }
  // Shape assignemnt/checking for inputs.
  for (size_t i = 0; i < in_attrs->size(); ++i) {
    if ( !shape_assign(&tshape, (*in_attrs)[i])) return false;
  }
  for (size_t i = 0; i < in_attrs->size(); ++i) {
    SHAPE_ASSIGN_CHECK(*in_attrs, i, tshape);
  }
  if (tshape.ndim() > 0) {
    // Shape assignment/check for propagation from inputs to output.
    std::vector<int> cshape(tshape.begin(), tshape.end());
    cshape.insert(cshape.end(), sshape.begin(), sshape.end());
    TShape oshape(cshape.begin(), cshape.end());
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  }
  return true;
}

inline bool MultiSampleOpType(const nnvm::NodeAttrs& attrs,
                              std::vector<int>* in_attrs,
                              std::vector<int>* out_attrs) {
  CHECK_GT(in_attrs->size(), 0)
    << "sampling operator takes 1 or 2 arguments (" << in_attrs->size() << " given)";
  CHECK_LT(in_attrs->size(), 3)
    << "sampling operator takes 1 or 2 arguments (" << in_attrs->size() << " given)";
  CHECK_EQ(out_attrs->size(), 1);

  // All inputs must have same type.
  int dtype = -1;
  for (size_t i = 0; i < in_attrs->size(); ++i) {
    if (!type_assign(&dtype, (*in_attrs)[i])) return false;
  }
  for (size_t i = 0; i < in_attrs->size(); ++i) {
    TYPE_ASSIGN_CHECK(*in_attrs, i, dtype);
  }
  if (-1 == dtype) return false;

  // The output may have a different type so we can't infer from inputs.
  const MultiSampleParam& param = nnvm::get<MultiSampleParam>(attrs.parsed);
  dtype = (*out_attrs)[0];
  if (dtype != -1) {
    if (param.dtype != -1) {
      // dtype given in args, check that it matches the output type
      CHECK_EQ(dtype, param.dtype) << "Inferred output type does not match requested type: "
      << dtype << " vs " << param.dtype;
    }
  } else {
    // Output type can't be inferred. Use type in args or default.
    dtype = (param.dtype == -1 ? mshadow::kFloat32 : param.dtype);
  }
  bool dtype_ok = (dtype == mshadow::kFloat16) || (dtype == mshadow::kFloat32) ||
    (dtype == mshadow::kFloat64);
  CHECK_EQ(dtype_ok, true) << "Output type must be float16, float32, or float64: dtype is "
    << dtype<< " vs " << mshadow::kFloat16 << " or " << mshadow::kFloat32 << " or "
    << mshadow::kFloat64;
  TYPE_ASSIGN_CHECK(*out_attrs, 0, dtype);
  return true;
}

using namespace mxnet::common::random;

template<typename xpu, typename IType, typename OType, typename Sampler, int inum>
struct SamplerCaller;

template<typename xpu, typename IType, typename OType, typename Sampler>
struct SamplerCaller<xpu, IType, OType, Sampler, 1> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 RandGenerator<xpu, OType> *pgen,
                 mshadow::Stream<xpu> *s) {
    Sampler sampler;
    sampler.Sample(inputs[0].FlatTo1D<xpu, IType>(s),
                   outputs[0].FlatTo1D<xpu, OType>(s),
                   pgen, s);
  }
};

template<typename xpu, typename IType, typename OType, typename Sampler>
struct SamplerCaller<xpu, IType, OType, Sampler, 2> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 RandGenerator<xpu, OType> *pgen,
                 mshadow::Stream<xpu> *s) {
    Sampler sampler;
    sampler.Sample(inputs[0].FlatTo1D<xpu, IType>(s),
                   inputs[1].FlatTo1D<xpu, IType>(s),
                   outputs[0].FlatTo1D<xpu, OType>(s),
                   pgen, s);
  }
};

template<typename xpu, typename Sampler, int inum>
void MultiSampleOpForward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), inum);
  CHECK_EQ(outputs.size(), 1);
  CHECK_GT(inputs[0].Size(), 0);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      RandGenerator<xpu, OType> *pgen = ctx.requested[0].get_parallel_random<xpu, OType>();
      SamplerCaller<xpu, IType, OType, Sampler, inum>::op(inputs, outputs, pgen, s);
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_RANDOM_MULTISAMPLE_OP_H_
