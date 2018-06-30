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
 * \file allreduce-inl.h
 * \brief all reduce operator
 * \author Hang Zhang
 */
#ifndef MXNET_OPERATOR_ALL_REDUCE_INL_H_
#define MXNET_OPERATOR_ALL_REDUCE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/ndarray.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../ndarray/ndarray_function.h"
#include "./operator_common.h"
#include "./mxnet_op.h"
#include "./mshadow_op.h"
#include "../kvstore/comm.h"

namespace mxnet {
namespace op {

struct AllReduceOpParam : public dmlc::Parameter<AllReduceOpParam> {
    int num_args;
    DMLC_DECLARE_PARAMETER(AllReduceOpParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
    .describe("Number of inputs to be allreduced.");
  }
}; // struct AllReduceOpParam

template<typename xpu>
inline void AllReduceOpForwardEx(const nnvm::NodeAttrs& attrs,    
                                 const OpContext &ctx,
                                 const std::vector<NDArray> &inputs,
                                 const std::vector<OpReqType> &req,
                                 const std::vector<NDArray> &outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(inputs.size(), outputs.size());
  CHECK_EQ(inputs.size(), req.size());
  //int priority = 0;
  // create buf
  std::vector<NDArray> reduce(inputs.size());
  NDArray out(outputs[0].shape(), outputs[0].ctx(), false, outputs[0].dtype());
  // copy to buf
  for (size_t i = 0; i < inputs.size(); ++i) {
    //inputs[i].WaitToRead();
    reduce[i] = NDArray(
      outputs[0].shape(), outputs[0].ctx(), false, outputs[0].dtype());
    //CopyFromTo(inputs[i], &(reduce[i]), priority);
    TBlob tmp = reduce[i].data();
    ndarray::Copy<xpu, xpu>(inputs[i].data(), &tmp,
                            inputs[i].ctx(), reduce[i].ctx(), ctx.run_ctx);
  }
  // all reduce
  std::vector<TBlob> source_tblob(reduce.size());
  for (size_t i = 0; i < reduce.size(); ++i) {
    source_tblob[i] = reduce[i].data();
  }
  TBlob tmp = out.data();
  ndarray::ElementwiseSum<xpu>(source_tblob, &tmp, ctx.run_ctx);
  // copy to each
  for (size_t i = 0; i < outputs.size(); ++i) {
    TBlob tmp = outputs[i].data();
    ndarray::Copy<xpu, xpu>(out.data(), &tmp,
                            out.ctx(), outputs[i].ctx(), ctx.run_ctx);
  }
}


inline bool AllReduceShape(const nnvm::NodeAttrs& attrs,
                           std::vector<TShape> *in_attrs,
                           std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), out_attrs->size());
  for (int i = 0; i < static_cast<int>(in_attrs->size()); ++i) {
    TShape& ishape = (*in_attrs)[i];
    SHAPE_ASSIGN_CHECK(*out_attrs, i, ishape);
  }
  for (int i = 0; i < static_cast<int>(in_attrs->size()); ++i) {
    TShape& ishape = (*out_attrs)[i];
    SHAPE_ASSIGN_CHECK(*in_attrs, i, ishape);
  }
  return true;
}

inline bool AllReduceType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  int dtype = (*in_attrs)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";
  // assign
  for (int i = 0; i < static_cast<int>(in_attrs->size()); ++i) {
    dtype = (*in_attrs)[i];
    TYPE_ASSIGN_CHECK(*out_attrs, i, dtype);
  }
  for (int i = 0; i < static_cast<int>(in_attrs->size()); ++i) {
    dtype = (*out_attrs)[i];
    TYPE_ASSIGN_CHECK(*in_attrs, i, dtype);
  }
  return true;
}

inline bool AllReduceStorageType(const nnvm::NodeAttrs& attrs,
                                 const int dev_mask,
                                 DispatchMode* dispatch_mode,
                                 std::vector<int>* in_attrs,
                                 std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), out_attrs->size());
  *dispatch_mode = DispatchMode::kFComputeEx;
  for (int& v : *in_attrs) {
    if (v == - 1) v = kDefaultStorage;
  }
  for (size_t i = 0; i < out_attrs->size(); i++) {
    (*out_attrs)[i] = kDefaultStorage;
  }
  return true;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ALL_REDUCE_INL_H_
