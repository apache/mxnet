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
 * \file np_delete_op.cc
 * \brief CPU Implementation of numpy insert operations
 */

#include <vector>
#include "./np_delete_op-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyDeleteParam);

bool NumpyDeleteType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *in_type,
                     std::vector<int> *out_type) {
  const NumpyDeleteParam& param = nnvm::get<NumpyDeleteParam>(attrs.parsed);
  int insize = (param.step.has_value() || param.int_ind.has_value()) ? 1 : 2;
  CHECK_EQ(in_type->size(), insize);
  CHECK_EQ(out_type->size(), 1U);
  if (insize == 3) {
    CHECK_NE((*in_type)[1], -1) << "Index type must be set for insert operator\n";
    CHECK(((*in_type)[1] == mshadow::DataType<int64_t>::kFlag) ||
          ((*in_type)[1] == mshadow::DataType<int32_t>::kFlag))
      << "Index type only support int32 or int64.\n";
  }
  TYPE_ASSIGN_CHECK(*out_type, 0, (*in_type)[0]);
  TYPE_ASSIGN_CHECK(*in_type, 0, (*out_type)[0]);
  return (*in_type)[0] != -1;
}

inline bool NumpyDeleteStorageType(const nnvm::NodeAttrs& attrs,
                                   const int dev_mask,
                                   DispatchMode* dispatch_mode,
                                   std::vector<int> *in_attrs,
                                   std::vector<int> *out_attrs) {
  const NumpyDeleteParam& param = nnvm::get<NumpyDeleteParam>(attrs.parsed);
  unsigned int insize = (param.step.has_value() || param.int_ind.has_value()) ? 1U : 2U;
  CHECK_EQ(in_attrs->size(), insize);
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

NNVM_REGISTER_OP(_npi_delete)
.describe(R"code(Delete values along the given axis before the given indices.)code" ADD_FILELINE)
.set_attr_parser(ParamParser<NumpyDeleteParam>)
.set_num_inputs([](const NodeAttrs& attrs) {
  const NumpyDeleteParam& params = nnvm::get<NumpyDeleteParam>(attrs.parsed);
  return (params.step.has_value() || params.int_ind.has_value()) ? 1U : 2U;
})
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const NumpyDeleteParam& params = nnvm::get<NumpyDeleteParam>(attrs.parsed);
    return (params.step.has_value() || params.int_ind.has_value()) ?
            std::vector<std::string>{"arr"} :
            std::vector<std::string>{"arr", "obj"};
})
.set_attr<nnvm::FInferType>("FInferType", NumpyDeleteType)
.set_attr<mxnet::FComputeEx>("FComputeEx<cpu>", NumpyDeleteCompute<cpu>)
.set_attr<FInferStorageType>("FInferStorageType", NumpyDeleteStorageType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("arr", "NDArray-or-Symbol", "Input ndarray")
.add_argument("obj", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(NumpyDeleteParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
