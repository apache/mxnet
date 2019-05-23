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

#include "./fused_op.h"
#include "../operator_common.h"
#include "../../executor/exec_pass.h"

namespace mxnet {

DMLC_REGISTER_PARAMETER(FusedOpConfig);

void FusedOpParamParser(nnvm::NodeAttrs* attrs) {
  FusedOpConfig param;
  try {
    param.Init(attrs->dict);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  CHECK(!param.symbol_json.empty());
  attrs->parsed = FusedOpPtr(new FusedOp(param));
}

NNVM_REGISTER_OP(FusedOp)
.set_num_inputs([](const NodeAttrs& attrs) {
    const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
    return op->num_inputs();
  })
.set_num_outputs([](const NodeAttrs& attrs) {
    const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
    return op->num_outputs();
  })
.set_attr_parser(FusedOpParamParser)
.add_argument("data", "NDArray-or-Symbol[]", "Data");

}  // namespace mxnet
