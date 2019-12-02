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
 * \file prepare_op-common.h
 * \brief Common functions for intgemm's PrepareA and PrepareB functions.
 * These are used to convert float tensors to values suitable for
 * multiplication.
 */

#include <mxnet/operator_util.h>
#include <vector>
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../tensor/init_op.h"

namespace mxnet {
namespace op {

struct PrepareParam : public dmlc::Parameter<PrepareParam> {
  float multiplier;
  DMLC_DECLARE_PARAMETER(PrepareParam) {
    DMLC_DECLARE_FIELD(multiplier)
      .describe("Multiply floats by this constant before casting to int8.  Typically you would set this to 127.0 / max absolute value.");
  }
};

bool PrepareOpShape(const nnvm::NodeAttrs& attrs,
                    mxnet::ShapeVector* in_attrs,
                    mxnet::ShapeVector* out_attrs);

bool PrepareOpType(const nnvm::NodeAttrs& attrs,
                   std::vector<int>* in_attrs,
                   std::vector<int>* out_attrs);

bool PrepareOpStorageType(const nnvm::NodeAttrs& attrs,
                          const int dev_mask,
                          DispatchMode* dispatch_mode,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs);

}  // namespace op
}  // namespace mxnet
