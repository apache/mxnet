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
* \file digitize_op.cc
* \brief
* \author Contributors
*/

#include "./digitize_op.h"
#include <mxnet/base.h>
#include <vector>
#include <algorithm>


namespace mxnet {
namespace op {

// TODO: How to use templated pointers instead of TBlobs here?
template<>
void DigitizeOp::ForwardKernel::Map<cpu>(int i,
                                         const OpContext &ctx,
                                         const TBlob &input_data,
                                         const TBlob &bins,
                                         TBlob &out_data,
                                         const bool right){
  using namespace mshadow;

  auto s = ctx.get_stream<cpu>();

  MSHADOW_TYPE_SWITCH(bins.type_flag_, BType, {
    const Tensor<cpu, 1, BType> bins_tensor = bins.FlatTo1D<cpu, BType>(s);

    MSHADOW_TYPE_SWITCH(input_data.type_flag_, OType, {
      const auto *data = input_data.FlatTo1D<cpu, OType>(s).dptr_;

      auto elem = right ? std::lower_bound(bins.dptr_, bins.dptr_ + bins.size(0), data[i])
                        : std::upper_bound(bins.dptr_, bins.dptr_ + bins.size(0), data[i]);

      out_data[i] = std::distance(bins.dptr_, elem);
    });
  });

}


DMLC_REGISTER_PARAMETER(DigitizeParam);

NNVM_REGISTER_OP(digitize)
    .describe(R"code(Full operator description:
.. math::

    ...

)code" ADD_FILELINE)
    .set_attr_parser(ParamParser<DigitizeParam>)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs &attrs) {
                                       return std::vector<std::string>{ "data" };
                                     })
    .set_attr<nnvm::FInferShape>("FInferShape", DigitizeOp::InferShape)
        //.set_attr<nnvm::FInferType>("FInferType", DigitizeOpType)
    .set_attr<FCompute>("FCompute", DigitizeOp::Forward<cpu>)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs &attrs) {
                                      return std::vector<std::pair<int, int>>{{ 0, 0 }};
                                    })
    .add_argument("data", "NDArray-or-Symbol", "Input ndarray")
    .add_arguments(DigitizeParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
