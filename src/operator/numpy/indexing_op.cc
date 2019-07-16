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
 * Copyright (c) 2019 by Contributors
 * \file indexing_op.cc
 * \brief CPU implementation of numpy indexing operator
*/

#include "./indexing_op.h"

namespace mxnet {
namespace op {

template<>
void NumpyTakeOpForward<cpu>(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[take_::kOut] == kNullOp) return;
  const NumpyTakeParam& param = nnvm::get<NumpyTakeParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  const mxnet::TShape& idxshape = inputs[take_::kIdx].shape_;
  const mxnet::TShape& arrshape = inputs[take_::kArr].shape_;
  const mxnet::TShape& oshape = outputs[take_::kOut].shape_;

  if (idxshape.Size() == 0) {
    return;
  }

  Stream<cpu> *s = ctx.get_stream<cpu>();

  if (param.axis.has_value()) {
    const int actual_axis = param.axis.value() + ((param.axis.value() < 0) ? arrshape.ndim() : 0);

    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {  // output data type
      MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType, {  // index data type
        if (actual_axis == 0) {
          if (param.mode == take_::kClip) {
            Kernel<TakeCPU<true>, cpu>::Launch(s, idxshape.Size(),
                                               outputs[take_::kOut].dptr<DType>(),
                                               inputs[take_::kArr].dptr<DType>(),
                                               inputs[take_::kIdx].dptr<IType>(),
                                               oshape.Size()/idxshape.Size(), arrshape[0]);
          } else {
            Kernel<TakeCPU<false>, cpu>::Launch(s, idxshape.Size(),
                                                outputs[take_::kOut].dptr<DType>(),
                                                inputs[take_::kArr].dptr<DType>(),
                                                inputs[take_::kIdx].dptr<IType>(),
                                                oshape.Size()/idxshape.Size(), arrshape[0]);
          }
        } else {
          mshadow::Shape<10> in_strides;
          int stride = 1;
          for (int i = arrshape.ndim() - 1; i >= 0; stride *= arrshape[i], --i) {
            in_strides[i] = stride;
          }
          mshadow::Shape<10> out_strides;
          stride = 1;
          for (int i = oshape.ndim() - 1; i >= 0; stride *= oshape[i], --i) {
            out_strides[i] = stride;
          }
          if (param.mode == take_::kClip) {
            Kernel<Take<true>, cpu>::Launch(s, oshape.Size(),
                                            outputs[take_::kOut].dptr<DType>(),
                                            inputs[take_::kArr].dptr<DType>(),
                                            inputs[take_::kIdx].dptr<IType>(),
                                            in_strides, out_strides, arrshape.ndim(),
                                            oshape.ndim(), idxshape.ndim(),
                                            arrshape[actual_axis], actual_axis);
          } else if (param.mode == take_::kWrap) {
            Kernel<Take<false>, cpu>::Launch(s, oshape.Size(),
                                             outputs[take_::kOut].dptr<DType>(),
                                             inputs[take_::kArr].dptr<DType>(),
                                             inputs[take_::kIdx].dptr<IType>(),
                                             in_strides, out_strides, arrshape.ndim(),
                                             oshape.ndim(), idxshape.ndim(),
                                             arrshape[actual_axis], actual_axis);
          }
        }
      });
    });
  } else {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {  // output data type
      MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType, {  // index data type
        if (param.mode == take_::kClip) {
          Kernel<TakeCPU<true>, cpu>::Launch(s, idxshape.Size(),
                                             outputs[take_::kOut].dptr<DType>(),
                                             inputs[take_::kArr].dptr<DType>(),
                                             inputs[take_::kIdx].dptr<IType>(),
                                             static_cast<size_t>(1), static_cast<int64_t>(arrshape.Size()));
        } else {
          Kernel<TakeCPU<false>, cpu>::Launch(s, idxshape.Size(),
                                              outputs[take_::kOut].dptr<DType>(),
                                              inputs[take_::kArr].dptr<DType>(),
                                              inputs[take_::kIdx].dptr<IType>(),
                                              static_cast<size_t>(1), static_cast<int64_t>(arrshape.Size()));
        }
      });
    });
  }
}

DMLC_REGISTER_PARAMETER(NumpyTakeParam);

NNVM_REGISTER_OP(_npi_take)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyTakeParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "indices"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", NumpyTakeOpShape)
.set_attr<nnvm::FInferType>("FInferType", TakeOpType)
.set_attr<FCompute>("FCompute<cpu>", NumpyTakeOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n,  const std::vector<nnvm::NodeEntry>& ograds) {
    return MakeNonlossGradNode("_backward_np_take", n, ograds,
                               {n->inputs[1]}, n->attrs.dict);
  })
.add_argument("a", "NDArray-or-Symbol", "The input array.")
.add_argument("indices", "NDArray-or-Symbol", "The indices of the values to be extracted.")
.add_arguments(NumpyTakeParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_np_take)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr_parser(ParamParser<NumpyTakeParam>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", NumpyTakeOpBackward<cpu>);

}  // namespace op
}  // namespace mxnet
