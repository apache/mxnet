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
 * \file indexing_op.cu
 * \brief GPU implementation of indexing operator
 */

#include "./indexing_op.h"

namespace mxnet {
namespace op {

template<>
void NumpyTakeOpForward<gpu>(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[take_::kOut] == kNullOp) return;
  const TakeParam& param = nnvm::get<TakeParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  const mxnet::TShape& idxshape = inputs[take_::kIdx].shape_;
  const mxnet::TShape& arrshape = inputs[take_::kArr].shape_;
  const mxnet::TShape& oshape = outputs[take_::kOut].shape_;

  Stream<gpu> *s = ctx.get_stream<gpu>();

  if (param.axis.has_value()) {
    const int actual_axis = param.axis + ((param.axis < 0) ? arrshape.ndim() : 0);

    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {  // output data type
      MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType, {  // index data type
        if (actual_axis == 0) {
          if (param.mode == take_::kClip) {
            Kernel<TakeGPU<true>, gpu>::Launch(s, oshape.Size(),
                                               outputs[take_::kOut].dptr<DType>(),
                                               inputs[take_::kArr].dptr<DType>(),
                                               inputs[take_::kIdx].dptr<IType>(),
                                               oshape.Size()/idxshape.Size(), arrshape[0]);
          } else {
            Kernel<TakeGPU<false>, gpu>::Launch(s, oshape.Size(),
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
            Kernel<Take<true>, gpu>::Launch(s, oshape.Size(),
                                            outputs[take_::kOut].dptr<DType>(),
                                            inputs[take_::kArr].dptr<DType>(),
                                            inputs[take_::kIdx].dptr<IType>(),
                                            in_strides, out_strides, arrshape.ndim(), oshape.ndim(),
                                            idxshape.ndim(), arrshape[actual_axis], actual_axis);
          } else if (param.mode == take_::kWrap) {
            Kernel<Take<false>, gpu>::Launch(s, oshape.Size(),
                                             outputs[take_::kOut].dptr<DType>(),
                                             inputs[take_::kArr].dptr<DType>(),
                                             inputs[take_::kIdx].dptr<IType>(),
                                             in_strides, out_strides, arrshape.ndim(), oshape.ndim(),
                                             idxshape.ndim(), arrshape[actual_axis], actual_axis);
          }
        }
      });
    });
  } else {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {  // output data type
      MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType, {  // index data type
        if (param.mode == take_::kClip) {
          Kernel<TakeGPU<true>, gpu>::Launch(s, oshape.Size(),
                                             outputs[take_::kOut].dptr<DType>(),
                                             inputs[take_::kArr].dptr<DType>(),
                                             inputs[take_::kIdx].dptr<IType>(),
                                             static_cast<size_t>(1), static_cast<int64_t>(arrshape.Size()));
        } else {
          Kernel<TakeGPU<false>, gpu>::Launch(s, oshape.Size(),
                                              outputs[take_::kOut].dptr<DType>(),
                                              inputs[take_::kArr].dptr<DType>(),
                                              inputs[take_::kIdx].dptr<IType>(),
                                              static_cast<size_t>(1), static_cast<int64_t>(arrshape.Size()));
        }
      });
    });
  }
}

NNVM_REGISTER_OP(_npi_take)
.set_attr<FCompute>("FCompute<gpu>", NumpyTakeOpForward<gpu>);

NNVM_REGISTER_OP(_backward_np_take)
.set_attr<FCompute>("FCompute<gpu>", NumpyTakeOpBackward<gpu>);

}  // namespace op
}  // namespace mxnet
