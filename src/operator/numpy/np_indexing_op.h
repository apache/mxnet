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
 * \file np_indexing_op.h
 * \brief Function definition of numpy indexing operator
*/
#ifndef MXNET_OPERATOR_NUMPY_NP_INDEXING_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_INDEXING_OP_H_

#include <vector>
#include "../contrib/boolean_mask-inl.h"
#include "../tensor/indexing_op.h"
#include "../tensor/broadcast_reduce_op.h"
#ifdef __CUDACC__
#include "../tensor/indexing_op-inl.cuh"
#endif

namespace mxnet {
namespace op {

namespace np_indexing_ {  // to avoid name conflict
enum Inputs {kArr, kIdx};
enum Outputs {kOut};
}  // namespace np_indexing_

struct AdvancedIndexingMultipleParam: public dmlc::Parameter<AdvancedIndexingMultipleParam> {
  int axis;
  DMLC_DECLARE_PARAMETER(AdvancedIndexingMultipleParam) {
    DMLC_DECLARE_FIELD(axis)
    .set_default(0)
    .describe("The axis of tuple type indexing");
  }
};

template<typename xpu>
void AdvancedIndexingOpForward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<NDArray>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<NDArray>& outputs);

// template<typename xpu>
// void AdvancedIndexingMultipleOpForward(const nnvm::NodeAttrs& attrs,
//                    const OpContext& ctx,
//                    const std::vector<NDArray>& inputs,
//                    const std::vector<OpReqType>& req,
//                    const std::vector<NDArray>& outputs);

template<typename xpu>
void AdvancedIndexingOpBackward(const nnvm::NodeAttrs& attrs,
                                     const OpContext &ctx,
                                     const std::vector<NDArray> &inputs,
                                     const std::vector<OpReqType> &req,
                                     const std::vector<NDArray> &outputs);

// template<typename xpu>
// void AdvancedIndexingMultipleOpBackward(const nnvm::NodeAttrs& attrs,
//                          const OpContext& ctx,
//                          const std::vector<NDArray>& inputs,
//                          const std::vector<OpReqType>& req,
//                          const std::vector<NDArray>& outputs);

template<typename xpu>
void AdvancedIndexingMultipleBackward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using nnvm::dim_t;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  if (req[0] == kNullOp) return;

  if (inputs[np_indexing_::kIdx].type_flag_ == mshadow::kBool) {
    LOG(FATAL)
    << "Multi-dimension boolean indexing is not supported.";
  } else if (inputs[np_indexing_::kIdx].type_flag_ == mshadow::kInt8 ||
             inputs[np_indexing_::kIdx].type_flag_ == mshadow::kInt16 ||
             inputs[np_indexing_::kIdx].type_flag_ == mshadow::kInt32 ||
             inputs[np_indexing_::kIdx].type_flag_ == mshadow::kInt64) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    const mxnet::TShape& oshape = outputs[0].shape_;
    const mxnet::TShape& ishape = inputs[1].shape_;
    dim_t M = ishape[0];
    dim_t N = ishape.Size() / M;
    dim_t K = oshape.ProdShape(M, oshape.ndim());
    mshadow::Shape<10> strides;
    for (dim_t i = M-1, stride = K; i >= 0; stride *= oshape[i], --i) strides[i] = stride;
    if (kWriteTo == req[0]) {
      Fill<true>(s, outputs[0], req[0], 0);
    }
    MXNET_NO_INT8_TYPE_SWITCH(inputs[0].type_flag_, DType, {  // output data type switch
      MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType, {  // indices data type switch
        GatherNDBackwardImpl(N, M, K, strides,
                            outputs[0].dptr<DType>(),
                            inputs[0].dptr<DType>(),
                            inputs[1].dptr<IType>(),
                            s);
      });
    });
  } else {
    LOG(FATAL)
    << "arrays used as indices must be explictly declared as integer (or boolean) type."
    << "Use np.astype() to cast indices to integer or boolean.";
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_INDEXING_OP_H_
