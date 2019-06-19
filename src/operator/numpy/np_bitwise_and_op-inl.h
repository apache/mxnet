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
 * \file np_bitwise_and_op-inl.h
 * \brief Function definition of element-wise binary operator: bitwise_and
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_BITWISE_AND_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_BITWISE_AND_OP_INL_H_

#include "../mxnet_op.h"




namespace mxnet{
namespace op {

//struct BitwiseAndParam : public dmlc::Parameter<BitwiseAndParam> {
//        float a, b, c;
//        DMLC_DECLARE_PARAMETER(QuadraticParam) {
//                DMLC_DECLARE_FIELD(a)
//                        .set_default(0.0)
//                        .describe("Coefficient of the quadratic term in the quadratic function.");
//                DMLC_DECLARE_FIELD(b)
//                .set_default(0.0)
//                .describe("Coefficient of the linear term in the quadratic function.");
//                DMLC_DECLARE_FIELD(c)
//                .set_default(0.0)
//                .describe("Constant term in the quadratic function.");
//        }
//};

/*!
 * \brief Shape inference
 */
inline bool BitwiseAndOpShape(const nnvm::NodeAttrs& attrs,
                              mxnet::ShapeVector* in_attrs,
                              mxnet::ShapeVector* out_attrs) {
    CHECK_EQ(in_attrs->size(), 2U);
    CHECK_EQ(out_attrs->size(), 1U);  // only one array (or value) as output

    SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
    return out_attrs->at(0).ndim() != 0U && out_attrs->at(0).Size() != 0U;
}

/*!
 * \brief Data type inference
 */
inline bool BitwiseAndOpType(const nnvm::NodeAttrs& attrs,
                              mxnet::ShapeVector* in_attrs,
                              mxnet::ShapeVector* out_attrs) {
    CHECK_EQ(in_attrs->size(), 2U);
    CHECK_EQ(out_attrs->size(), 1U);  // only one array (or value) as output

    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
    TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
    return out_attrs->at(0) != -1;
}

/*!
 * \brief kernel struct
 */
template<int req>
struct bitwise_and_forward {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType* out, const DType* lhs, const DType* rhs) {
        KERNEL_ASSIGN(out[i], req, lhs[i] & rhs[i]);
    }
};

/*!
 * \brief forward function
 * \input two tensors
 * \output a single tensor
 */
template<typename xpu>
void BitwiseAndOpForward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector& inputs,
                         const std::vector& req,
                         const std::vector& outputs) {
    CHECK_EQ(inputs.size(), 2U);
    CHECK_EQ(outputs.size(), 1U);
    CHECK_EQ(req.size(), 1U);
    mshadow::Stream *s = ctx.get_stream();
    const TBlob& lhs = inputs[0];
    const TBlob& rhs = inputs[1];
    const TBlob& out_data = outputs[0];
    // const QuadraticParam& param = nnvm::get(attrs.parsed);
    using namespace mxnet_op;
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
            MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
                    Kernel<quadratic_forward<req_type>, xpu>::Launch(
                            s, out_data.Size(), out_data.dptr(), in_data.dptr());
            });
    });
}

} // namespace op
} // namespace mxnet
#endif
