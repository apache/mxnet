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
 * \file np_elemwise_broadcast_op_extended_thi.cc
 * \brief CPU Implementation of extended functions for elementwise numpy binary broadcast operator.
 * (Third extended file)
 */

#include "../../common/utils.h"
#include "./np_elemwise_broadcast_op.h"

namespace mxnet {
namespace op {

#define MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(name)                        \
  NNVM_REGISTER_OP(name)                                                      \
      .set_num_inputs(1)                                                      \
      .set_num_outputs(1)                                                     \
      .set_attr_parser(ParamParser<NumpyBinaryScalarParam>)                   \
      .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)       \
      .set_attr<nnvm::FInferType>("FInferType", NumpyBinaryScalarType)        \
      .set_attr<FResourceRequest>(                                            \
          "FResourceRequest",                                                 \
          [](const NodeAttrs& attrs) {                                        \
            return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; \
          })                                                                  \
      .add_argument("data", "NDArray-or-Symbol", "source input")              \
      .add_arguments(NumpyBinaryScalarParam::__FIELDS__())

MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_INT_PRECISION(_npi_bitwise_left_shift)
    .set_attr<FCompute>("FCompute<cpu>",
                        NumpyBinaryBroadcastIntCompute<cpu, mshadow_op::bitwise_left_shift>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_bitwise_left_shift"});

NNVM_REGISTER_OP(_npi_bitwise_left_shift_scalar)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyBinaryScalarParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseIntType<1, 1>)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}};
                                    })
    .set_attr<FCompute>("FCompute<cpu>",
                        BinaryScalarOp::Compute<cpu, mshadow_op::bitwise_left_shift>)
    .set_attr<nnvm::FGradient>("FGradient",
                               ElemwiseGradUseIn{"_backward_npi_bitwise_left_shift_scalar"})
    .add_argument("data", "NDArray-or-Symbol", "source input")
    .add_arguments(NumpyBinaryScalarParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_rbitwise_left_shift_scalar)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyBinaryScalarParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseIntType<1, 1>)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}};
                                    })
    .set_attr<FCompute>("FCompute<cpu>",
                        BinaryScalarOp::Compute<cpu, mshadow_op::rbitwise_left_shift>)
    .set_attr<nnvm::FGradient>("FGradient",
                               ElemwiseGradUseIn{"_backward_npi_rbitwise_left_shift_scalar"})
    .add_argument("data", "NDArray-or-Symbol", "source input")
    .add_arguments(NumpyBinaryScalarParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_npi_bitwise_left_shift)
    .set_num_inputs(3)
    .set_num_outputs(2)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 1}};
                                    })
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FCompute>("FCompute<cpu>",
                        BinaryBroadcastBackwardUseIn<cpu,
                                                     mshadow_op::bitwise_left_shift_grad,
                                                     mshadow_op::bitwise_left_shift_rgrad>);

NNVM_REGISTER_OP(_backward_npi_bitwise_left_shift_scalar)
    .add_arguments(NumpyBinaryScalarParam::__FIELDS__())
    .set_attr_parser(ParamParser<NumpyBinaryScalarParam>)
    .set_attr<FCompute>("FCompute<cpu>",
                        BinaryScalarOp::Backward<cpu, mshadow_op::bitwise_left_shift_grad>);

MXNET_OPERATOR_REGISTER_BINARY(_backward_npi_rbitwise_left_shift_scalar)
    .add_arguments(NumpyBinaryScalarParam::__FIELDS__())
    .set_attr_parser(ParamParser<NumpyBinaryScalarParam>)
    .set_attr<FCompute>("FCompute<cpu>",
                        BinaryScalarOp::Backward<cpu, mshadow_op::rbitwise_left_shift_grad>);

MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_INT_PRECISION(_npi_bitwise_right_shift)
    .set_attr<FCompute>("FCompute<cpu>",
                        NumpyBinaryBroadcastIntCompute<cpu, mshadow_op::bitwise_right_shift>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_bitwise_right_shift"});

NNVM_REGISTER_OP(_npi_bitwise_right_shift_scalar)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyBinaryScalarParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseIntType<1, 1>)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}};
                                    })
    .set_attr<FCompute>("FCompute<cpu>",
                        BinaryScalarOp::Compute<cpu, mshadow_op::bitwise_right_shift>)
    .set_attr<nnvm::FGradient>("FGradient",
                               ElemwiseGradUseIn{"_backward_npi_bitwise_right_shift_scalar"})
    .add_argument("data", "NDArray-or-Symbol", "source input")
    .add_arguments(NumpyBinaryScalarParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_rbitwise_right_shift_scalar)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyBinaryScalarParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseIntType<1, 1>)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}};
                                    })
    .set_attr<FCompute>("FCompute<cpu>",
                        BinaryScalarOp::Compute<cpu, mshadow_op::rbitwise_right_shift>)
    .set_attr<nnvm::FGradient>("FGradient",
                               ElemwiseGradUseIn{"_backward_npi_rbitwise_right_shift_scalar"})
    .add_argument("data", "NDArray-or-Symbol", "source input")
    .add_arguments(NumpyBinaryScalarParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_npi_bitwise_right_shift)
    .set_num_inputs(3)
    .set_num_outputs(2)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 1}};
                                    })
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FCompute>("FCompute<cpu>",
                        BinaryBroadcastBackwardUseIn<cpu,
                                                     mshadow_op::bitwise_right_shift_grad,
                                                     mshadow_op::bitwise_right_shift_rgrad>);

NNVM_REGISTER_OP(_backward_npi_bitwise_right_shift_scalar)
    .add_arguments(NumpyBinaryScalarParam::__FIELDS__())
    .set_attr_parser(ParamParser<NumpyBinaryScalarParam>)
    .set_attr<FCompute>("FCompute<cpu>",
                        BinaryScalarOp::Backward<cpu, mshadow_op::bitwise_right_shift_grad>);

MXNET_OPERATOR_REGISTER_BINARY(_backward_npi_rbitwise_right_shift_scalar)
    .add_arguments(NumpyBinaryScalarParam::__FIELDS__())
    .set_attr_parser(ParamParser<NumpyBinaryScalarParam>)
    .set_attr<FCompute>("FCompute<cpu>",
                        BinaryScalarOp::Backward<cpu, mshadow_op::rbitwise_right_shift_grad>);

}  // namespace op
}  // namespace mxnet
