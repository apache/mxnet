/*!
 *  Copyright (c) 2016 by Contributors
 * \file init_op.cc
 * \brief CPU Implementation of init op
 */
#include "./init_op.h"
#include "./elemwise_unary_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(InitOpParam);
DMLC_REGISTER_PARAMETER(RangeParam);


NNVM_REGISTER_OP(_zeros)
.describe("fill target with zeros")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<nnvm::FInferShape>("FInferShape", InitShape<InitOpParam>)
.set_attr<nnvm::FInferType>("FInferType", InitType<InitOpParam>)
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 0>)
.add_arguments(InitOpParam::__FIELDS__());

NNVM_REGISTER_OP(_ones)
.describe("fill target with ones")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<nnvm::FInferShape>("FInferShape", InitShape<InitOpParam>)
.set_attr<nnvm::FInferType>("FInferType", InitType<InitOpParam>)
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 1>)
.add_arguments(InitOpParam::__FIELDS__());

NNVM_REGISTER_OP(_arange)
.describe("Return evenly spaced values within a given interval. Similar to Numpy")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(RangeParamParser)
.set_attr<nnvm::FInferShape>("FInferShape", RangeShape)
.set_attr<nnvm::FInferType>("FInferType", InitType<RangeParam>)
.set_attr<FCompute>("FCompute<cpu>", RangeCompute<cpu>)
.add_arguments(RangeParam::__FIELDS__());

NNVM_REGISTER_OP(zeros_like)
.describe(R"code(Return an array of zeros with the same shape and type
as the input array.

Examples::

  x = [[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]]

  zeros_like(x) = [[ 0.,  0.,  0.],
                   [ 0.,  0.,  0.]]

)code")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FIgnoreInputs>("FIgnoreInputs",
    [](const NodeAttrs& attrs) { return std::vector<uint32_t>(1, 0); })
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 0>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("data", "NDArray-or-Symbol", "The input");

NNVM_REGISTER_OP(ones_like)
.describe(R"code(Return an array of ones with the same shape and type
as the input array.

Examples::

  x = [[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]]

  ones_like(x) = [[ 1.,  1.,  1.],
                  [ 1.,  1.,  1.]]

)code")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FIgnoreInputs>("FIgnoreInputs",
    [](const NodeAttrs& attrs) { return std::vector<uint32_t>(1, 0); })
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 1>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("data", "NDArray-or-Symbol", "The input");

}  // namespace op
}  // namespace mxnet
