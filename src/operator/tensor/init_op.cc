/*!
 *  Copyright (c) 2016 by Contributors
 * \file init_op.cc
 * \brief CPU Implementation of init op
 */
#include "./init_op.h"

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

}  // namespace op
}  // namespace mxnet
