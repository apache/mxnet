/*!
 *  Copyright (c) 2016 by Contributors
 * \file init_op.cc
 * \brief CPU Implementation of init op
 */
#include "./init_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(InitOpParam);

NNVM_REGISTER_OP(_zeros)
.describe("fill target with zeros")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<nnvm::FInferShape>("FInferShape", InitShape<InitOpParam>)
.set_attr<nnvm::FInferType>("FInferType", InitType)
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 0>)
.add_arguments(InitOpParam::__FIELDS__());

NNVM_REGISTER_OP(_ones)
.describe("fill target with ones")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<nnvm::FInferShape>("FInferShape", InitShape<InitOpParam>)
.set_attr<nnvm::FInferType>("FInferType", InitType)
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 1>)
.add_arguments(InitOpParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
