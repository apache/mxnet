/*!
 *  Copyright (c) 2016 by Contributors
 * \file optimizer_op.cc
 * \brief Optimizer operators
 * \author Junyuan Xie
 */
#include "./optimizer_op-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(SGDParam);

NNVM_REGISTER_OP(sgd_update)
.describe("Updater function for sgd optimizer")
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SGDParam>)
.attr<nnvm::FInferShape>("FInferShape", UniformShape)
.attr<nnvm::FInferType>("FInferType", UniformType)
.attr<FCompute>("FCompute<cpu>", SGDUpdate<cpu>);

NNVM_REGISTER_OP(sgd_mom_update)
.describe("Updater function for sgd optimizer")
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SGDParam>)
.attr<nnvm::FInferShape>("FInferShape", UniformShape)
.attr<nnvm::FInferType>("FInferType", UniformType)
.attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2};
  })
.attr<FCompute>("FCompute<cpu>", SGDMomUpdate<cpu>);


DMLC_REGISTER_PARAMETER(AdamParam);

NNVM_REGISTER_OP(adam_update)
.describe("Updater function for adam optimizer")
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr_parser(ParamParser<AdamParam>)
.attr<nnvm::FInferShape>("FInferShape", UniformShape)
.attr<nnvm::FInferType>("FInferType", UniformType)
.attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3};
  })
.attr<FCompute>("FCompute<cpu>", AdamUpdate<cpu>);

}  // namespace op
}  // namespace mxnet
