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
DMLC_REGISTER_PARAMETER(SGDMomParam);
DMLC_REGISTER_PARAMETER(AdamParam);
DMLC_REGISTER_PARAMETER(RMSPropParam);
DMLC_REGISTER_PARAMETER(RMSPropAlexParam);

NNVM_REGISTER_OP(sgd_update)
.describe("Updater function for sgd optimizer")
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SGDParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<2, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FCompute>("FCompute<cpu>", SGDUpdate<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "gradient")
.add_arguments(SGDParam::__FIELDS__());

NNVM_REGISTER_OP(sgd_mom_update)
.describe("Updater function for sgd optimizer")
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SGDMomParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<3, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 1>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2};
  })
.set_attr<FCompute>("FCompute<cpu>", SGDMomUpdate<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mom", "NDArray-or-Symbol", "Momentum")
.add_arguments(SGDMomParam::__FIELDS__());

NNVM_REGISTER_OP(adam_update)
.describe("Updater function for adam optimizer")
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr_parser(ParamParser<AdamParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<4, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<4, 1>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3};
  })
.set_attr<FCompute>("FCompute<cpu>", AdamUpdate<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mean", "NDArray-or-Symbol", "Moving mean")
.add_argument("var", "NDArray-or-Symbol", "Moving variance")
.add_arguments(AdamParam::__FIELDS__());

NNVM_REGISTER_OP(rmsprop_update)
.describe("Updater function for RMSProp optimizer."
          " The RMSProp code follows the version in"
          " http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf "
          "Tieleman & Hinton, 2012.")
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<RMSPropParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<3, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 1>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs &attrs) {
    return std::vector<uint32_t>{2};
  })
.set_attr<FCompute>("FCompute<cpu>", RMSPropUpdate<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("n", "NDArray-or-Symbol", "n")
.add_arguments(RMSPropParam::__FIELDS__());

NNVM_REGISTER_OP(rmspropalex_update)
.describe("Updater function for RMSPropAlex optimizer."
          " The RMSPropAlex code follows the version in"
          " http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.")
.set_num_inputs(5)
.set_num_outputs(1)
.set_attr_parser(ParamParser<RMSPropAlexParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<5, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<5, 1>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3, 4};
  })
.set_attr<FCompute>("FCompute<cpu>", RMSPropAlexUpdate<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("n", "NDArray-or-Symbol", "n")
.add_argument("g", "NDArray-or-Symbol", "g")
.add_argument("delta", "NDArray-or-Symbol", "delta")
.add_arguments(RMSPropAlexParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
