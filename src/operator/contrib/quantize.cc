/*!
 *  Copyright (c) 2017 by Contributors
 * \file quantize.cc
 * \brief
 */
#include "./quantize-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(QuantizeParam);

NNVM_REGISTER_OP(_contrib_quantize)
.MXNET_DESCRIBE("Quantize the input tensor of type float to output tensor of type 'out_type'.")
.set_attr_parser(ParamParser<QuantizeParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", QuantizeType)
.set_attr<FCompute>("FCompute<cpu>", QuantizeCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_quantize"})
.add_argument("input", "NDArray-or-Symbol", "A ndarray/symbol of type `float32`")
.add_arguments(QuantizeParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
