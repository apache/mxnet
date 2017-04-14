/*!
 *  Copyright (c) 2017 by Contributors
 * \file dequantize.cc
 * \brief
 */
#include "./dequantize-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(DequantizeParam);

NNVM_REGISTER_OP(_contrib_dequantize)
.MXNET_DESCRIBE("Dequantize")
.set_attr_parser(ParamParser<DequantizeParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", DequantizeType)
.set_attr<FCompute>("FCompute<cpu>", DequantizeCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_dequantize"})
.add_argument("input", "NDArray-or-Symbol", "A ndarray/symbol of type `uint8`")
.add_arguments(DequantizeParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
