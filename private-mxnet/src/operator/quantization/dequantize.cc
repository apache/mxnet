/*!
 *  Copyright (c) 2017 by Contributors
 * \file dequantize.cc
 * \brief
 */
#include "./dequantize-inl.h"
#if MXNET_USE_MKLDNN == 1
#include <mkl_memory.h>
#include "../mkl/mkldnn_memory-inl.h"
#include "./mkl/mkldnn_dequantize-inl.h"
#endif

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(DequantizeParam);

NNVM_REGISTER_OP(_contrib_dequantize)
.describe(R"code(Dequantize the input tensor into a float tensor.
[min_range, max_range] are scalar floats that spcify the range for
the output data.

Each value of the tensor will undergo the following:

`out[i] = min_range + (in[i] * (max_range - min_range) / range(INPUT_TYPE))`

here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<DequantizeParam>)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", DequantizeShape)
.set_attr<nnvm::FInferType>("FInferType", DequantizeType)
#if MXNET_USE_MKLDNN == 1 
.set_attr<FCompute>("FCompute<cpu>", MKLDequantizeCompute)
#else
.set_attr<FCompute>("FCompute<cpu>", DequantizeCompute<cpu>)
#endif 
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_dequantize"})
.add_argument("input", "NDArray-or-Symbol", "A ndarray/symbol of type `uint8`")
.add_argument("min_range", "NDArray-or-Symbol", "The minimum scalar value "
  "possibly produced for the input")
.add_argument("max_range", "NDArray-or-Symbol", "The maximum scalar value "
  "possibly produced for the input")
.add_arguments(DequantizeParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
