/*!
 * Copyright (c) 2015 by Contributors
 * \file quantized_lrn.cc
 * \brief
 * \author Ziheng Jiang
*/

#include "./quantized_lrn-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(QuantizedLRNParam param, int dtype) {
  Operator *op = NULL;
  LOG(FATAL) << "not implemented yet";
  // MSHADOW_TYPE_SWITCH(dtype, DType, {
  //   op = new QuantizedLRNOp<DType>(param);
  // })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator* QuantizedLRNProp::CreateOperatorEx(Context ctx,
    std::vector<TShape> *in_shape, std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(QuantizedLRNParam);

MXNET_REGISTER_OP_PROPERTY(quantized_lrn, QuantizedLRNProp)
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_argument("min_data", "NDArray-or-Symbol", "")
.add_argument("max_data", "NDArray-or-Symbol", "")
.add_arguments(QuantizedLRNParam::__FIELDS__())
.describe(R"code(Applies local response normalization to the input.

The local response normalization layer performs “lateral inhibition” by normalizing
over local input regions.

If :math:`a_{x,y}^{i}` is the activity of a neuron computed by applying kernel :math:`i` at position
:math:`(x, y)` and then applying the ReLU nonlinearity, the response-normalized
activity :math:`b_{x,y}^{i}` is given by the expression:

.. math::
   b_{x,y}^{i} = \frac{a_{x,y}^{i}}{\Bigg({k + \alpha \sum_{j=max(0, i-\frac{n}{2})}^{min(N-1, i+\frac{n}{2})} (a_{x,y}^{j})^{2}}\Bigg)^{\beta}}

where the sum runs over :math:`n` “adjacent” kernel maps at the same spatial position, and :math:`N` is the total
number of kernels in the layer.

)code" ADD_FILELINE);

}  // namespace op
}  // namespace mxnet
