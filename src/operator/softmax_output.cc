/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_output.cc
 * \brief
 * \author Bing Xu
*/
#include "./softmax_output-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SoftmaxOutputParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SoftmaxOutputOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SoftmaxOutputProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SoftmaxOutputParam);

MXNET_REGISTER_OP_PROPERTY(SoftmaxOutput, SoftmaxOutputProp)
.describe(R"code(Computes softmax with logit loss.

In the forward pass, the softmax output is returned. Assume the input data has
shape *(n,k)*, then the output will have the same shape as the input, which is computed by

.. math::
   out[i,:] = softmax(data[i,:])

for :math:`i=0,...,n-1`, where

.. math::
   softmax(x) = \left[..., \frac{exp(x[j])}{exp(x[0])+...+exp(x[k-1])}, ...\right]

For general *N*-D input array with shape :math:`(d_1, ..., d_n)`. Denoted by the size
:math:`s=d_1d_2...d_n`. The way to compute softmax various:

- ``preserve_shape`` is false (default). Reshape input into a 2-D array with
  shape :math:`(d_1, s/d_1)` beforing computing the softmax, and then reshaped back to the
  original shape.

- ``preserve_shape`` is true. For all :math:`i_1, ..., i_{n-1}`, compute

  .. math::
    out[i_1, ..., i_{n-1}, :] = softmax(data[i_1, ..., i_{n-1},:])

- ``multi_output`` is true. For all :math:`i_1, ..., i_{n-1}`, compute

  .. math::
    out[i_1, :, ..., i_{n-1}] = softmax(data[i_1, :, ..., i_{n-1}])

In the backward pass, the logit loss, also called cross-entroy loss, is
added. The provided label can be a *(N-1)*-D label index array or a *N*-D label
probability array.

Examples with a particular label can be ignored during backward by specifying
``ignore_label`` (also need ``use_ignore`` to be true).

A scale can be applied to the gradient by ``grad_scale``, which is often used in
mutli-loss object function in which we can given each loss different weight. It
also supports various ways to normalize the gradient by ``normalization``:

- **null**: do nothing
- **batch**: divide by batch size (number of examples)
- **valid**: divide by the number of examples which are not ignored.
)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_argument("label", "NDArray-or-Symbol", "Ground truth label.")
.add_arguments(SoftmaxOutputParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(Softmax, DeprecatedSoftmaxProp)
.describe("Perform a softmax transformation on input. Please use SoftmaxOutput"
          ".. note::  `Softmax`` is deprecated. Use `softmax`")
.add_argument("data", "NDArray-or-Symbol", "Input data to softmax.")
.add_arguments(SoftmaxOutputParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
