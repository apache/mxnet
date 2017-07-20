/*!
 * Copyright (c) 2015 by Contributors
 * \file l2_normalization.cc
 * \brief l2 normalization operator
*/
#include "./l2_normalization-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(L2NormalizationParam param) {
  return new L2NormalizationOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator* L2NormalizationProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(L2NormalizationParam);

MXNET_REGISTER_OP_PROPERTY(L2Normalization, L2NormalizationProp)
.describe(R"code(Normalize the input array using the L2 norm.

For 1-D NDArray, it computes::

  out = data / sqrt(sum(data ** 2) + eps)

For N-D NDArray, if the input array has shape (N, N, ..., N),

with ``mode`` = ``instance``, it normalizes each instance in the multidimensional
array by its L2 norm.::

  for i in 0...N
    out[i,:,:,...,:] = data[i,:,:,...,:] / sqrt(sum(data[i,:,:,...,:] ** 2) + eps)

with ``mode`` = ``channel``, it normalizes each channel in the array by its L2 norm.::

  for i in 0...N
    out[:,i,:,...,:] = data[:,i,:,...,:] / sqrt(sum(data[:,i,:,...,:] ** 2) + eps)

with ``mode`` = ``spatial``, it normalizes the cross channel norm for each position
in the array by its L2 norm.::

  for dim in 2...N
    for i in 0...N
      out[.....,i,...] = take(out, indices=i, axis=dim) / sqrt(sum(take(out, indices=i, axis=dim) ** 2) + eps)
          -dim-

Example::

  x = [[[1,2],
        [3,4]],
       [[2,2],
        [5,6]]]

  L2Normalization(x, mode='instance')
  =[[[ 0.18257418  0.36514837]
     [ 0.54772252  0.73029673]]
    [[ 0.24077171  0.24077171]
     [ 0.60192931  0.72231513]]]

  L2Normalization(x, mode='channel')
  =[[[ 0.31622776  0.44721359]
     [ 0.94868326  0.89442718]]
    [[ 0.37139067  0.31622776]
     [ 0.92847669  0.94868326]]]

  L2Normalization(x, mode='spatial')
  =[[[ 0.44721359  0.89442718]
     [ 0.60000002  0.80000001]]
    [[ 0.70710677  0.70710677]
     [ 0.6401844   0.76822126]]]

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input array to normalize.")
.add_arguments(L2NormalizationParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
