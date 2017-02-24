/*!
 * Copyright (c) 2015 by Contributors
 * \file slice_channel.cc
 * \brief
 * \author Bing Xu
*/

#include "./slice_channel-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(SliceChannelParam param) {
  return new SliceChannelOp<cpu>(param);
}

Operator* SliceChannelProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(SliceChannelParam);

MXNET_REGISTER_OP_PROPERTY(SliceChannel, SliceChannelProp)
.describe(R"code(Split an array along a particular axis into multiple sub-arrays.

Assume the input array has shape ``(d_0, ..., d_n)`` and we slice it into *m*
(``num_outputs=m``) subarrays along axis *k*, then we will obtain a list of *m*
arrays with each of which has shape ``(d_0, ..., d_k/m, ..., d_n)``.

For example::

  x = [[1, 2],
       [3, 4],
       [5, 6],
       [7, 8]]  // 4x2 array

  y = split(x, axis=0, num_outputs=4) // a list of 4 arrays
  y[0] = [[ 1.,  2.]]  // 1x2 array

  z = split(x, axis=0, num_outputs=2) // a list of 2 arrays
  z[0] = [[ 1.,  2.],
          [ 3.,  4.]]

When setting optional argument ``squeeze_axis=1``, then the *k*-dimension will
be removed from the shape if it becomes 1::

  y = split(x, axis=0, num_outputs=4, squeeze_axis=1)
  y[0] = [ 1.,  2.]  // (2,) vector

)code" ADD_FILELINE)
.set_return_type("ndarray-or-symbol[]")
.add_arguments(SliceChannelParam::__FIELDS__());

NNVM_REGISTER_OP(SliceChannel).add_alias("split");

}  // namespace op
}  // namespace mxnet
