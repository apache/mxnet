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
Operator* CreateOp<cpu>(SliceChannelParam param, int dtype) {
  Operator* op = nullptr;
  MSHADOW_TYPE_SWITCH(dtype, DType, {
    op = new SliceChannelOp<cpu, DType>(param);
  })
  return op;
}

Operator* SliceChannelProp::CreateOperatorEx(Context ctx,
                                             std::vector<TShape>* in_shape,
                                             std::vector<int>* in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SliceChannelParam);

MXNET_REGISTER_OP_PROPERTY(SliceChannel, SliceChannelProp)
.describe(R"code(Splits an array along a particular axis into multiple sub-arrays.

.. note:: ``SliceChannel`` is depreacted. Use ``split`` instead.

**Note** that `num_outputs` should evenly divide the length of the axis 
along which to split the array.

Example::

   x  = [[[ 1.]
          [ 2.]]
         [[ 3.]
          [ 4.]]
         [[ 5.]
          [ 6.]]]
   x.shape = (3, 2, 1)

   y = split(x, axis=1, num_outputs=2) // a list of 2 arrays with shape (3, 1, 1)
   y = [[[ 1.]]
        [[ 3.]]
        [[ 5.]]]

       [[[ 2.]]
        [[ 4.]]
        [[ 6.]]]

   y[0].shape = (3, 1, 1)

   z = split(x, axis=0, num_outputs=3) // a list of 3 arrays with shape (1, 2, 1)
   z = [[[ 1.]
         [ 2.]]]

       [[[ 3.]
         [ 4.]]]

       [[[ 5.]
         [ 6.]]]

   z[0].shape = (1, 2, 1)

`squeeze_axis=1` removes the axis with length 1 from the shapes of the output arrays.
**Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 only
along the `axis` which it is split.
Also `squeeze_axis` can be set to true only if ``input.shape[axis] == num_outputs``.

   z = split(x, axis=0, num_outputs=3, squeeze_axis=1) // a list of 3 arrays with shape (2, 1)
   z = [[ 1.]
        [ 2.]]

       [[ 3.]
        [ 4.]]

       [[ 5.]
        [ 6.]]
   z[0].shape = (2 ,1 )

)code" ADD_FILELINE)
.set_return_type("NDArray-or-Symbol[]")
.add_argument("data", "NDArray-or-Symbol", "The input")
.add_arguments(SliceChannelParam::__FIELDS__());

NNVM_REGISTER_OP(SliceChannel).add_alias("split");

}  // namespace op
}  // namespace mxnet
