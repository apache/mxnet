/*!
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file deformable_convolution.cc
 * \brief
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai
*/

#include "./deformable_convolution-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(DeformableConvolutionParam);

template<>
Operator* CreateOp<cpu>(DeformableConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new DeformableConvolutionOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *DeformableConvolutionProp::CreateOperatorEx(Context ctx,
                                            std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

MXNET_REGISTER_OP_PROPERTY(_contrib_DeformableConvolution, DeformableConvolutionProp)
.describe(R"code(Compute *N*-D convolution on *(N+2)*-D input.

In the 2-D convolution, given input data with shape *(batch_size,
channel, height, width)*, the output is computed by

.. math::

   out[n,i,:,:] = bias[i] + \sum_{j=0}^{num\_filter} data[n,j,:,:] \star
   weight[i,j,:,:]

where :math:`\star` is the 2-D cross-correlation operator.

For general 2-D convolution, the shapes are

- **data**: *(batch_size, channel, height, width)*
- **weight**: *(num_filter, channel, kernel[0], kernel[1])*
- **bias**: *(num_filter,)*
- **out**: *(batch_size, num_filter, out_height, out_width)*.

Define::

  f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1

then we have::

  out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
  out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])

If ``no_bias`` is set to be true, then the ``bias`` term is ignored.

The default data ``layout`` is *NCHW*, namely *(batch_size, channle, height,
width)*. We can choose other layouts such as *NHWC*.

If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
evenly into *g* parts along the channel axis, and also evenly split ``weight``
along the first dimension. Next compute the convolution on the *i*-th part of
the data with the *i*-th weight part. The output is obtained by concating all
the *g* results.

Both ``weight`` and ``bias`` are learnable parameters.


)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the DeformableConvolutionOp.")
.add_argument("offset", "NDArray-or-Symbol", "Input offset to the DeformableConvolutionOp.")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_argument("bias", "NDArray-or-Symbol", "Bias parameter.")
.add_arguments(DeformableConvolutionParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
