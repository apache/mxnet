/*!
 * Copyright (c) 2017 by Contributors
 * \file max_pool.cc
 * \brief
 * \author Bing Xu, Jun Wu
*/
#include "./max_pool-inl.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(MaxPoolParam param, int dtype) {
  LOG(INFO) << "not implemented";
  Operator *op = NULL;
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator* MaxPoolProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(MaxPoolParam);

MXNET_REGISTER_OP_PROPERTY(max_pool, MaxPoolProp)
.describe(R"code(Performs max_pool on the input.

The shapes for 1-D max_pool are

- **data**: *(batch_size, channel, width)*,
- **out**: *(batch_size, num_filter, out_width)*.

The shapes for 2-D max_pool are

- **data**: *(batch_size, channel, height, width)*
- **out**: *(batch_size, num_filter, out_height, out_width)*, with::

    out_height = f(height, kernel[0], pad[0], stride[0])
    out_width = f(width, kernel[1], pad[1], stride[1])

The defintion of *f* depends on ``max_pool_convention``, which has two options:

- **valid** (default)::

    f(x, k, p, s) = floor(x+2*p-k)/s+1

- **full**, which is compatible with Caffe::

    f(x, k, p, s) = ceil(x+2*p-k)/s+1

But ``global_pool`` is set to be true, then do a global max_pool, namely reset
``kernel=(height, width)``.

Three max_pool options are supported by ``pool_type``:

- **avg**: average max_pool
- **max**: max max_pool
- **sum**: sum max_pool

For 3-D max_pool, an additional *depth* dimension is added before
*height*. Namely the input data will have shape *(batch_size, channel, depth,
height, width)*.

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the max_pool operator.")
.add_arguments(MaxPoolParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
