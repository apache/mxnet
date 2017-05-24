/*!
 * Copyright (c) 2015 by Contributors
 * \file pooling_v1.cc
 * \brief
 * \author Bing Xu
*/
#include "./pooling_v1-inl.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(PoolingV1Param param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    switch (param.pool_type) {
      case pool_v1_enum::kMaxPooling:
        op = new PoolingV1Op<cpu, mshadow::red::maximum, DType>(param);
        break;
      case pool_v1_enum::kAvgPooling:
        op = new PoolingV1Op<cpu, mshadow::red::sum, DType>(param);
        break;
      case pool_v1_enum::kSumPooling:
        op = new PoolingV1Op<cpu, mshadow::red::sum, DType>(param);
        break;
      default:
        LOG(FATAL) << "unknown pooling type";
        return NULL;
    }
  })

  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator* PoolingV1Prop::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(PoolingV1Param);

MXNET_REGISTER_OP_PROPERTY(Pooling_v1, PoolingV1Prop)
.describe(R"code(This operator is DEPRECATED.
Perform pooling on the input.

The shapes for 2-D pooling is

- **data**: *(batch_size, channel, height, width)*
- **out**: *(batch_size, num_filter, out_height, out_width)*, with::

    out_height = f(height, kernel[0], pad[0], stride[0])
    out_width = f(width, kernel[1], pad[1], stride[1])

The definition of *f* depends on ``pooling_convention``, which has two options:

- **valid** (default)::

    f(x, k, p, s) = floor((x+2*p-k)/s)+1

- **full**, which is compatible with Caffe::

    f(x, k, p, s) = ceil((x+2*p-k)/s)+1

But ``global_pool`` is set to be true, then do a global pooling, namely reset
``kernel=(height, width)``.

Three pooling options are supported by ``pool_type``:

- **avg**: average pooling
- **max**: max pooling
- **sum**: sum pooling

1-D pooling is special case of 2-D pooling with *weight=1* and
*kernel[1]=1*.

For 3-D pooling, an additional *depth* dimension is added before
*height*. Namely the input data will have shape *(batch_size, channel, depth,
height, width)*.

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the pooling operator.")
.add_arguments(PoolingV1Param::__FIELDS__());

}  // namespace op
}  // namespace mxnet
