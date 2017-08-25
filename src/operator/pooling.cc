/*!
 * Copyright (c) 2017 by Contributors
 * \file pooling.cc
 * \brief
 * \author Bing Xu, Jun Wu
*/
#include "./pooling-inl.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_pooling-inl.h"
#endif  // MXNET_USE_MKL2017
#if MXNET_USE_NNPACK == 1
#include "./nnpack/nnpack_pooling-inl.h"
#endif  // MXNET_USE_NNPACK

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(PoolingParam param, int dtype) {
  Operator *op = NULL;
  // TODO(lingyan): kFull use exclude padding algorithm now
#if MXNET_USE_MKL2017 == 1
    if (param.kernel.ndim() == 2
      && (param.pooling_convention == pool_enum::kValid)
      && ((param.pool_type == pool_enum::kMaxPooling)
      || (param.pool_type == pool_enum::kAvgPooling))) {
      switch (dtype) {
      case mshadow::kFloat32:
        return new MKLPoolingOp<cpu, float>(param);
      case mshadow::kFloat64:
        return new MKLPoolingOp<cpu, double>(param);
      default:
        break;
      }
    }
    LOG(INFO) << MKLPoolingOp<cpu, float>::getName() << " Skip MKL optimization";
#endif
#if MXNET_USE_NNPACK == 1
  // NNPACK only support max-pooling with kernel = 2, stride = 2, pooling_convention
  // = kFull(note that the default value is kValid in MXNet)
  if ((param.pool_type == pool_enum::kMaxPooling)
    && (param.pooling_convention == pool_enum::kFull)
    && (param.kernel.ndim() == 2) && (param.stride.ndim() == 2)
    && (param.kernel[0] == 2) && (param.kernel[1] == 2)
    && (param.stride[0] == 2) && (param.stride[1] == 2)) {
    switch (dtype) {
    case mshadow::kFloat32:
      return new NNPACKPoolingOp<cpu, float>(param);
    default:
      break;
    }
  }
#endif
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (pool_enum::kMaxPooling == param.pool_type
        || pool_enum::kAvgPooling == param.pool_type
        || pool_enum::kSumPooling == param.pool_type) {
      op = new PoolingOp<cpu, DType>(param);
    } else {
      LOG(FATAL) << "unknown pooling type";
      return NULL;
    }
  });

  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator* PoolingProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(PoolingParam);

MXNET_REGISTER_OP_PROPERTY(Pooling, PoolingProp)
.describe(R"code(Performs pooling on the input.

The shapes for 1-D pooling are

- **data**: *(batch_size, channel, width)*,
- **out**: *(batch_size, num_filter, out_width)*.

The shapes for 2-D pooling are

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

For 3-D pooling, an additional *depth* dimension is added before
*height*. Namely the input data will have shape *(batch_size, channel, depth,
height, width)*.

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the pooling operator.")
.add_arguments(PoolingParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
