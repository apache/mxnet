/*!
 * Copyright (c) 2017 by Contributors
 * \file convolution.cc
 * \brief
 * \author Jun Wu
*/

#include "./convolution2-inl.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_convolution-inl.h"
#endif  // MXNET_USE_MKL2017
#if MXNET_USE_NNPACK == 1
#include "./nnpack/nnpack_convolution-inl.h"
#endif  // MXNET_USE_NNPACK
#include "./nn/im2col.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(Convolution2Param);

template<typename xpu, typename DType>
inline
void Convolution2Op<xpu, DType>::ConvIm2Col(const index_t n, const TBlob& data,
                                            TBlob* col_buffer, const cpu& dev_cpu) const {
  const DType* data_ptr = data.dptr<DType>() + n * input_dim_;
  if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
    im2col_cpu(data_ptr, conv_in_channels_, data.size(2), data.size(3),
               param_.kernel[0], param_.kernel[1], param_.pad[0], param_.pad[1],
               param_.stride[0], param_.stride[1], param_.dilate[0], param_.dilate[1],
               col_buffer->dptr<DType>());
  } else {
    im2col_nd_cpu(data_ptr, num_spatial_axes_,
                  reinterpret_cast<const int*>(&(data.shape_[1])),
                  reinterpret_cast<int*>(col_buffer->shape_.data()),
                  reinterpret_cast<const int*>(param_.kernel.data()),
                  reinterpret_cast<const int*>(param_.pad.data()),
                  reinterpret_cast<const int*>(param_.stride.data()),
                  reinterpret_cast<const int*>(param_.dilate.data()),
                  col_buffer->dptr<DType>());
  }
}

template<>
Operator* CreateOp<cpu>(Convolution2Param param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
#if MXNET_USE_MKL2017 == 1
  if ((param.dilate[0] == 1 && param.dilate[1] == 1)
      && param.kernel.ndim() == 2) {
    switch (dtype) {
    case mshadow::kFloat32:
      return new MKLConvolution2Op<cpu, float>(param);
    case mshadow::kFloat64:
      return new MKLConvolution2Op<cpu, double>(param);
    default:
      break;
    }
  }
  LOG(INFO) << MKLConvolution2Op<cpu, float>::getName() << " Skip MKL optimization";
#endif
#if MXNET_USE_NNPACK == 1
  const size_t batch_size = (*in_shape)[0][0];
  if ((param.dilate[0] == 1 && param.dilate[1] == 1)
      && param.kernel.ndim() == 2 && (!param.no_bias)
      && param.num_group == 1 && (batch_size == 1 ||
      ((batch_size > 1) && (param.stride[0] == 1) &&
      (param.stride[1] == 1)))) {
    switch (dtype) {
    case mshadow::kFloat32:
      return new NNPACKConvolution2Op<cpu, float>(param);
    default:
      break;
    }
  }
#endif
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new Convolution2Op<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *Convolution2Prop::CreateOperatorEx(Context ctx,
                                            std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

MXNET_REGISTER_OP_PROPERTY(Convolution2, Convolution2Prop)
.add_argument("data", "Symbol", "Input data to the Convolution2Op.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(Convolution2Param::__FIELDS__())
.describe("Apply convolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet
