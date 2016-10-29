/*!
 * Copyright (c) 2015 by Contributors
 * \file pad.cc
 * \brief
 * \author Sebastian Bodenstein
*/
#include "./pad-inl.h"

namespace mshadow {

////////////////////////////////////////////////////////////////////////////////
// Special Case: 2d image (so only pad width + height)

// Case 1: Replication Padding
// single_image_2d_replicate adapted from Torch
// https://github.com/torch/nn/blob/master/lib/THNN/generic/SpatialReplicationPadding.c
template <typename DType>
void single_image_2d_replicate(const Tensor<cpu, 3, DType> &dst,
                               const Tensor<cpu, 3, DType> src,
                               mxnet::TShape pad) {
  const int nslices = src.size(0);
  const int iheight = src.size(1);
  const int iwidth = src.size(2);

  const int oheight = dst.size(1);
  const int owidth = dst.size(2);

  const int pad_t = pad[4];
  const int pad_l = pad[6];
  int iStartX = std::max(0, -pad_l);
  int iStartY = std::max(0, -pad_t);
  int oStartX = std::max(0, pad_l);
  int oStartY = std::max(0, pad_t);

  int k, ip_x, ip_y;
#pragma omp parallel for private(k, ip_x, ip_y)
  for (k = 0; k < nslices; k++) {
    int i, j;
    for (i = 0; i < oheight; i++) {
      for (j = 0; j < owidth; j++) {
        if (j < pad_l) {
          ip_x = pad_l;
        } else if (j >= pad_l && j < iwidth + pad_l) {
          ip_x = j;
        } else {
          ip_x = iwidth + pad_l - 1;
        }
        ip_x = ip_x - oStartX + iStartX;
        if (i < pad_t) {
          ip_y = pad_t;
        } else if (i >= pad_t && i < iheight + pad_t) {
          ip_y = i;
        } else {
          ip_y = iheight + pad_t - 1;
        }
        ip_y = ip_y - oStartY + iStartY;

        DType *dest_p = dst.dptr_ + k * owidth * oheight + i * owidth + j;
        DType *src_p = src.dptr_ + k * iwidth * iheight + ip_y * iwidth + ip_x;
        *dest_p = *src_p;
      }
    }
  }
}

template <typename DType>
void single_image_2d_replicate_grad(const Tensor<cpu, 3, DType> &grad_in,
                                    const Tensor<cpu, 3, DType> grad_out,
                                    mxnet::TShape pad) {
  const int nslices = grad_in.size(0);
  const int iheight = grad_in.size(1);
  const int iwidth = grad_in.size(2);

  const int oheight = grad_out.size(1);
  const int owidth = grad_out.size(2);

  const int pad_t = pad[4];
  const int pad_l = pad[6];
  int iStartX = std::max(0, -pad_l);
  int iStartY = std::max(0, -pad_t);
  int oStartX = std::max(0, pad_l);
  int oStartY = std::max(0, pad_t);

  int k, ip_x, ip_y;
#pragma omp parallel for private(k, ip_x, ip_y)
  for (k = 0; k < nslices; k++) {
    int i, j;
    for (i = 0; i < oheight; i++) {
      for (j = 0; j < owidth; j++) {
        if (j < pad_l) {
          ip_x = pad_l;
        } else if (j >= pad_l && j < iwidth + pad_l) {
          ip_x = j;
        } else {
          ip_x = iwidth + pad_l - 1;
        }
        ip_x = ip_x - oStartX + iStartX;

        if (i < pad_t) {
          ip_y = pad_t;
        } else if (i >= pad_t && i < iheight + pad_t) {
          ip_y = i;
        } else {
          ip_y = iheight + pad_t - 1;
        }
        ip_y = ip_y - oStartY + iStartY;

        DType *src_p = grad_out.dptr_ + k * owidth * oheight + i * owidth + j;
        DType *dest_p =
            grad_in.dptr_ + k * iwidth * iheight + ip_y * iwidth + ip_x;
        *dest_p += *src_p;
      }
    }
  }
}

// Case 2: Zero Padding
template <typename DType>
void single_image_2d_constant(const Tensor<cpu, 3, DType> &dst,
                              const Tensor<cpu, 3, DType> src,
                              mxnet::TShape pad, DType padding_constant) {
  const int pad_t = pad[4];
  const int pad_l = pad[6];
#pragma omp parallel for private(c, w, h)
  for (index_t c = 0; c < dst.size(0); ++c) {
    for (index_t h = 0; h < dst.size(1); ++h) {
      for (index_t w = 0; w < dst.size(2); ++w) {
        if ((w < pad_l) || (h < pad_t) || (h >= (src.size(1) + pad_t)) ||
            (w >= (src.size(2) + pad_l))) {
          dst[c][h][w] = padding_constant;
        } else {
          dst[c][h][w] = src[c][h - pad_t][w - pad_l];
        }
      }
    }
  }
}

template <typename DType>
void single_image_2d_constant_grad(const Tensor<cpu, 3, DType> &in_grad,
                                   const Tensor<cpu, 3, DType> out_grad,
                                   mxnet::TShape pad) {
  const int pad_t = pad[4];
  const int pad_l = pad[6];
#pragma omp parallel for private(c, w, h)
  for (index_t c = 0; c < in_grad.size(0); ++c) {
    for (index_t h = 0; h < in_grad.size(1); ++h) {
      for (index_t w = 0; w < in_grad.size(2); ++w) {
        in_grad[c][h][w] += out_grad[c][h + pad_t][w + pad_l];
      }
    }
  }
}

// General 2d image case
template <typename DType>
void pad_image_2d(const Tensor<cpu, 4, DType> &dst,
                  const Tensor<cpu, 4, DType> src, mxnet::TShape pad,
                  int pad_type, DType padding_constant) {
  for (index_t n = 0; n < dst.size(0); ++n) {
    switch (pad_type) {
      case mxnet::op::pad_enum::kReplicate:
        single_image_2d_replicate(dst[n], src[n], pad);
        break;
      case mxnet::op::pad_enum::kConstant:
        single_image_2d_constant(dst[n], src[n], pad, padding_constant);
        break;
    }
  }
}

template <typename DType>
void pad_image_2d_grad(const Tensor<cpu, 4, DType> &in_grad,
                       const Tensor<cpu, 4, DType> out_grad, mxnet::TShape pad,
                       int pad_type) {
  for (index_t n = 0; n < in_grad.size(0); ++n) {
    switch (pad_type) {
      case mxnet::op::pad_enum::kReplicate:
        single_image_2d_replicate_grad(in_grad[n], out_grad[n], pad);
        break;
      case mxnet::op::pad_enum::kConstant:
        single_image_2d_constant_grad(in_grad[n], out_grad[n], pad);
        break;
    }
  }
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template <>
Operator *CreateOp<cpu>(PadParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, { op = new PadOp<cpu, DType>(param); })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *PadProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                    std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(PadParam);

MXNET_REGISTER_OP_PROPERTY(Pad, PadProp)
    .describe("")
    .add_argument("data", "Symbol", 
    "Pads an n-dimensional input tensor. The padding amount pad_shape is a tuple"
    " of size 2*n. For example, a pad_shape of [9,5,4,2] adds 9 padding values before"
    "the first dimension, 5 padding values after the first dimension, 4 padding values "
    "before the second dimension, and 2 padding values after the second dimension."
  )
    .add_arguments(PadParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
