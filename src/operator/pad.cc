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

// Case 1: Edge Padding (or Replication Padding)
// single_image_2d_edge adapted from Torch
// https://github.com/torch/nn/blob/master/lib/THNN/generic/SpatialReplicationPadding.c
template <typename DType>
void single_image_edge(const Tensor<cpu, 3, DType> dst,
                       const Tensor<cpu, 3, DType> src, mxnet::TShape pad) {
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
void single_image_edge_grad(const Tensor<cpu, 3, DType> &grad_in,
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
void single_image_constant(const Tensor<cpu, 3, DType> &dst,
                           const Tensor<cpu, 3, DType> src, mxnet::TShape pad,
                           DType constant_value) {
  const int pad_t = pad[4];
  const int pad_l = pad[6];
  int c, w, h;
#pragma omp parallel for private(c, w, h)
  for (c = 0; c < dst.size(0); ++c) {
    for (h = 0; h < dst.size(1); ++h) {
      for (w = 0; w < dst.size(2); ++w) {
        if ((w < pad_l) || (h < pad_t) || (h >= (src.size(1) + pad_t)) ||
            (w >= (src.size(2) + pad_l))) {
          dst[c][h][w] = constant_value;
        } else {
          dst[c][h][w] = src[c][h - pad_t][w - pad_l];
        }
      }
    }
  }
}

template <typename DType>
void single_image_constant_grad(const Tensor<cpu, 3, DType> &in_grad,
                                const Tensor<cpu, 3, DType> out_grad,
                                mxnet::TShape pad) {
  const int pad_t = pad[4];
  const int pad_l = pad[6];
  int c, h, w;
#pragma omp parallel for private(c, w, h)
  for (c = 0; c < in_grad.size(0); ++c) {
    for (h = 0; h < in_grad.size(1); ++h) {
      for (w = 0; w < in_grad.size(2); ++w) {
        in_grad[c][h][w] += out_grad[c][h + pad_t][w + pad_l];
      }
    }
  }
}

// Case 3: Reflection Padding
template <typename DType>
void single_image_reflect(const Tensor<cpu, 3, DType> &dst,
                           const Tensor<cpu, 3, DType> src, mxnet::TShape pad) {
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
          ip_x = pad_l * 2 - j;
        } else if (j >= pad_l && j < iwidth + pad_l) {
          ip_x = j;
        } else {
          ip_x = (iwidth + pad_l - 1) * 2 - j;
        }
        ip_x = ip_x - oStartX + iStartX;

        if (i < pad_t) {
          ip_y = pad_t * 2 - i;
        } else if (i >= pad_t && i < iheight + pad_t) {
          ip_y = i;
        } else {
          ip_y = (iheight + pad_t - 1) * 2 - i;
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
void single_image_reflect_grad(const Tensor<cpu, 3, DType> &grad_in,
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
          ip_x = pad_l * 2 - j;
        } else if (j >= pad_l && j < iwidth + pad_l) {
          ip_x = j;
        } else {
          ip_x = (iwidth + pad_l - 1) * 2 - j;
        }
        ip_x = ip_x - oStartX + iStartX;

        if (i < pad_t) {
          ip_y = pad_t * 2 - i;
        } else if (i >= pad_t && i < iheight + pad_t) {
          ip_y = i;
        } else {
          ip_y = (iheight + pad_t - 1) * 2 - i;
        }
        ip_y = ip_y - oStartY + iStartY;

        DType *src_p = grad_out.dptr_ + k * owidth * oheight + i * owidth + j;
        DType *dest_p = grad_in.dptr_ + k * iwidth * iheight + ip_y * iwidth + ip_x;
        *dest_p += *src_p;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Special Case: 3d image (so only pad width + height + depth)

// Case 1: Edge Padding (or Replication Padding)
// single_image_3d_edge adapted from Torch
// https://github.com/torch/nn/blob/master/lib/THNN/generic/VolumetricReplicationPadding.c
template <typename DType>
void single_image_edge(const Tensor<cpu, 4, DType> dst,
                       const Tensor<cpu, 4, DType> src, mxnet::TShape pad) {
  const int nslices = src.size(0);
  const int idepth = src.size(1);
  const int iheight = src.size(2);
  const int iwidth = src.size(3);

  const int odepth = dst.size(1);
  const int oheight = dst.size(2);
  const int owidth = dst.size(3);

  const int pad_f = pad[4];
  const int pad_t = pad[6];
  const int pad_l = pad[8];
  int iStartX = std::max(0, -pad_l);
  int iStartY = std::max(0, -pad_t);
  int iStartZ = std::max(0, -pad_f);
  int oStartX = std::max(0, pad_l);
  int oStartY = std::max(0, pad_t);
  int oStartZ = std::max(0, pad_f);

  int k, ip_x, ip_y, ip_z;
#pragma omp parallel for private(k, ip_x, ip_y, ip_z)
  for (k = 0; k < nslices; k++) {
    int i, j, z;
    for (z = 0; z < odepth; z++) {
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

          if (z < pad_f) {
            ip_z = pad_f;
          } else if (z >= pad_f && z < idepth + pad_f) {
            ip_z = z;
          } else {
            ip_z = idepth + pad_f - 1;
          }
          ip_z = ip_z - oStartZ + iStartZ;

          DType *dest_p = dst.dptr_ + k * owidth * oheight * odepth +
                          z * owidth * oheight + i * owidth + j;
          DType *src_p = src.dptr_ + k * iwidth * iheight * idepth +
                         ip_z * iwidth * iheight + ip_y * iwidth + ip_x;
          *dest_p = *src_p;
        }
      }
    }
  }
}

template <typename DType>
void single_image_edge_grad(const Tensor<cpu, 4, DType> &grad_in,
                            const Tensor<cpu, 4, DType> grad_out,
                            mxnet::TShape pad) {
  const int nslices = grad_in.size(0);
  const int idepth = grad_in.size(1);
  const int iheight = grad_in.size(2);
  const int iwidth = grad_in.size(3);

  const int odepth = grad_out.size(1);
  const int oheight = grad_out.size(2);
  const int owidth = grad_out.size(3);

  const int pad_f = pad[4];
  const int pad_t = pad[6];
  const int pad_l = pad[8];
  int iStartX = std::max(0, -pad_l);
  int iStartY = std::max(0, -pad_t);
  int iStartZ = std::max(0, -pad_f);
  int oStartX = std::max(0, pad_l);
  int oStartY = std::max(0, pad_t);
  int oStartZ = std::max(0, pad_f);

  int k, ip_x, ip_y, ip_z;
#pragma omp parallel for private(k, ip_x, ip_y, ip_z)
  for (k = 0; k < nslices; k++) {
    int i, j, z;
    for (z = 0; z < odepth; z++) {
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

          if (z < pad_f) {
            ip_z = pad_f;
          } else if (z >= pad_f && z < idepth + pad_f) {
            ip_z = z;
          } else {
            ip_z = idepth + pad_f - 1;
          }
          ip_z = ip_z - oStartZ + iStartZ;

          DType *src_p = grad_out.dptr_ + k * owidth * oheight * odepth +
                         z * owidth * oheight + i * owidth + j;
          DType *dest_p = grad_in.dptr_ + k * iwidth * iheight * idepth +
                          ip_z * iwidth * iheight + ip_y * iwidth + ip_x;
          *dest_p += *src_p;
        }
      }
    }
  }
}

// Case 2: Zero Padding
template <typename DType>
void single_image_constant(const Tensor<cpu, 4, DType> &dst,
                           const Tensor<cpu, 4, DType> src, mxnet::TShape pad,
                           DType constant_value) {
  const int pad_f = pad[4];
  const int pad_t = pad[6];
  const int pad_l = pad[8];
  int c, d, w, h;
#pragma omp parallel for private(c, d, w, h)
  for (c = 0; c < dst.size(0); ++c) {
    for (d = 0; d < dst.size(1); ++d) {
      for (h = 0; h < dst.size(2); ++h) {
        for (w = 0; w < dst.size(3); ++w) {
          if ((w < pad_l) || (h < pad_t) || (d < pad_f) ||
              (d >= (src.size(1) + pad_f)) || (h >= (src.size(2) + pad_t)) ||
              (w >= (src.size(3) + pad_l))) {
            dst[c][d][h][w] = constant_value;
          } else {
            dst[c][d][h][w] = src[c][d - pad_f][h - pad_t][w - pad_l];
          }
        }
      }
    }
  }
}

template <typename DType>
void single_image_constant_grad(const Tensor<cpu, 4, DType> &in_grad,
                                const Tensor<cpu, 4, DType> out_grad,
                                mxnet::TShape pad) {
  const int pad_f = pad[4];
  const int pad_t = pad[6];
  const int pad_l = pad[8];
  int c, d, w, h;
  #pragma omp parallel for private(c, d, w, h)
  for (c = 0; c < in_grad.size(0); ++c) {
    for (d = 0; d < in_grad.size(1); ++d) {
      for (h = 0; h < in_grad.size(2); ++h) {
        for (w = 0; w < in_grad.size(3); ++w) {
          in_grad[c][d][h][w] += out_grad[c][d + pad_f][h + pad_t][w + pad_l];
        }
      }
    }
  }
}

// Case 3: Reflection Padding
template <typename DType>
void single_image_reflect(const Tensor<cpu, 4, DType> &dst,
                           const Tensor<cpu, 4, DType> src, mxnet::TShape pad) {
  const int nslices = src.size(0);
  const int idepth = src.size(1);
  const int iheight = src.size(2);
  const int iwidth = src.size(3);

  const int odepth = dst.size(1);
  const int oheight = dst.size(2);
  const int owidth = dst.size(3);

  const int pad_f = pad[4];
  const int pad_t = pad[6];
  const int pad_l = pad[8];
  int iStartX = std::max(0, -pad_l);
  int iStartY = std::max(0, -pad_t);
  int iStartZ = std::max(0, -pad_f);
  int oStartX = std::max(0, pad_l);
  int oStartY = std::max(0, pad_t);
  int oStartZ = std::max(0, pad_f);

  int l, ip_x, ip_y, ip_z;
#pragma omp parallel for private(l, ip_x, ip_y, ip_z)
  for (l = 0; l < nslices; l++) {
    int i, j, k;
    for (k = 0; k < odepth; k++) {
      for (i = 0; i < oheight; i++) {
        for (j = 0; j < owidth; j++) {
          if (j < pad_l) {
            ip_x = pad_l * 2 - j;
          } else if (j >= pad_l && j < iwidth + pad_l) {
            ip_x = j;
          } else {
            ip_x = (iwidth + pad_l - 1) * 2 - j;
          }
          ip_x = ip_x - oStartX + iStartX;

          if (i < pad_t) {
            ip_y = pad_t * 2 - i;
          } else if (i >= pad_t && i < iheight + pad_t) {
            ip_y = i;
          } else {
            ip_y = (iheight + pad_t - 1) * 2 - i;
          }
          ip_y = ip_y - oStartY + iStartY;

          if (k < pad_f) {
            ip_z = pad_f * 2 - k;
          } else if (k >= pad_f && k < idepth + pad_f) {
            ip_z = k;
          } else {
            ip_z = (idepth + pad_f - 1) * 2 - k;
          }
          ip_z = ip_z - oStartZ + iStartZ;

          DType *dest_p = dst.dptr_ + l * owidth * oheight * odepth +
                          k * owidth * oheight + i * owidth + j;
          DType *src_p = src.dptr_ + l * iwidth * iheight * idepth +
                         ip_z * iwidth * iheight + ip_y * iwidth + ip_x;
          *dest_p = *src_p;
        }
      }
    }
  }
}

template <typename DType>
void single_image_reflect_grad(const Tensor<cpu, 4, DType> &grad_in,
                                const Tensor<cpu, 4, DType> grad_out,
                                mxnet::TShape pad) {
  const int nslices = grad_in.size(0);
  const int idepth = grad_in.size(1);
  const int iheight = grad_in.size(2);
  const int iwidth = grad_in.size(3);

  const int odepth = grad_out.size(1);
  const int oheight = grad_out.size(2);
  const int owidth = grad_out.size(3);

  const int pad_f = pad[4];
  const int pad_t = pad[6];
  const int pad_l = pad[8];
  int iStartX = std::max(0, -pad_l);
  int iStartY = std::max(0, -pad_t);
  int iStartZ = std::max(0, -pad_f);
  int oStartX = std::max(0, pad_l);
  int oStartY = std::max(0, pad_t);
  int oStartZ = std::max(0, pad_f);

  int l, ip_x, ip_y, ip_z;
/*#pragma omp parallel for private(l, ip_x, ip_y, ip_z)*/
  for (l = 0; l < nslices; l++) {
    int i, j, k;
    for (k = 0; k < odepth; k++) {
      for (i = 0; i < oheight; i++) {
        for (j = 0; j < owidth; j++) {
          if (j < pad_l) {
            ip_x = pad_l * 2 - j;
          } else if (j >= pad_l && j < iwidth + pad_l) {
            ip_x = j;
          } else {
            ip_x = (iwidth + pad_l - 1) * 2 - j;
          }
          ip_x = ip_x - oStartX + iStartX;

          if (i < pad_t) {
            ip_y = pad_t * 2 - i;
          } else if (i >= pad_t && i < iheight + pad_t) {
            ip_y = i;
          } else {
            ip_y = (iheight + pad_t - 1) * 2 - i;
          }
          ip_y = ip_y - oStartY + iStartY;

          if (k < pad_f) {
            ip_z = pad_f * 2 - k;
          } else if (k >= pad_f && k < idepth + pad_f) {
            ip_z = k;
          } else {
            ip_z = (idepth + pad_f - 1) * 2 - k;
          }
          ip_z = ip_z - oStartZ + iStartZ;

          DType *src_p = grad_out.dptr_ + l * owidth * oheight * odepth +
                         k * owidth * oheight + i * owidth + j;
          DType *dest_p = grad_in.dptr_ + l * iwidth * iheight * idepth +
                          ip_z * iwidth * iheight + ip_y * iwidth + ip_x;
          *dest_p += *src_p;
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Interface to 2d and 3d image pad methods

template <int dim, typename DType>
void pad_image(const Tensor<cpu, dim, DType> &dst,
               const Tensor<cpu, dim, DType> src, mxnet::TShape pad, int mode,
               DType constant_value) {
  for (index_t n = 0; n < dst.size(0); ++n) {
    switch (mode) {
      case mxnet::op::pad_enum::kEdge:
        single_image_edge(dst[n], src[n], pad);
        break;
      case mxnet::op::pad_enum::kConstant:
        single_image_constant(dst[n], src[n], pad, constant_value);
        break;
      case mxnet::op::pad_enum::kReflect:
        single_image_reflect(dst[n], src[n], pad);
        break;
    }
  }
}

template <int dim, typename DType>
void pad_image_grad(const Tensor<cpu, dim, DType> &in_grad,
                    const Tensor<cpu, dim, DType> out_grad, mxnet::TShape pad,
                    int mode) {
  for (index_t n = 0; n < in_grad.size(0); ++n) {
    switch (mode) {
      case mxnet::op::pad_enum::kEdge:
        single_image_edge_grad(in_grad[n], out_grad[n], pad);
        break;
      case mxnet::op::pad_enum::kConstant:
        single_image_constant_grad(in_grad[n], out_grad[n], pad);
        break;
      case mxnet::op::pad_enum::kReflect:
        single_image_reflect_grad(in_grad[n], out_grad[n], pad);
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
.describe(R"code(Pads an input array with a constant or edge values of the array.

.. note:: `Pad` is deprecated. Use `pad` instead.

.. note:: Current implementation only supports 4D and 5D input arrays with padding applied
   only on axes 1, 2 and 3. Expects axes 4 and 5 in `pad_width` to be zero.

This operation pads an input array with either a `constant_value` or edge values
along each axis of the input array. The amount of padding is specified by `pad_width`.

`pad_width` is a tuple of integer padding widths for each axis of the format
``(before_1, after_1, ... , before_N, after_N)``. The `pad_width` should be of length ``2*N``
where ``N`` is the number of dimensions of the array.

For dimension ``N`` of the input array, ``before_N`` and ``after_N`` indicates how many values
to add before and after the elements of the array along dimension ``N``.
The widths of the higher two dimensions ``before_1``, ``after_1``, ``before_2``,
``after_2`` must be 0.

Example::

   x = [[[[  1.   2.   3.]
          [  4.   5.   6.]]

         [[  7.   8.   9.]
          [ 10.  11.  12.]]]


        [[[ 11.  12.  13.]
          [ 14.  15.  16.]]

         [[ 17.  18.  19.]
          [ 20.  21.  22.]]]]

   pad(x,mode="edge", pad_width=(0,0,0,0,1,1,1,1)) =

         [[[[  1.   1.   2.   3.   3.]
            [  1.   1.   2.   3.   3.]
            [  4.   4.   5.   6.   6.]
            [  4.   4.   5.   6.   6.]]

           [[  7.   7.   8.   9.   9.]
            [  7.   7.   8.   9.   9.]
            [ 10.  10.  11.  12.  12.]
            [ 10.  10.  11.  12.  12.]]]


          [[[ 11.  11.  12.  13.  13.]
            [ 11.  11.  12.  13.  13.]
            [ 14.  14.  15.  16.  16.]
            [ 14.  14.  15.  16.  16.]]

           [[ 17.  17.  18.  19.  19.]
            [ 17.  17.  18.  19.  19.]
            [ 20.  20.  21.  22.  22.]
            [ 20.  20.  21.  22.  22.]]]]

   pad(x, mode="constant", constant_value=0, pad_width=(0,0,0,0,1,1,1,1)) =

         [[[[  0.   0.   0.   0.   0.]
            [  0.   1.   2.   3.   0.]
            [  0.   4.   5.   6.   0.]
            [  0.   0.   0.   0.   0.]]

           [[  0.   0.   0.   0.   0.]
            [  0.   7.   8.   9.   0.]
            [  0.  10.  11.  12.   0.]
            [  0.   0.   0.   0.   0.]]]


          [[[  0.   0.   0.   0.   0.]
            [  0.  11.  12.  13.   0.]
            [  0.  14.  15.  16.   0.]
            [  0.   0.   0.   0.   0.]]

           [[  0.   0.   0.   0.   0.]
            [  0.  17.  18.  19.   0.]
            [  0.  20.  21.  22.   0.]
            [  0.   0.   0.   0.   0.]]]]


)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "An n-dimensional input array.")
.add_arguments(PadParam::__FIELDS__());

NNVM_REGISTER_OP(Pad).add_alias("pad");

}  // namespace op
}  // namespace mxnet
