/*!
 * Copyright (c) 2017 by Contributors
 * \file bilinear_sampler.cc
 * \brief
 * \author Xu Dong
*/

#include "./bilinear_sampler-inl.h"

namespace mshadow {
template<typename DType>
bool between(DType value, int lowerBound, int upperBound) {
  return (value >= lowerBound && value <= upperBound);
}
template<typename DType>
inline void BilinearSamplerForward(const Tensor<cpu, 4, DType> &output,
                                    const Tensor<cpu, 4, DType> &input,
                                    const Tensor<cpu, 4, DType> &grid_src) {
  DType *out = output.dptr_;
  const DType *data = input.dptr_;
  const DType *grid = grid_src.dptr_;
  int o_n = output.size(0), o_c = output.size(1), o_h = output.size(2), o_w = output.size(3);
  int i_c = input.size(1), i_h = input.size(2), i_w = input.size(3);
  for (index_t n = 0; n < o_n; ++n) {
    for (index_t c = 0; c < o_c; ++c) {
      for (index_t h = 0; h < o_h; ++h) {
        for (index_t w = 0; w < o_w; ++w) {
          index_t out_index = n * o_c * o_h * o_w + c * o_h * o_w + h * o_w + w;
          index_t grid_index = n * o_h * o_w * 2 + h * o_w + w;
          DType y_real = (*(grid + grid_index + o_h * o_w) + 1) * (i_h - 1) / 2;
          DType x_real = (*(grid + grid_index) + 1) * (i_w - 1) / 2;
          int top_left_y = static_cast<int>(floor(y_real));
          int top_left_x = static_cast<int>(floor(x_real));
          DType top_left_y_w = 1.0 - (y_real - top_left_y);
          DType top_left_x_w = 1.0 - (x_real - top_left_x);
          int data_index = n * i_c * i_h * i_w + c * i_h * i_w +
            top_left_y * i_w + top_left_x;
          DType top_left_v = 0;
          DType top_right_v = 0;
          DType bottom_left_v = 0;
          DType bottom_right_v = 0;
          if (between(top_left_x, 0, i_w-1) && between(top_left_y, 0, i_h-1))
            top_left_v = *(data + data_index);
          if (between(top_left_x + 1, 0, i_w-1) && between(top_left_y, 0, i_h-1))
            top_right_v = *(data + data_index + 1);
          if (between(top_left_x, 0, i_w-1) && between(top_left_y + 1, 0, i_h-1))
            bottom_left_v = *(data + data_index + i_w);
          if (between(top_left_x+1, 0, i_w-1) && between(top_left_y + 1, 0, i_h-1))
            bottom_right_v = *(data + data_index + i_w + 1);
          *(out+out_index) = top_left_v * top_left_y_w * top_left_x_w +
                              top_right_v * top_left_y_w * (1.0 - top_left_x_w) +
                              bottom_left_v * (1.0 - top_left_y_w) * top_left_x_w +
                              bottom_right_v * (1.0 - top_left_y_w) * (1.0 - top_left_x_w);
        }
      }
    }
  }
}

template<typename DType>
inline void BilinearSamplerBackward(const Tensor<cpu, 4, DType> &gdata,
                                     const Tensor<cpu, 4, DType> &ggrid,
                                     const Tensor<cpu, 4, DType> &output_grad,
                                     const Tensor<cpu, 4, DType> &input_data,
                                     const Tensor<cpu, 4, DType> &grid) {
  DType *g_input = gdata.dptr_;
  DType *grad_grid = ggrid.dptr_;
  const DType *grid_src = grid.dptr_;
  const DType *grad = output_grad.dptr_;
  const DType *data = input_data.dptr_;
  int o_n = output_grad.size(0), o_c = output_grad.size(1),
      o_h = output_grad.size(2), o_w = output_grad.size(3);
  int i_c = input_data.size(1), i_h = input_data.size(2), i_w = input_data.size(3);
  for (index_t n = 0; n < o_n; ++n) {
     for (index_t h = 0; h < o_h; ++h) {
        for (index_t w = 0; w < o_w; ++w) {
          DType top_left_y_gw = 0.0;
          DType top_left_x_gw = 0.0;
          index_t grid_src_index = n * o_h * o_w * 2 + h * o_w + w;
          DType y_real = (*(grid_src + grid_src_index + o_h * o_w) + 1) * (i_h - 1) / 2;
          DType x_real = (*(grid_src + grid_src_index) + 1) * (i_w - 1) / 2;
          int top_left_y = static_cast<int>(floor(y_real));
          int top_left_x = static_cast<int>(floor(x_real));
          DType top_left_y_w = 1.0 - (y_real - top_left_y);
          DType top_left_x_w = 1.0 - (x_real - top_left_x);
          for (index_t c = 0; c < o_c; ++c) {
            index_t grad_index = n * o_c * o_h * o_w + c * o_h * o_w + h * o_w + w;
            int data_index = n * i_c * i_h * i_w + c * i_h * i_w + top_left_y * i_w
                                  + top_left_x;
            // calc 4 vertex value in input data
            DType top_left_v = 0;
            DType top_right_v = 0;
            DType bottom_left_v = 0;
            DType bottom_right_v = 0;
            // calc input grad
            if (between(top_left_x, 0, i_w-1) && between(top_left_y, 0, i_h-1)) {
              *(g_input + data_index) += *(grad + grad_index) * top_left_y_w * top_left_x_w;
              top_left_v = *(data + data_index);
            }
            if (between(top_left_x+1, 0, i_w-1) && between(top_left_y, 0, i_h-1)) {
              *(g_input + data_index + 1) += *(grad + grad_index) * top_left_y_w
                                              * (1.0 - top_left_x_w);
              top_right_v = *(data + data_index + 1);
            }
            if (between(top_left_x, 0, i_w-1) && between(top_left_y+1, 0, i_h-1)) {
              *(g_input + data_index+ i_w) += *(grad + grad_index) * (1.0 - top_left_y_w)
                                              * top_left_x_w;
              bottom_left_v = *(data + data_index + i_w);
            }
            if (between(top_left_x+1, 0, i_w-1) && between(top_left_y+1, 0, i_h-1)) {
              *(g_input + data_index+ i_w + 1) += *(grad + grad_index) * (1.0 - top_left_y_w)
                                                  * (1.0 - top_left_x_w);
              bottom_right_v = *(data + data_index + i_w + 1);
            }
            // calc weight grad of top_left_w, then multiple -1 is the grad of grid_src
            top_left_y_gw -= *(grad + grad_index) * (top_right_v - bottom_right_v +
                              (top_left_v - top_right_v - bottom_left_v + bottom_right_v)
                              * top_left_x_w);
            top_left_x_gw -= *(grad + grad_index) * (bottom_left_v - bottom_right_v +
                              (top_left_v - top_right_v - bottom_left_v + bottom_right_v)
                              * top_left_y_w);
          }
          // calc grad of grid
          *(grad_grid + grid_src_index + o_h * o_w) += top_left_y_gw * (i_h - 1) / 2;
          *(grad_grid + grid_src_index) += top_left_x_gw * (i_w - 1) / 2;
        }
      }
    }
  }
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(BilinearSamplerParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new BilinearSamplerOp<cpu, DType>(param);
  })
  return op;
}

Operator *BilinearSamplerProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(BilinearSamplerParam);

MXNET_REGISTER_OP_PROPERTY(BilinearSampler, BilinearSamplerProp)
.add_argument("data", "NDArray-or-Symbol", "Input data to the BilinearsamplerOp.")
.add_argument("grid", "NDArray-or-Symbol", "Input grid to the BilinearsamplerOp."
                                "grid has two channels: x_src, y_src")
.add_arguments(BilinearSamplerParam::__FIELDS__())
.describe("Applies bilinear sampling to input feature map,"
" which is the key of \"[NIPS2015] Spatial Transformer Networks\"\n    "
"output[batch, channel, y_dst, x_dst] = G(data[batch, channel, y_src, x_src)\n    "
"x_dst, y_dst enumerate all spatial locations in output\n    "
"x_src = grid[batch, 0, y_dst, x_dst]\n    "
"y_src = grid[batch, 1, y_dst, x_dst]\n    "
"G() denotes the bilinear interpolation kernel\n"
"The out-boundary points will be padded as zeros. (The boundary is defined to be [-1, 1])\n"
"The shape of output will be (data.shape[0], data.shape[1], grid.shape[2], grid.shape[3])\n"
"The operator assumes that grid has been nomalized. "
"If you want to design a CustomOp to manipulate grid, "
"please refer to GridGeneratorOp.");
}  // namespace op
}  // namespace mxnet
