/*!
 * Copyright (c) 2016 by Contributors
 * \file warp.cc
 * \brief warp op
 * \author Xu Dong
*/
#include "./warp-inl.h"
#include "./mshadow_op.h"
#include "stdlib.h"

namespace mshadow {
template<typename Dtype>
void WarpForward(
      const Tensor<cpu, 4, Dtype> &data,
      const Tensor<cpu, 4, Dtype> &grid,
      const Tensor<cpu, 4, Dtype> &out) {
  /* assume BHWD */
  int batchsize = data.size(0);
  int inputImages_height = data.size(1);
  int inputImages_width = data.size(2);
  int output_height = out.size(1);
  int output_width = out.size(2);
  int inputImages_channels = data.size(3);

  int output_strideWidth = out.size(3);
  int output_strideHeight = output_strideWidth*out.size(2);
  int output_strideBatch = output_strideHeight*out.size(1);

  int inputImages_strideWidth = data.size(3);
  int inputImages_strideHeight = inputImages_strideWidth*data.size(2);
  int inputImages_strideBatch = inputImages_strideHeight*data.size(1);

  int grids_strideWidth = grid.size(3);
  int grids_strideHeight = grids_strideWidth*grid.size(2);
  int grids_strideBatch = grids_strideHeight*grid.size(1);

  Dtype *inputImages_data, *output_data, *grids_data;
  inputImages_data = data.dptr_;
  output_data = out.dptr_;
  grids_data = grid.dptr_;

  int b, yOut, xOut;

  for (b=0; b < batchsize; b++) {
    for (yOut=0; yOut < output_height; yOut++) {
      for (xOut=0; xOut < output_width; xOut++) {
        // read the grid
        double xf = grids_data[b*grids_strideBatch +
         yOut*grids_strideHeight + xOut*grids_strideWidth];
        double yf = grids_data[b*grids_strideBatch +
         yOut*grids_strideHeight + xOut*grids_strideWidth + 1];

        // get the weights for interpolation
        int yInTopLeft, xInTopLeft;
        double yWeightTopLeft, xWeightTopLeft;

        double xcoord = xf + xOut;
        xInTopLeft = floor(xcoord);
        xWeightTopLeft = std::max(1 - fabs(xcoord - xInTopLeft), 0.0);

        double ycoord = yf + yOut;
        yInTopLeft = floor(ycoord);
        yWeightTopLeft = std::max(1 - fabs(ycoord - yInTopLeft), 0.0);

        int outAddress = output_strideBatch * b + output_strideHeight *
         yOut + output_strideWidth * xOut;
        int inTopLeftAddress = inputImages_strideBatch * b +
         inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
        int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
        int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
        int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

        double v = 0;
        double inTopLeft = 0;
        double inTopRight = 0;
        double inBottomLeft = 0;
        double inBottomRight = 0;

        // we are careful with the boundaries
        bool topLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 &&
         yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool topRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 &&
         yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool bottomLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 &&
         yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;
        bool bottomRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 &&
         yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;

        int t;
        // interpolation happens here
        for (t=0; t < inputImages_channels; t++) {
           if (topLeftIsIn) inTopLeft = inputImages_data[inTopLeftAddress + t];
           if (topRightIsIn) inTopRight = inputImages_data[inTopRightAddress + t];
           if (bottomLeftIsIn) inBottomLeft = inputImages_data[inBottomLeftAddress + t];
           if (bottomRightIsIn) inBottomRight = inputImages_data[inBottomRightAddress + t];

           v = xWeightTopLeft * yWeightTopLeft * inTopLeft
             + (1 - xWeightTopLeft) * yWeightTopLeft * inTopRight
             + xWeightTopLeft * (1 - yWeightTopLeft) * inBottomLeft
             + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * inBottomRight;

           output_data[outAddress + t] = v;
        }
      }
    }
  }
}

template<typename Dtype>
void WarpBackward(const Tensor<cpu, 4, Dtype> &grad_data,
                  const Tensor<cpu, 4, Dtype> &grad_grid,
                  const Tensor<cpu, 4, Dtype> &out_grad,
                  const Tensor<cpu, 4, Dtype> &data,
                  const Tensor<cpu, 4, Dtype> &grid,
                  bool onlyGrid) {
  /* assume BHWD */
  int batchsize = data.size(0);
  int inputImages_height = data.size(1);
  int inputImages_width = data.size(2);
  int gradOutput_height = out_grad.size(1);
  int gradOutput_width = out_grad.size(2);
  int inputImages_channels = data.size(3);

  int gradOutput_strideWidth = out_grad.size(3);
  int gradOutput_strideHeight = gradOutput_strideWidth*out_grad.size(2);
  int gradOutput_strideBatch = gradOutput_strideHeight*out_grad.size(1);

  int inputImages_strideWidth = data.size(3);
  int inputImages_strideHeight = inputImages_strideWidth*data.size(2);
  int inputImages_strideBatch = inputImages_strideHeight*data.size(1);

  int grids_strideWidth = grid.size(3);
  int grids_strideHeight = grids_strideWidth*grid.size(2);
  int grids_strideBatch = grids_strideHeight*grid.size(1);

  int gradInputImages_strideWidth = grad_data.size(3);
  int gradInputImages_strideHeight = gradInputImages_strideWidth*grad_data.size(2);
  int gradInputImages_strideBatch = gradInputImages_strideHeight*grad_data.size(1);

  int gradGrids_strideWidth = grad_grid.size(3);
  int gradGrids_strideHeight = gradGrids_strideWidth*grad_grid.size(2);
  int gradGrids_strideBatch = gradGrids_strideHeight*grad_grid.size(1);

  Dtype *inputImages_data, *gradOutput_data, *grids_data, *gradGrids_data, *gradInputImages_data;
  inputImages_data = data.dptr_;
  gradOutput_data = out_grad.dptr_;
  grids_data = grid.dptr_;
  gradGrids_data = grad_grid.dptr_;
  gradInputImages_data = grad_data.dptr_;

  int b, yOut, xOut;

  for (b=0; b < batchsize; b++) {
    for (yOut=0; yOut < gradOutput_height; yOut++) {
      for (xOut=0; xOut < gradOutput_width; xOut++) {
        // read the grid
        double xf = grids_data[b*grids_strideBatch +
          yOut*grids_strideHeight + xOut*grids_strideWidth];
        double yf = grids_data[b*grids_strideBatch +
          yOut*grids_strideHeight + xOut*grids_strideWidth + 1];

        // get the weights for interpolation
        int yInTopLeft, xInTopLeft;
        double yWeightTopLeft, xWeightTopLeft;

        double xcoord = xf + xOut;
        xInTopLeft = floor(xcoord);
        xWeightTopLeft = std::max(1 - fabs(xcoord - xInTopLeft), 0.0);

        double ycoord = yf + yOut;
        yInTopLeft = floor(ycoord);
        yWeightTopLeft = std::max(1 - fabs(ycoord - yInTopLeft), 0.0);

        const int inTopLeftAddress = inputImages_strideBatch * b +
         inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
        const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
        const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
        const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

        const int gradInputImagesTopLeftAddress = gradInputImages_strideBatch * b +
         gradInputImages_strideHeight * yInTopLeft + gradInputImages_strideWidth * xInTopLeft;
        const int gradInputImagesTopRightAddress =
         gradInputImagesTopLeftAddress + gradInputImages_strideWidth;
        const int gradInputImagesBottomLeftAddress =
         gradInputImagesTopLeftAddress + gradInputImages_strideHeight;
        const int gradInputImagesBottomRightAddress =
         gradInputImagesBottomLeftAddress + gradInputImages_strideWidth;

        const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideHeight *
         yOut + gradOutput_strideWidth * xOut;

        double topLeftDotProduct = 0;
        double topRightDotProduct = 0;
        double bottomLeftDotProduct = 0;
        double bottomRightDotProduct = 0;

        // we are careful with the boundaries
        bool topLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 &&
         yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool topRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 &&
         yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool bottomLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 &&
         yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;
        bool bottomRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 &&
         yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;

        int t;

        for (t=0; t < inputImages_channels; t++) {
           double gradOutValue = gradOutput_data[gradOutputAddress + t];
           if (topLeftIsIn) {
              double inTopLeft = inputImages_data[inTopLeftAddress + t];
              topLeftDotProduct += inTopLeft * gradOutValue;
              if (!onlyGrid) gradInputImages_data[gradInputImagesTopLeftAddress + t] +=
               xWeightTopLeft * yWeightTopLeft * gradOutValue;
           }

           if (topRightIsIn) {
              double inTopRight = inputImages_data[inTopRightAddress + t];
              topRightDotProduct += inTopRight * gradOutValue;
              if (!onlyGrid) gradInputImages_data[gradInputImagesTopRightAddress + t] +=
               (1 - xWeightTopLeft) * yWeightTopLeft * gradOutValue;
           }

           if (bottomLeftIsIn) {
              double inBottomLeft = inputImages_data[inBottomLeftAddress + t];
              bottomLeftDotProduct += inBottomLeft * gradOutValue;
              if (!onlyGrid) gradInputImages_data[gradInputImagesBottomLeftAddress + t] +=
               xWeightTopLeft * (1 - yWeightTopLeft) * gradOutValue;
           }

           if (bottomRightIsIn) {
              double inBottomRight = inputImages_data[inBottomRightAddress + t];
              bottomRightDotProduct += inBottomRight * gradOutValue;
              if (!onlyGrid) gradInputImages_data[gradInputImagesBottomRightAddress + t] +=
               (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * gradOutValue;
           }
        }

        yf = - xWeightTopLeft * topLeftDotProduct + xWeightTopLeft * bottomLeftDotProduct -
         (1-xWeightTopLeft) * topRightDotProduct + (1-xWeightTopLeft) * bottomRightDotProduct;
        xf = - yWeightTopLeft * topLeftDotProduct + yWeightTopLeft * topRightDotProduct -
         (1-yWeightTopLeft) * bottomLeftDotProduct + (1-yWeightTopLeft) * bottomRightDotProduct;

        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight +
         xOut*gradGrids_strideWidth + 1] = yf;
        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight +
         xOut*gradGrids_strideWidth] = xf;
      }
    }
  }
}
}  // namespace mshadow
namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(WarpParam param) {
  return new WarpOp<cpu>(param);
}

Operator* WarpProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(WarpParam);
MXNET_REGISTER_OP_PROPERTY(Warp, WarpProp)
.describe("According to grid, this operator warp data via bilinear interpolation. "
"Grid has two channels : displacements of x-axis and y-axis. "
"We assume [batch, y, x, channel] format on both input and output. \n"
"output[batch, i, j, channel] = G(data[batch, i+y_displacement, j+x_displacement, channel]) \n"
"i, j enumerate all spatial locations in data. \n"
"G() denotes the bilinear interpolation kernel")
.add_argument("data", "Symbol", "Input data to the WarpOp")
.add_argument("grid", "Symbol", "Input grid to the WarpOp")
.add_arguments(WarpParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
