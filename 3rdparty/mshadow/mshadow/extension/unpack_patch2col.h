/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file unpack_patch2col.h
 * \brief support for unpack
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_UNPACK_PATCH2COL_H_
#define MSHADOW_EXTENSION_UNPACK_PATCH2COL_H_
#include "../extension.h"
namespace mshadow {
namespace expr {
/*!
 * \brief unpack local (overlap) patches of image to column of mat,
 *  can be used to implement convolution, this expression allow unpack of a batch
 *  this is a version support unpacking multiple images
 *  after getting unpacked mat, we can use: output = dot(weight, mat) to get covolved results, the relations:
 * \tparam SrcExp source expression
 * \tparam dstdim destination dimension
 */
template<typename SrcExp, typename DType, int srcdim>
struct UnpackPatchToColXExp:
      public MakeTensorExp<UnpackPatchToColXExp<SrcExp, DType, srcdim>,
                           SrcExp, 2, DType>{
  /*! \brief source operand */
  const SrcExp &img_;
  /*! \brief patch height */
  index_t psize_y_;
  /*! \brief patch width */
  index_t psize_x_;
  /*! \brief patch stride */
  index_t pstride_y_;
  index_t pstride_x_;
  /*! \brief patch dilate */
  index_t pdilate_y_;
  index_t pdilate_x_;
  /*! \brief number of input channel */
  index_t i_channel_;
  /*! \brief height of img */
  index_t i_height_;
  /*! \brief width of img */
  index_t i_width_;
  /*! \brief constructor */
  UnpackPatchToColXExp(const SrcExp &img,
                       index_t psize_y,
                       index_t psize_x,
                       index_t pstride_y,
                       index_t pstride_x,
                       index_t pdilate_y,
                       index_t pdilate_x)
      : img_(img), psize_y_(psize_y), psize_x_(psize_x),
      pstride_y_(pstride_y), pstride_x_(pstride_x),
      pdilate_y_(pdilate_y), pdilate_x_(pdilate_x){
    Shape<srcdim> imshape = ShapeCheck<srcdim, SrcExp>::Check(img_);
    CHECK(imshape[srcdim - 1] >= psize_x && imshape[srcdim - 2] >= psize_y)
      << "UnpackPatchToCol:image shape smaller than patch size";
    this->i_channel_ = imshape[srcdim - 3];
    this->i_height_  = imshape[srcdim - 2];
    this->i_width_   = imshape[srcdim - 1];
    // calculate number of batches
    const index_t num = imshape.ProdShape(0, srcdim - 3);
    const index_t o_height = (i_height_ -
        (pdilate_y * (psize_y - 1) + 1)) / pstride_y + 1;
    const index_t o_width  = (i_width_  -
        (pdilate_x * (psize_x - 1) + 1)) / pstride_x + 1;
    this->shape_[1] = o_height * o_width * num;
    this->shape_[0] = psize_y * psize_x * i_channel_;
  }
};

/*!
 * \brief  unpack local (overlap) patches of image to column of mat, can be used to implement convolution
 *  after getting unpacked mat, we can use: output = dot(weight, mat) to get covolved results, the relations:
 *
 *  weight; shape[0]: out_channel, shape[1]: ichannel * psize_y * psize_x
 *  output; shape[0]: out_channel, shape[1]: out_height * out_width * num_of_images
 *  out_height = (in_height - psize_y) / pstride + 1, this means we pad inperfect patch with 0
 *  out_width  = (in_width - psize_x) / pstride + 1
 *
 * \return mat target matrix; shape[0]: in_channel*psize_y*psize_x  shape[1]: out_height*out_width * num_of_images
 * \param img source image; shape[-3]: in_channels, shape[-2]: in_height, shape[-1]: in_width, can be 3D or 4D tensor(multiple images)
 * \param psize_y height of each patch
 * \param psize_x width of each patch
 * \param pstride stride of each patch
 * \param pdilate dilate of each patch
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam etype type of expression
 */
template<typename SrcExp, typename DType, int etype>
inline UnpackPatchToColXExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
unpack_patch2col(const Exp<SrcExp, DType, etype> &img,
                 index_t psize_y, index_t psize_x, index_t pstride, index_t pdilate) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 3>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return UnpackPatchToColXExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
      (img.self(), psize_y, psize_x, pstride, pstride, pdilate, pdilate);
}

/*!
 *if you want to specify stride_x and stride_y
 */
template<typename SrcExp, typename DType, int etype>
inline UnpackPatchToColXExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
unpack_patch2col(const Exp<SrcExp, DType, etype> &img,
                 index_t psize_y, index_t psize_x, index_t pstride_y_, index_t pstride_x_,
                 index_t pdilate_y_, index_t pdilate_x_) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 3>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return UnpackPatchToColXExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
      (img.self(), psize_y, psize_x, pstride_y_, pstride_x_, pdilate_y_, pdilate_x_);
}
//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int srcdim>
struct Plan<UnpackPatchToColXExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const UnpackPatchToColXExp<SrcExp, DType, srcdim> &e)
      :src_(MakePlan(e.img_)),
       psize_y_(e.psize_y_), psize_x_(e.psize_x_),
       pstride_y_(e.pstride_y_), pstride_x_(e.pstride_x_),
       i_channel_(e.i_channel_), pdilate_y_(e.pdilate_y_), pdilate_x_(e.pdilate_x_),
       i_height_(e.i_height_), i_width_(e.i_width_),
       o_height_((i_height_ - (pdilate_y_ * (psize_y_ - 1) + 1)) / pstride_y_ + 1),
       o_width_((i_width_ - (pdilate_x_ * (psize_x_ - 1) + 1)) / pstride_x_ + 1) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t x_offset = i % psize_x_ * pdilate_x_;
    const index_t idivp    = i / psize_x_;
    const index_t y_offset = idivp % psize_y_ * pdilate_y_;
    const index_t c = idivp / psize_y_;
    const index_t x = (j % o_width_) * pstride_x_ + x_offset;
    const index_t jdivw = j / o_width_;
    const index_t y = (jdivw % o_height_) * pstride_y_ + y_offset;
    const index_t n = jdivw / o_height_;

    if (x < i_width_ && y < i_height_) {
      return src_.Eval((n * i_channel_  + c) * i_height_ + y, x);
    } else {
      return DType(0.0f);
    }
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t psize_y_, psize_x_, pstride_y_, pstride_x_, i_channel_;
  const index_t pdilate_y_, pdilate_x_;
  const index_t i_height_, i_width_, o_height_, o_width_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_UNPACK_PATCH2COL_H_
