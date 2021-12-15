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
 * \file crop.h
 * \brief support for crop
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_CROP_H_
#define MSHADOW_EXTENSION_CROP_H_
#include "../extension.h"
namespace mshadow {
namespace expr {
/*!
 * \brief crop expression, cut off the boundary region, reverse operation of padding
 * \tparam SrcExp source expression to be pooled from
 * \tparam DType the type of elements
 * \tparam srcdim dimension of src
 */
template<typename SrcExp, typename DType, int srcdim>
struct CroppingExp:
      public MakeTensorExp<CroppingExp<SrcExp, DType, srcdim>,
                           SrcExp, srcdim, DType> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief pad height */
  index_t pad_height_;
  /*! \brief pad height */
  index_t pad_width_;
  /*! \brief src height */
  index_t src_height_;
  /*! \brief constructor */
  explicit CroppingExp(const SrcExp &src, Shape<2> cshape)
      : src_(src) {
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
    CHECK_GE(this->shape_[srcdim - 2], cshape[0]) << "CroppingExp: height requirement not met";
    CHECK_GE(this->shape_[srcdim - 1], cshape[1]) << "CroppingExp: width requirement not met";
    pad_height_ = (this->shape_[srcdim - 2] - cshape[0]) / 2;
    pad_width_ = (this->shape_[srcdim - 1] - cshape[1]) / 2;
    src_height_ = this->shape_[srcdim - 2];
    this->shape_[srcdim - 2] = cshape[0];  // height
    this->shape_[srcdim - 1] = cshape[1];  // width
  }
  /*! \brief constructor */
  explicit CroppingExp(const SrcExp &src, Shape<2> cshape,
                       index_t start_height, index_t start_width)
      : src_(src), pad_height_(start_height), pad_width_(start_width) {
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
    CHECK_GE(this->shape_[srcdim - 2], cshape[0] + start_height)
      << "CroppingExp: height requirement not met";
    CHECK_GE(this->shape_[srcdim - 1], cshape[1] + start_width)
      << "CroppingExp: width requirement not met";
    src_height_ = this->shape_[srcdim - 2];
    this->shape_[srcdim - 2] = cshape[0];  // height
    this->shape_[srcdim - 1] = cshape[1];  // width
  }
};  // struct CroppingExp
/*!
 * \brief revserse operationg of padding, cut off boundaries,
 *   crop output from center of input
 * \param src original image batches
 * \param oshape output shape to be cropped
 * \return expression corresponding to padded result
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam etype type of expression
 */
template<typename SrcExp, typename DType, int etype>
inline CroppingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
crop(const Exp<SrcExp, DType, etype> &src, Shape<2> oshape) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return CroppingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), oshape);
}
/*!
 * \brief same as crop, but can specify starting position to do cropping
 * \param src original image batches
 * \param oshape output shape to be cropped
 * \param start_height start height position to do cropping
 * \param start_width  start width position to do cropping
 * \return expression corresponding to padded result
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam etype type of expression
 */
template<typename SrcExp, typename DType, int etype>
inline CroppingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
crop(const Exp<SrcExp, DType, etype> &src, Shape<2> oshape,
     index_t start_height, index_t start_width) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return CroppingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
      (src.self(), oshape, start_height, start_width);
}
//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int srcdim>
struct Plan<CroppingExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const CroppingExp<SrcExp, DType, srcdim> &e)
      : src_(MakePlan(e.src_)),
        pad_height_(e.pad_height_), pad_width_(e.pad_width_),
        new_height_(e.shape_[srcdim - 2]), src_height_(e.src_height_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t x = j;
    const index_t y = i % new_height_;
    const index_t c = i / new_height_;
    const index_t h = y + pad_height_;
    const index_t w = x + pad_width_;
    return src_.Eval(c * src_height_ + h, w);
  }
 private:
  Plan<SrcExp, DType> src_;
  const index_t pad_height_, pad_width_;
  const index_t new_height_;
  const index_t src_height_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_CROP_H_
