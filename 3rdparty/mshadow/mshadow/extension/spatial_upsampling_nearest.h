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
 * \file spatial_upsampling.h
 * \brief
 * \author Bing Xu
*/
#ifndef MSHADOW_EXTENSION_SPATIAL_UPSAMPLING_NEAREST_H_
#define MSHADOW_EXTENSION_SPATIAL_UPSAMPLING_NEAREST_H_
#include "../extension.h"

namespace mshadow {
namespace expr {

/*! \brief nearest neighboor upsampling
 *         out(x, y) = in(int(x / scale_x), int(y / scale_y))
 *  \tparam SrcExp source expression
 *  \tparam DType data type
 *  \tparam srcdim source dimension
 */
template<typename SrcExp, typename DType, int srcdim>
struct UpSamplingNearestExp :
  public MakeTensorExp<UpSamplingNearestExp<SrcExp, DType, srcdim>,
                       SrcExp, srcdim, DType> {
  /*! \brief source oprand */
  const SrcExp &src_;
  /*! \brief up sampling scale */
  index_t scale_;
  /*! \brief constructor */
  UpSamplingNearestExp(const SrcExp &src, index_t scale)
    : src_(src), scale_(scale) {
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
    this->shape_[srcdim - 2] *= scale_;
    this->shape_[srcdim - 1] *= scale_;
  }
};


template<typename SrcExp, typename DType, int etype>
inline UpSamplingNearestExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
upsampling_nearest(const Exp<SrcExp, DType, etype> &src, index_t scale) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>
    ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return UpSamplingNearestExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), scale);
}

template<typename SrcExp, typename DType, int srcdim>
struct Plan<UpSamplingNearestExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const UpSamplingNearestExp<SrcExp, DType, srcdim> &e)
    : src_(MakePlan(e.src_)),
      scale_(e.scale_),
      new_height_(e.shape_[srcdim - 2]),
      src_height_(static_cast<index_t>(e.shape_[srcdim - 2] / e.scale_)) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t x = j;
    const index_t y = i % new_height_;
    const index_t c = i / new_height_;
    const index_t h = static_cast<index_t>(y / scale_);
    const index_t w = static_cast<index_t>(x / scale_);
    return src_.Eval(c * src_height_ + h, w);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t scale_;
  const index_t new_height_;
  const index_t src_height_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_SPATIAL_UPSAMPLING_NEAREST_H_
