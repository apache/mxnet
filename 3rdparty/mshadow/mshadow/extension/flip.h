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
 * \file flip.h
 * \brief support for flip a certain dimension.
 * \author Junyuan Xie
 */
#ifndef MSHADOW_EXTENSION_FLIP_H_
#define MSHADOW_EXTENSION_FLIP_H_

#include "../extension.h"

namespace mshadow {
namespace expr {
/*!
 * \brief slice expression, slice a tensor's channel
 * \tparam SrcExp left expression
 * \tparam DType the type of elements
 * \tparam srcdim dimension of src
 * \tparam dimsrc_m_cat dimsrc - dimcat
 */
template<typename SrcExp, typename Device,
         typename DType, int srcdim>
struct FlipExp : public TRValue<FlipExp<SrcExp,
                                        Device, DType,
                                        srcdim>,
                                Device, srcdim, DType> {
  const SrcExp &src_;
  index_t trailing_;
  index_t stride_;
  index_t stride_j_;
  Shape<srcdim> shape_;
  FlipExp(const SrcExp &src, int dim)
      : src_(src) {
    shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
    stride_ = shape_[dim];
    stride_j_ = shape_[srcdim-1];
    trailing_ = 1;
    for (int i = dim + 1; i < srcdim; ++i) {
      trailing_ *= shape_[i];
    }
  }
  template<typename E, int etype>
  inline void
  operator=(const expr::Exp<E, DType, etype> &exp) {
    this->__assign(exp);
  }
  inline void
  operator=(const DType &exp) {
    this->__assign(exp);
  }
};  // struct Flip

/*!
 * \brief Flip a Tensor
 * \param src source tensor
 * \param begin The beginning slice.
 * \param end The end slice.
 * \return sliced tensor
 * \tparam sdim the dimension to slice on
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam etype type of expression
 */
template<typename SrcExp, typename Device,
         typename DType, int srcdim>
inline FlipExp<SrcExp, Device, DType, srcdim>
flip(const TRValue<SrcExp, Device, srcdim, DType> &src, int dim) {
  return FlipExp<SrcExp, Device, DType, srcdim>(src.self(), dim);
}
//------------------------
//  engine plugin
//------------------------
// runtime shapecheck
template<typename SrcExp, typename Device,
         typename DType, int srcdim>
struct ShapeCheck<srcdim, FlipExp<SrcExp, Device, DType, srcdim> >{
  inline static Shape<srcdim> Check(const FlipExp<SrcExp,
                                    Device, DType, srcdim> &t) {
    return t.shape_;
  }
};
template<typename SrcExp, typename Device,
         typename DType, int srcdim>
struct StreamInfo<Device, FlipExp<SrcExp, Device, DType, srcdim> >{
  inline static Stream<Device> *
  Get(const FlipExp<SrcExp, Device, DType, srcdim> &t) {
    return StreamInfo<Device, SrcExp>::Get(t.src_);
  }
};
// static typecheck
template<typename SrcExp, typename Device,
         typename DType, int srcdim>
struct ExpInfo<FlipExp<SrcExp, Device, DType, srcdim> >{
  static const int kDim = ExpInfo<SrcExp>::kDim;
  static const int kDevMask = ExpInfo<SrcExp>::kDevMask;
};
//----------------------
// Execution plan
//---------------------
template<typename SrcExp, typename Device,
         typename DType, int srcdim>
struct Plan<FlipExp<SrcExp, Device, DType, srcdim>, DType> {
 public:
  explicit Plan(const FlipExp<SrcExp, Device, DType, srcdim> &e)
      : src_(MakePlan(e.src_)), stride_j_(e.stride_j_),
        trailing_(e.trailing_), stride_(e.stride_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    index_t idx = i*stride_j_+j;
    const index_t low = idx%trailing_;
    index_t high = idx/trailing_;
    const index_t x = high%stride_;
    high /= stride_;
    idx = (high*stride_+stride_-1-x)*trailing_+low;
    return src_.Eval(idx/stride_j_, idx%stride_j_);
  }
  MSHADOW_XINLINE DType &REval(index_t i, index_t j) const {
    index_t idx = i*stride_j_+j;
    const index_t low = idx%trailing_;
    index_t high = idx/trailing_;
    const index_t x = high%stride_;
    high /= stride_;
    idx = (high*stride_+stride_-1-x)*trailing_+low;
    return src_.REval(idx/stride_j_, idx%stride_j_);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t stride_j_, trailing_, stride_;
};  // struct Plan
}  // namespace expr
}   // namespace mshadow
#endif  // MSHADOW_EXTENSION_FLIP_H_
