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
 * \file slice.h
 * \brief support for slice a certain dimension.
 */
#ifndef MSHADOW_EXTENSION_SLICE_H_
#define MSHADOW_EXTENSION_SLICE_H_

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
template<typename SrcExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_slice>
struct SliceExp : public TRValue<SliceExp<SrcExp,
                                          Device, DType,
                                          srcdim, dimsrc_m_slice>,
                                 Device, srcdim, DType> {
  static const int dimslice = srcdim - dimsrc_m_slice;
  const SrcExp &src_;
  index_t ch_begin_;
  index_t ch_old_;
  Shape<srcdim> shape_;
  SliceExp(const SrcExp &src, index_t begin, index_t end)
      : src_(src), ch_begin_(begin) {
    shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
    ch_old_ = shape_[dimslice];
    CHECK(begin <= shape_[dimslice] && end <= shape_[dimslice])
        << "The slice went out of range. ";
    shape_[dimslice] = end - begin;
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
};  // struct Slice

/*!
 * \brief Slice a Tensor
 * \param src source tensor
 * \param begin The beginning slice.
 * \param end The end slice.
 * \return sliced tensor
 * \tparam sdim the dimension to slice on
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam etype type of expression
 */
template<int sdim, typename SrcExp,
         typename Device, typename DType, int srcdim>
inline SliceExp<SrcExp, Device, DType, srcdim, srcdim - sdim>
slice(const TRValue<SrcExp, Device, srcdim, DType> &src, index_t begin, index_t end) {
  TypeCheckPass<sdim < srcdim && ExpInfo<SrcExp>::kDim == srcdim>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return SliceExp<SrcExp, Device, DType, srcdim, srcdim - sdim>(src.self(), begin, end);
}
//------------------------
//  engine plugin
//------------------------
// runtime shapecheck
template<typename SrcExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_slice>
struct ShapeCheck<srcdim, SliceExp<SrcExp, Device, DType, srcdim, dimsrc_m_slice> >{
  inline static Shape<srcdim> Check(const SliceExp<SrcExp,
                                    Device, DType, srcdim, dimsrc_m_slice> &t) {
    return t.shape_;
  }
};
template<typename SrcExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_slice>
struct StreamInfo<Device, SliceExp<SrcExp, Device, DType, srcdim, dimsrc_m_slice> >{
  inline static Stream<Device> *
  Get(const SliceExp<SrcExp, Device, DType, srcdim, dimsrc_m_slice> &t) {
    return StreamInfo<Device, SrcExp>::Get(t.src_);
  }
};
// static typecheck
template<typename SrcExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_slice>
struct ExpInfo<SliceExp<SrcExp, Device, DType, srcdim, dimsrc_m_slice> >{
  static const int kDim = ExpInfo<SrcExp>::kDim;
  static const int kDevMask = ExpInfo<SrcExp>::kDevMask;
};
//----------------------
// Execution plan
//---------------------
template<typename SrcExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_slice>
struct Plan<SliceExp<SrcExp, Device, DType, srcdim, dimsrc_m_slice>, DType> {
 public:
  static const int dimslice = srcdim - dimsrc_m_slice;
  explicit Plan(const SliceExp<SrcExp, Device, DType, srcdim, dimsrc_m_slice> &e)
      : src_(MakePlan(e.src_)),
        height_(e.shape_.ProdShape(dimslice + 1, srcdim - 1)),
        ch_begin_(e.ch_begin_), ch_old_(e.ch_old_), ch_(e.shape_[dimslice]) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t y = i % height_;
    i /= height_;
    const index_t c = i % ch_ + ch_begin_;
    const index_t b = i / ch_;
    const index_t x = j;
    return src_.Eval((b * ch_old_ + c) * height_ + y, x);
  }
  MSHADOW_XINLINE DType &REval(index_t i, index_t j) {
    const index_t y = i % height_;
    i /= height_;
    const index_t c = i % ch_ + ch_begin_;
    const index_t b = i / ch_;
    const index_t x = j;
    return src_.REval((b * ch_old_ + c) * height_ + y, x);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t height_, ch_begin_, ch_old_, ch_;
};  // struct Plan

template<typename SrcExp,
         typename Device, typename DType,
         int srcdim>
struct Plan<SliceExp<SrcExp, Device, DType, srcdim, 1>, DType> {
 public:
  explicit Plan(const SliceExp<SrcExp, Device, DType, srcdim, 1> &e)
      : src_(MakePlan(e.src_)),
        ch_begin_(e.ch_begin_) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return src_.Eval(y, x + ch_begin_);
  }
  MSHADOW_XINLINE DType &REval(index_t y, index_t x) {
    return src_.REval(y, x + ch_begin_);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t ch_begin_;
};
}  // namespace expr
}   // namespace mshadow
#endif  // MSHADOW_EXTENSION_SLICE_H_
