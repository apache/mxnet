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
#ifndef MSHADOW_EXTENSION_SLICE_EX_H_
#define MSHADOW_EXTENSION_SLICE_EX_H_

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
struct SliceExExp : public TRValue<SliceExExp<SrcExp,
                                              Device, DType,
                                              srcdim>,
                                   Device, srcdim, DType> {
  const SrcExp &src_;
  Shape<srcdim> src_shape_;
  Shape<srcdim> shape_;
  const Shape<srcdim> begin_;
  const Shape<srcdim> end_;
  SliceExExp(const SrcExp &src, Shape<srcdim> begin, Shape<srcdim> end)
      : src_(src), begin_(begin), end_(end) {
    src_shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
    for (int i = 0; i < srcdim; ++i) {
      shape_[i] = end_[i] - begin_[i];
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
};  // struct SliceEx

/*!
 * \brief SliceEx a Tensor
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
inline SliceExExp<SrcExp, Device, DType, srcdim>
slice(const TRValue<SrcExp, Device, srcdim, DType> &src, Shape<srcdim> begin, Shape<srcdim> end) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim == srcdim>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return SliceExExp<SrcExp, Device, DType, srcdim>(src.self(), begin, end);
}
//------------------------
//  engine plugin
//------------------------
// runtime shapecheck
template<typename SrcExp, typename Device,
         typename DType, int srcdim>
struct ShapeCheck<srcdim, SliceExExp<SrcExp, Device, DType, srcdim> >{
  inline static Shape<srcdim> Check(const SliceExExp<SrcExp,
                                    Device, DType, srcdim> &t) {
    return t.shape_;
  }
};

template<typename SrcExp, typename Device,
         typename DType, int srcdim>
struct StreamInfo<Device, SliceExExp<SrcExp, Device, DType, srcdim> >{
  inline static Stream<Device> *
  Get(const SliceExExp<SrcExp, Device, DType, srcdim> &t) {
    return StreamInfo<Device, SrcExp>::Get(t.src_);
  }
};
// static typecheck
template<typename SrcExp, typename Device,
         typename DType, int srcdim>
struct ExpInfo<SliceExExp<SrcExp, Device, DType, srcdim> >{
  static const int kDim = ExpInfo<SrcExp>::kDim;
  static const int kDevMask = ExpInfo<SrcExp>::kDevMask;
};
//----------------------
// Execution plan
//---------------------
template<typename SrcExp, typename Device,
         typename DType, int srcdim>
struct Plan<SliceExExp<SrcExp, Device, DType, srcdim>, DType> {
 public:
  explicit Plan(const SliceExExp<SrcExp, Device, DType, srcdim> &e)
      : src_(MakePlan(e.src_)), begin_(e.begin_),
        src_shape_(e.src_shape_), shape_(e.shape_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    index_t idx = 0;
    index_t stride = 1;
    #pragma unroll
    for (int k = srcdim-2; k >= 0; --k) {
      idx += stride * (i%shape_[k] + begin_[k]);
      i /= shape_[k];
      stride *= src_shape_[k];
    }
    return src_.Eval(idx, j + begin_[srcdim-1]);
  }
  MSHADOW_XINLINE DType &REval(index_t i, index_t j) {
    index_t idx = 0;
    index_t stride = 1;
    #pragma unroll
    for (int k = srcdim-2; k >= 0; --k) {
      idx += stride * (i%shape_[k] + begin_[k]);
      i /= shape_[k];
      stride *= src_shape_[k];
    }
    return src_.REval(idx, j + begin_[srcdim-1]);
  }

 private:
  Plan<SrcExp, DType> src_;
  const Shape<srcdim> begin_, src_shape_, shape_;
};  // struct Plan
}  // namespace expr
}   // namespace mshadow
#endif  // MSHADOW_EXTENSION_SLICE_EX_H_
