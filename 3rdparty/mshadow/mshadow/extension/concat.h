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
 * \file concat.h
 * \brief support for concatenation
 */
#ifndef MSHADOW_EXTENSION_CONCAT_H_
#define MSHADOW_EXTENSION_CONCAT_H_

#include "../extension.h"

namespace mshadow {
namespace expr {
/*!
 * \brief concat expression, concat two tensor's channel
 * \tparam LhsExp left expression
 * \tparam RhsExp right expression
 * \tparam DType the type of elements
 * \tparam srcdim dimension of src
 * \tparam dimsrc_m_cat dimsrc - dimcat
 */
template<typename LhsExp, typename RhsExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_cat>
struct ConcatExp : public TRValue<ConcatExp<LhsExp, RhsExp,
                                            Device, DType,
                                            srcdim, dimsrc_m_cat>,
                                  Device, srcdim, DType> {
  static const int dimcat = srcdim - dimsrc_m_cat;
  const LhsExp &src1_;
  const RhsExp &src2_;
  index_t dcat_src1_;
  index_t dcat_src2_;
  Shape<4> shape_;
  ConcatExp(const LhsExp &src1, const RhsExp &src2) : src1_(src1), src2_(src2) {
    Shape<srcdim> sshape1 = ShapeCheck<srcdim, LhsExp>::Check(src1_);
    Shape<srcdim> sshape2 = ShapeCheck<srcdim, RhsExp>::Check(src2_);
    #pragma unroll
    for (int i = 0; i < srcdim; ++i) {
      if (i != dimcat) {
        CHECK_EQ(sshape1[i], sshape2[i]) << "ConcatExp: shape mismatch";
      }
    }
    this->shape_ = sshape1;
    this->shape_[dimcat] = sshape1[dimcat] + sshape2[dimcat];
    this->dcat_src1_ = sshape1[dimcat];
    this->dcat_src2_ = sshape2[dimcat];
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
};  // struct ConcatExp
/*!
 * \brief concat two 4D tensor
 * \param src1 source tensor1
 * \param src2 source tensor2
 * \return concated 4D tensor
 * \tparam cdim the dimension to concatnate on
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam etype type of expression
 */
template<int cdim, typename LhsExp, typename RhsExp,
         typename Device, typename DType, int srcdim>
inline ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, srcdim - cdim>
concat(const TRValue<LhsExp, Device, srcdim, DType> &src1,
       const TRValue<RhsExp, Device, srcdim, DType> &src2) {
  TypeCheckPass<ExpInfo<LhsExp>::kDim == ExpInfo<RhsExp>::kDim>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  TypeCheckPass<cdim < srcdim && ExpInfo<LhsExp>::kDim == srcdim>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, srcdim - cdim>
      (src1.self(), src2.self());
}
//------------------------
//  engine plugin
//------------------------
// runtime shapecheck
template<typename LhsExp, typename RhsExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_cat>
struct ShapeCheck<srcdim, ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, dimsrc_m_cat> >{
  inline static Shape<srcdim> Check(const ConcatExp<LhsExp, RhsExp,
                                    Device, DType, srcdim, dimsrc_m_cat> &t) {
    return t.shape_;
  }
};
template<typename LhsExp, typename RhsExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_cat>
struct StreamInfo<Device, ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, dimsrc_m_cat> >{
  inline static Stream<Device> *
  Get(const ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, dimsrc_m_cat> &t) {
    Stream<Device> *lhs = StreamInfo<Device, LhsExp>::Get(t.src1_);
    Stream<Device> *rhs = StreamInfo<Device, RhsExp>::Get(t.src2_);
    if (lhs != rhs) return NULL;
    return lhs;
  }
};
// static typecheck
template<typename LhsExp, typename RhsExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_cat>
struct ExpInfo<ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, dimsrc_m_cat> >{
  static const int kDimLhs = ExpInfo<LhsExp>::kDim;
  static const int kDimRhs = ExpInfo<RhsExp>::kDim;
  // copy from binarymap
  static const int kDim = (kDimLhs >= 0 && kDimRhs >= 0) ?\
      (kDimLhs == 0 ?\
       kDimRhs :\
       ((kDimRhs == 0 || kDimLhs == kDimRhs) ? kDimLhs : -1)) : -1;
  static const int kDevMask = ExpInfo<LhsExp>::kDevMask & ExpInfo<RhsExp>::kDevMask;
};
//----------------------
// Execution plan
//---------------------
template<typename LhsExp, typename RhsExp,
         typename Device, typename DType,
         int srcdim, int dimsrc_m_cat>
struct Plan<ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, dimsrc_m_cat>, DType> {
 public:
  static const int dimcat = srcdim - dimsrc_m_cat;
  explicit Plan(const ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, dimsrc_m_cat> &e)
      : src1_(MakePlan(e.src1_)), src2_(MakePlan(e.src2_)),
        height_(e.shape_.ProdShape(dimcat + 1, srcdim - 1)),
        ch_src1_(e.dcat_src1_), ch_src2_(e.dcat_src2_), ch_(e.shape_[dimcat]) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t y = i % height_;
    i /= height_;
    const index_t c = i % ch_;
    const index_t b = i / ch_;
    const index_t x = j;
    if (c < ch_src1_) {
      return src1_.Eval((b * ch_src1_ + c) * height_ + y, x);
    } else {
      return src2_.Eval((b * ch_src2_ + c - ch_src1_) * height_ + y, x);
    }
  }
  MSHADOW_XINLINE DType &REval(index_t i, index_t j) {
    const index_t y = i % height_;
    i /= height_;
    const index_t c = i % ch_;
    const index_t b = i / ch_;
    const index_t x = j;
    if (c < ch_src1_) {
      return src1_.REval((b * ch_src1_ + c) * height_ + y, x);
    } else {
      return src2_.REval((b * ch_src2_ + c - ch_src1_) * height_ + y, x);
    }
  }

 private:
  Plan<LhsExp, DType> src1_;
  Plan<RhsExp, DType> src2_;
  const index_t height_, ch_src1_, ch_src2_, ch_;
};  // struct Plan

// specialize for concat in x
template<typename LhsExp, typename RhsExp,
         typename Device, typename DType,
         int srcdim>
struct Plan<ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, 1>, DType> {
 public:
  explicit Plan(const ConcatExp<LhsExp, RhsExp, Device, DType, srcdim, 1> &e)
      : src1_(MakePlan(e.src1_)), src2_(MakePlan(e.src2_)),
        width_src1_(e.dcat_src1_) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    if (x < width_src1_) {
      return src1_.Eval(y, x);
    } else {
      return src2_.Eval(y, x - width_src1_);
    }
  }
  MSHADOW_XINLINE DType &REval(index_t y, index_t x) {
    if (x < width_src1_) {
      return src1_.REval(y, x);
    } else {
      return src2_.REval(y, x - width_src1_);
    }
  }

 private:
  Plan<LhsExp, DType> src1_;
  Plan<RhsExp, DType> src2_;
  const index_t width_src1_;
};
}  // namespace expr
}   // namespace mshadow
#endif  // MSHADOW_EXTENSION_CONCAT_H_
