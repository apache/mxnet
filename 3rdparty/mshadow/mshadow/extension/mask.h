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
 * \file mask.h
 * \brief
 * \author Bing Xu
*/
#ifndef MSHADOW_EXTENSION_MASK_H_
#define MSHADOW_EXTENSION_MASK_H_

#include "../extension.h"

namespace mshadow {
namespace expr {

/*! \brief Broadcast a mask and do element-wise multiplication
 *  \tparam IndexExp type of index expression
 *  \tparam SrcExp type of src expression
 *  \tparam DType data type
 */
template<typename IndexExp, typename SrcExp, typename DType>
struct MaskExp: public Exp<MaskExp<IndexExp, SrcExp, DType>,
                           DType, type::kChainer> {
  /*! \brief index oprand */
  const IndexExp &index_;
  /*! \brief matrix oprand */
  const SrcExp &src_;
  /*! constructor */
  MaskExp(const IndexExp &index, const SrcExp &src)
    : index_(index), src_(src) {}
};  // struct MaskExp



template<typename IndexExp,
         typename SrcExp,
         typename DType,
         int e1, int e2>
inline MaskExp<IndexExp, SrcExp, DType>
mask(const Exp<IndexExp, DType, e1> &index,
     const Exp<SrcExp, DType, e2> &src) {
  return MaskExp<IndexExp, SrcExp, DType>(index.self(), src.self());
}


//----------------------
// Execution plan
//----------------------

template<typename IndexExp, typename SrcExp, typename DType>
struct Plan<MaskExp<IndexExp, SrcExp, DType>, DType> {
 public:
  explicit Plan(const MaskExp<IndexExp, SrcExp, DType> &e)
    : index_(MakePlan(e.index_)), src_(MakePlan(e.src_)) {
  }

  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return static_cast<DType>(src_.Eval(y, x) * index_.Eval(0, y));
  }

 private:
  expr::Plan<IndexExp, DType> index_;
  expr::Plan<SrcExp, DType> src_;
};  // struct Plan

template<typename IndexExp, typename SrcExp, typename DType>
inline Plan<MaskExp<IndexExp, SrcExp, DType>, DType>
MakePlan(const MaskExp<IndexExp, SrcExp, DType> &exp) {
  return Plan<MaskExp<IndexExp, SrcExp, DType>, DType>(exp);
}

template<int dim, typename IndexExp, typename SrcExp, typename DType>
struct ShapeCheck<dim, MaskExp<IndexExp, SrcExp, DType> > {
  inline static Shape<dim>
  Check(const MaskExp<IndexExp, SrcExp, DType> &t) {
    CHECK(dim == 2)
      << "MaskExp only support 2D output";
    Shape<1> dshape = ShapeCheck<1, IndexExp>::Check(t.index_);
    Shape<2> wshape = ShapeCheck<2, SrcExp>::Check(t.src_);
    CHECK_EQ(dshape[0], wshape[0]) << "MaskExp require inputs in same first dimention";
    Shape<dim> ret;
    ret[0] = wshape[0];
    ret[1] = wshape[1];
    return ret;
  }
};


template<typename IndexExp, typename SrcExp, typename DType>
struct ExpInfo<MaskExp<IndexExp, SrcExp, DType> > {
  static const int kDim = 2;
  static const int kDevMask = ExpInfo<IndexExp>::kDevMask;
};

}  // namespace expr
}  // namespace mshadow

#endif  // MSHADOW_EXTENSION_MASK_H_
