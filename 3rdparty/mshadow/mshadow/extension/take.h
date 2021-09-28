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
 * \file take.h
 * \brief
 * \author Bing Xu
*/
#ifndef MSHADOW_EXTENSION_TAKE_H_
#define MSHADOW_EXTENSION_TAKE_H_

#include "../extension.h"

namespace mshadow {
namespace expr {

/*! \brief Take a column from a matrix
 *  \tparam IndexExp type of index expression
 *  \tparam SrcExp type of src expression
 *  \tparam DType data type
 */
template<typename IndexExp, typename SrcExp, typename DType>
struct TakeExp: public Exp<TakeExp<IndexExp, SrcExp, DType>,
                           DType, type::kChainer> {
  /*! \brief index oprand */
  const IndexExp &index_;
  /*! \brief embediing oprand */
  const SrcExp &src_;
  /*! constructor */
  TakeExp(const IndexExp &index, const SrcExp &src)
    : index_(index), src_(src) {}
};  // struct TakeExp



template<typename IndexExp,
         typename SrcExp,
         typename DType,
         int e1, int e2>
inline TakeExp<IndexExp, SrcExp, DType>
take(const Exp<IndexExp, DType, e1> &index,
     const Exp<SrcExp, DType, e2> &src) {
  return TakeExp<IndexExp, SrcExp, DType>(index.self(), src.self());
}


//----------------------
// Execution plan
//----------------------

template<typename IndexExp, typename SrcExp, typename DType>
struct Plan<TakeExp<IndexExp, SrcExp, DType>, DType> {
 public:
  explicit Plan(const TakeExp<IndexExp, SrcExp, DType> &e)
    : index_(MakePlan(e.index_)), src_(MakePlan(e.src_)) {
  }

  // TODO(xx): discuss W shape: in * out or out * in
  // Now I use in * out
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    index_t idx = static_cast<index_t>(index_.Eval(0, y));
    return static_cast<DType>(src_.Eval(idx, x));
  }

 private:
  expr::Plan<IndexExp, DType> index_;
  expr::Plan<SrcExp, DType> src_;
};  // struct Plan

template<typename IndexExp, typename SrcExp, typename DType>
inline Plan<TakeExp<IndexExp, SrcExp, DType>, DType>
MakePlan(const TakeExp<IndexExp, SrcExp, DType> &exp) {
  return Plan<TakeExp<IndexExp, SrcExp, DType>, DType>(exp);
}

template<int dim, typename IndexExp, typename SrcExp, typename DType>
struct ShapeCheck<dim, TakeExp<IndexExp, SrcExp, DType> > {
  inline static Shape<dim>
  Check(const TakeExp<IndexExp, SrcExp, DType> &t) {
    CHECK(dim == 2)
      << "TakeExp only support 2D output";
    Shape<1> dshape = ShapeCheck<1, IndexExp>::Check(t.index_);
    Shape<2> wshape = ShapeCheck<2, SrcExp>::Check(t.src_);
    Shape<dim> ret;
    ret[0] = dshape[0];
    ret[1] = wshape[1];
    return ret;
  }
};


template<typename IndexExp, typename SrcExp, typename DType>
struct ExpInfo<TakeExp<IndexExp, SrcExp, DType> > {
  static const int kDim = 2;
  static const int kDevMask = ExpInfo<IndexExp>::kDevMask;
};

}  // namespace expr
}  // namespace mshadow

#endif  // MSHADOW_EXTENSION_TAKE_H_
