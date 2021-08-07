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
 * \file choose.h
 * \brief support for implicit array selection operation
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_CHOOSE_H_
#define MSHADOW_EXTENSION_CHOOSE_H_

#include "../extension.h"

namespace mshadow {
namespace expr {
/*!
 * \brief Make a choice of index in the lowest changing dimension.
 * \tparam SrcExp type of lhs expression
 * \tparam IndexExp type of index expression
 * \tparam DType the type of elements
 */
template<typename SrcExp, typename IndexExp, typename DType>
struct MatChooseRowElementExp:
      public Exp<MatChooseRowElementExp<SrcExp, IndexExp, DType>,
                 DType, type::kChainer> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief index operand */
  const IndexExp &index_;
  /*! \brief constructor */
  MatChooseRowElementExp(const SrcExp &src, const IndexExp &index)
      : src_(src), index_(index) {}
};

template<typename SrcExp, typename IndexExp,
         typename DType, typename IDType, int e1, int e2>
inline MatChooseRowElementExp<SrcExp, IndexExp, DType>
mat_choose_row_element(const Exp<SrcExp, DType, e1> &src,
                       const Exp<IndexExp, IDType, e2> &index) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim == 2 && ExpInfo<IndexExp>::kDim == 1>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return MatChooseRowElementExp<SrcExp, IndexExp, DType>(src.self(), index.self());
}

//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename IndexExp, typename DType>
struct Plan<MatChooseRowElementExp<SrcExp, IndexExp, DType>, DType> {
 public:
  explicit Plan(const MatChooseRowElementExp<SrcExp, IndexExp, DType> &e)
      : src_(MakePlan(e.src_)),
        index_(MakePlan(e.index_)) {
  }
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    index_t idx = static_cast<index_t>(index_.Eval(0, x));
    return src_.Eval(x, idx);
  }

 private:
  expr::Plan<SrcExp, DType> src_;
  expr::Plan<IndexExp, DType> index_;
};

template<typename SrcExp, typename IndexExp, typename DType>
inline Plan<MatChooseRowElementExp<SrcExp, IndexExp, DType>, DType>
MakePlan(const MatChooseRowElementExp<SrcExp, IndexExp, DType> &exp) {
  return Plan<MatChooseRowElementExp<SrcExp, IndexExp, DType>, DType>(exp);
}

template<int dim, typename SrcExp, typename IndexExp, typename DType>
struct ShapeCheck<dim, MatChooseRowElementExp<SrcExp, IndexExp, DType> > {
  inline static Shape<dim>
  Check(const MatChooseRowElementExp<SrcExp, IndexExp, DType> &t) {
    CHECK(dim == 1)
        << "MatChooseRowElementExp only support 1 dimension output";
    Shape<2> shape1 = ShapeCheck<2, SrcExp>::Check(t.src_);
    Shape<dim> shape2 = ShapeCheck<dim, IndexExp>::Check(t.index_);
    CHECK_EQ(shape1[0], shape2[0])
        << "mat_choose_row_element index length and number of rows in matrix";
    return shape2;
  }
};

template<typename SrcExp, typename IndexExp, typename DType>
struct ExpInfo<MatChooseRowElementExp<SrcExp, IndexExp, DType> > {
  static const int kDim = 1;
  static const int kDevMask = ExpInfo<SrcExp>::kDevMask & ExpInfo<IndexExp>::kDevMask;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_CHOOSE_H_
