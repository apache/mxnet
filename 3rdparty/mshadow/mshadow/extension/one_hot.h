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
 * \file one_hot.h
 * \brief Create one-hot indicator array based on the index.
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_ONE_HOT_H_
#define MSHADOW_EXTENSION_ONE_HOT_H_

#include "../extension.h"


namespace mshadow {
namespace expr {
/*!
 * \brief Create a one-hot indicator array.
 * \tparam IndexExp type of index expression
 * \tparam DType the type of elements
 */
template<typename IndexExp, typename DType>
struct OneHotEncodeExp:
      public Exp<OneHotEncodeExp<IndexExp, DType>,
                 DType, type::kChainer> {
  /*! \brief index operand */
  const IndexExp &index_;
  /*! \brief number of choices we can have. */
  index_t num_choices_;
  /*! \brief constructor */
  OneHotEncodeExp(const IndexExp &index, index_t num_choices)
      : index_(index), num_choices_(num_choices) {}
};

template<typename IndexExp,
         typename IDType, int e1>
inline OneHotEncodeExp<IndexExp, default_real_t>
one_hot_encode(const Exp<IndexExp, IDType, e1> &index, index_t num_choices) {
  TypeCheckPass<ExpInfo<IndexExp>::kDim == 1>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return OneHotEncodeExp<IndexExp, default_real_t>(index.self(), num_choices);
}

//----------------------
// Execution plan
//----------------------
template<typename IndexExp, typename DType>
struct Plan<OneHotEncodeExp<IndexExp, DType>, DType> {
 public:
  explicit Plan(const OneHotEncodeExp<IndexExp, DType> &e)
      : index_(MakePlan(e.index_)) {
  }
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    index_t idx = static_cast<index_t>(index_.Eval(0, y));
    return static_cast<DType>(x == idx);
  }

 private:
  expr::Plan<IndexExp, DType> index_;
};

template<typename IndexExp, typename DType>
inline Plan<OneHotEncodeExp<IndexExp, DType>, DType>
MakePlan(const OneHotEncodeExp<IndexExp, DType> &exp) {
  return Plan<OneHotEncodeExp<IndexExp, DType>, DType>(exp);
}

template<int dim, typename IndexExp, typename DType>
struct ShapeCheck<dim, OneHotEncodeExp<IndexExp, DType> > {
  inline static Shape<dim>
  Check(const OneHotEncodeExp<IndexExp, DType> &t) {
    CHECK(dim == 2)
        << "OneHotEncodeExp only support 2 dimension output";
    Shape<1> shape = ShapeCheck<1, IndexExp>::Check(t.index_);
    Shape<dim> ret;
    ret[0] = shape[0];
    ret[1] = t.num_choices_;
    return ret;
  }
};

template<typename IndexExp, typename DType>
struct ExpInfo<OneHotEncodeExp<IndexExp, DType> > {
  static const int kDim = 2;
  static const int kDevMask = ExpInfo<IndexExp>::kDevMask;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_ONE_HOT_H_
