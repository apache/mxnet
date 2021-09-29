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
 * \file mirror.h
 * \brief support for mirror
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_MIRROR_H_
#define MSHADOW_EXTENSION_MIRROR_H_
#include "../extension.h"
namespace mshadow {
namespace expr {
/*!
 * \brief mirror expression, mirror a image in width
 * \tparam SrcExp source expression to be mirrored
 * \tparam DType the type of elements
 * \tparam srcdim dimension of src
 */
template<typename SrcExp, typename DType, int srcdim>
struct MirroringExp:
      public MakeTensorExp<MirroringExp<SrcExp, DType, srcdim>,
                           SrcExp, srcdim, DType> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief constructor */
  explicit MirroringExp(const SrcExp &src) : src_(src) {
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
  }
};
/*!
 * \brief mirroring expression, mirror images in width
 * \param src original image batches
 * \return expression corresponding to mirrored result
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam etype type of expression
 */
template<typename SrcExp, typename DType, int etype>
inline MirroringExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
mirror(const Exp<SrcExp, DType, etype> &src) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return MirroringExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self());
}
//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int srcdim>
struct Plan<MirroringExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const MirroringExp<SrcExp, DType, srcdim> &e)
      : src_(MakePlan(e.src_)), width_(e.shape_[srcdim - 1]) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    return src_.Eval(i, width_ - j - 1);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t width_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_MIRROR_H_
