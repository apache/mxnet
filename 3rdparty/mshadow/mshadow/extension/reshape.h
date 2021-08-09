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
 * \file reshape.h
 * \brief support for reshape
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_RESHAPE_H_
#define MSHADOW_EXTENSION_RESHAPE_H_
#include "../extension.h"
namespace mshadow {
namespace expr {
/*!
 * \brief reshape the content to another shape
 * input: Tensor<Device,dimsrc>: ishape
 * output: Tensor<Device,dimdst> ishape.Size() == oshape.Size()
 * \tparam SrcExp source expression
 * \tparam dimdst target dimension
 * \tparam dimsrc source dimension
 */
template<typename SrcExp, typename DType, int dimdst, int dimsrc>
struct ReshapeExp:
      public MakeTensorExp<ReshapeExp<SrcExp, DType, dimdst, dimsrc>,
                           SrcExp, dimdst, DType> {
  /*! \brief source expression */
  const SrcExp &src_;
  /*! \brief smallest dimension of input */
  index_t ishapex_;
  /*! \brief constructor */
  ReshapeExp(const SrcExp &src, Shape<dimdst> shape)
      : src_(src) {
    Shape<dimsrc> ishape = ShapeCheck<dimsrc, SrcExp>::Check(src_);
    CHECK_EQ(ishape.Size(), shape.Size()) << "reshape size must match";
    ishapex_ = ishape[dimsrc - 1];
    this->shape_ = shape;
  }
};
/*!
 * \brief a expression that reshapes a tensor to another shape
 * \param src Tensor<Device,dimsrc>:
 * \param oshape target shape
 * \return a expresion with type Tensor<Device,dimdst>
 * \tparam SrcExp source expression
 * \tparam etype source expression type
 * \tparam dimdst target dimension
 */
template<typename SrcExp, typename DType, int etype, int dimdst>
inline ReshapeExp<SrcExp, DType, dimdst, ExpInfo<SrcExp>::kDim>
reshape(const Exp<SrcExp, DType, etype> &src, Shape<dimdst> oshape) {
  return ReshapeExp<SrcExp, DType, dimdst, ExpInfo<SrcExp>::kDim>
      (src.self(), oshape);
}
//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int dimdst, int dimsrc>
struct Plan<ReshapeExp<SrcExp, DType, dimdst, dimsrc>, DType> {
 public:
  explicit Plan(const ReshapeExp<SrcExp, DType, dimdst, dimsrc> &e)
      : src_(MakePlan(e.src_)),
        oshapex_(e.shape_[dimdst - 1]), ishapex_(e.ishapex_) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    const index_t idx = y * oshapex_ + x;
    return src_.Eval(idx / ishapex_, idx % ishapex_);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t oshapex_, ishapex_;
};
// special work plan for 1 dimensional data
template<typename SrcExp, typename DType, int dimdst>
struct Plan<ReshapeExp<SrcExp, DType, dimdst, 1>, DType> {
 public:
  explicit Plan(const ReshapeExp<SrcExp, DType, dimdst, 1> &e)
      : src_(MakePlan(e.src_)), oshapex_(e.shape_[dimdst - 1]) {
  }
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return src_.Eval(0, y * oshapex_ + x);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t oshapex_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_RESHAPE_H_
