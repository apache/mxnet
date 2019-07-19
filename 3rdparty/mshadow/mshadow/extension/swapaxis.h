/*!
 *  Copyright (c) 2014 by Contributors
 * \file swapaxis.h
 * \brief support for swapaxis
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_SWAPAXIS_H_
#define MSHADOW_EXTENSION_SWAPAXIS_H_
#include <algorithm>
#include <utility>
#include "../extension.h"
namespace mshadow {
namespace expr {
/*!
 * \brief swap two axis of a tensor
 * input: Tensor<Device,dim>: ishape
 * output: Tensor<Device,dimdst> oshape[a1],oshape[a2] = ishape[a2],oshape[a1]
 *
 * \tparam SrcExp type of source expression
 * \tparam DType the type of elements 
 * \tparam dimsrc source dimension, assert a1 > a2
 * \tparam m_a1 one dimension to be swapped, encoded by dimsrc - a1 
 * \tparam a2 second dimension to be swapped, encoded by a2
 */
template<typename SrcExp, typename DType, int dimsrc, int m_a1, int a2>
struct SwapAxisExp:
      public MakeTensorExp<SwapAxisExp<SrcExp, DType, dimsrc, m_a1, a2>,
                           SrcExp, dimsrc, DType> {
  // decode the a1, a2
  static const int a1 = dimsrc - m_a1;
  /*! \brief source expression */
  const SrcExp &src_;
  /*! \brief constructor */
  explicit SwapAxisExp(const SrcExp &src) : src_(src) {
    this->shape_ = ShapeCheck<dimsrc, SrcExp>::Check(src);
    std::swap(this->shape_[a1], this->shape_[a2]);
  }
};
/*!
 * \brief a expression that reshapes a tensor to another shape
 * \param src Tensor<Device,dimsrc>:
 * \return a expresion with type Tensor<Device,dimdst>
 * \tparam a1 higher dimension to be swapped, assert a1 > a2
 * \tparam a2 lower dimension to be swapped
 * \tparam SrcExp source expression
 * \tparam DType the type of elements 
 * \tparam etype source expression type
 */
template<int a1, int a2, typename SrcExp, typename DType, int etype>
inline SwapAxisExp<SrcExp, DType, ExpInfo<SrcExp>::kDim,
                   ExpInfo<SrcExp>::kDim - a1, a2>
swapaxis(const Exp<SrcExp, DType, etype> &src) {
  typedef ExpInfo<SrcExp> Info;
  TypeCheckPass<Info::kDim >= a1 + 1 && Info::kDim >= a2 + 1 &&
                a2 < a1>::Error_Expression_Does_Not_Meet_Dimension_Req();
  return SwapAxisExp<SrcExp, DType, ExpInfo<SrcExp>::kDim,
                     ExpInfo<SrcExp>::kDim - a1, a2>(src.self());
}
template<typename SrcExp, typename DType, int dimsrc, int m_a1, int a2>
struct Plan<SwapAxisExp<SrcExp, DType, dimsrc, m_a1, a2>, DType> {
 public:
  // decode the a1
  static const int a1 = dimsrc - m_a1;
  explicit Plan(const SwapAxisExp<SrcExp, DType, dimsrc, m_a1, a2> &e)
      : src_(MakePlan(e.src_)),
        shapey_(e.shape_.ProdShape(a1 + 1, dimsrc - 1)),
        shapez_(e.shape_[a1]),
        shapec_(e.shape_.ProdShape(a2 + 1, a1)),
        shapen_(e.shape_[a2]) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t y = i % shapey_;
    i /= shapey_;
    const index_t z = i % shapez_;
    i /= shapez_;
    const index_t c = i % shapec_;
    i /= shapec_;
    const index_t n = i % shapen_;
    // swap z and n
    return src_.Eval(((((i / shapen_) * shapez_ + z) * shapec_ +
                          c) * shapen_ + n) * shapey_ + y, j);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t shapey_, shapez_, shapec_, shapen_;
};
template<typename SrcExp, typename DType, int dimsrc, int a2>
struct Plan<SwapAxisExp<SrcExp, DType, dimsrc, 1, a2>, DType> {
 public:
  explicit Plan(const SwapAxisExp<SrcExp, DType, dimsrc, 1, a2> &e)
      : src_(MakePlan(e.src_)),
        shapex_(e.shape_[dimsrc - 1]),
        shapey_(e.shape_.ProdShape(a2 + 1, dimsrc - 1)),
        shapez_(e.shape_[a2]) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t x) const {
    // swap x and z
    const index_t y = i % shapey_;
    i /= shapey_;
    const index_t z = i % shapez_;
    const index_t n = i / shapez_;
    return src_.Eval((n * shapex_ + x) * shapey_ + y , z);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t shapex_, shapey_, shapez_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_SWAPAXIS_H_
