/*!
 * Copyright (c) 2015 by Contributors
 * \file take_grad.h
 * \brief
 * \author Bing Xu
*/
#ifndef MSHADOW_EXTENSION_TAKE_GRAD_H_
#define MSHADOW_EXTENSION_TAKE_GRAD_H_

#include "../extension.h"

namespace mshadow {
namespace expr {

/*! \brief Calculate embedding gradient
 *  \tparam IndexExp type of index expression
 *  \tparam SrcExp type of src expression
 *  \tparam DType data type
 */

template<typename IndexExp, typename SrcExp, typename DType>
struct TakeGradExp : public Exp<TakeGradExp<IndexExp, SrcExp, DType>,
                                DType, type::kChainer> {
  /*! \brief index oprand */
  const IndexExp &index_;
  /*! \brief out gradient oprand */
  const SrcExp &src_;
  /*! \brief batch size */
  const index_t input_dim_;
  /*! \brief constructor */
  TakeGradExp(const IndexExp &index, const SrcExp &src, const index_t input_dim)
    : index_(index), src_(src), input_dim_(input_dim) {}
};  // struct TakeGradExp


template<typename IndexExp,
         typename SrcExp,
         typename DType,
         int e1, int e2>
inline TakeGradExp<IndexExp, SrcExp, DType>
take_grad(const Exp<IndexExp, DType, e1> &index,
          const Exp<SrcExp, DType, e2> &src,
          const index_t input_dim) {
  return TakeGradExp<IndexExp, SrcExp, DType>(index.self(),
                                                       src.self(),
                                                       input_dim);
}

//----------------------
// Execution plan
//----------------------

template<typename IndexExp, typename SrcExp, typename DType>
struct Plan<TakeGradExp<IndexExp, SrcExp, DType>, DType> {
 public:
  explicit Plan(const TakeGradExp<IndexExp, SrcExp, DType> &e)
    : index_(MakePlan(e.index_)),
      src_(MakePlan(e.src_)),
      batch_size_(ShapeCheck<1, IndexExp>::Check(e.index_)[0]) {
  }

  // now return shape: in * out
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    DType ret = 0.f;
    for (index_t i = 0; i < batch_size_; ++i) {
      index_t idx = static_cast<index_t>(index_.Eval(0, i));
      if (idx == y) {
        ret += static_cast<DType>(src_.Eval(i, x));
      }
    }
    return ret;
  }

 private:
  expr::Plan<IndexExp, DType> index_;
  expr::Plan<SrcExp, DType> src_;
  const index_t batch_size_;
};  // struct Plan


template<typename IndexExp, typename SrcExp, typename DType>
inline Plan<TakeGradExp<IndexExp, SrcExp, DType>, DType>
MakePlan(const TakeGradExp<IndexExp, SrcExp, DType> &exp) {
  return Plan<TakeGradExp<IndexExp, SrcExp, DType>, DType>(exp);
}

template<int dim, typename IndexExp, typename SrcExp, typename DType>
struct ShapeCheck<dim, TakeGradExp<IndexExp, SrcExp, DType> > {
  inline static Shape<dim>
  Check(const TakeGradExp<IndexExp, SrcExp, DType> &t) {
    CHECK(dim == 2)
      << "TakeGradExp only support 2D output";
    // Shape<1> dshape = ShapeCheck<1, IndexExp>::Check(t.index_);
    Shape<2> gshape = ShapeCheck<2, SrcExp>::Check(t.src_);
    Shape<dim> ret;
    ret[0] = t.input_dim_;
    ret[1] = gshape[1];
    return ret;
  }
};  // struct ShapeCheck

template<typename IndexExp, typename SrcExp, typename DType>
struct ExpInfo<TakeGradExp<IndexExp, SrcExp, DType> > {
  static const int kDim = 2;
  static const int kDevMask = ExpInfo<IndexExp>::kDevMask;
};

}  // namespace expr
}  // namespace mshadow

#endif  // MSHADOW_EXTENSION_TAKE_GRAD_H_
