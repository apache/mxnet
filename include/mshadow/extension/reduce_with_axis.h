/*!
 * Copyright (c) 2015 by Contributors
 * \file reduce_with_axis.h
 * \brief
 * \author Junyuan Xie
*/
#ifndef MSHADOW_EXTENSION_REDUCE_WITH_AXIS_H_
#define MSHADOW_EXTENSION_REDUCE_WITH_AXIS_H_

#include "../extension.h"

namespace mshadow {
namespace expr {

/*! \brief reduce out the dimension of src labeled by axis.
 *  \tparam Reducer type of reducer
 *  \tparam SrcExp type of source expression
 *  \tparam DType data type
 */
template<typename Reducer, typename SrcExp, typename DType, int dimsrc, bool mask, int dimdst>
struct ReduceWithAxisExp:
    public MakeTensorExp<ReduceWithAxisExp<Reducer, SrcExp, DType, dimsrc, mask, dimdst>,
                         SrcExp, dimdst, DType> {
  /*! \brief source oprand */
  const SrcExp &src_;
  /*! \brief size of last destination dimension */
  index_t last_dst_dim_;
  /*! \brief size of trailing dimensions */
  index_t trailing_;
  /*! \brief size of axis dimension */
  index_t size_;
  /*! \brief size of last src dimension */
  index_t last_;
  /*! constructor */
  explicit ReduceWithAxisExp(const SrcExp &src, int axis)
    : src_(src) {
    bool keepdim = (dimsrc == dimdst);
    CHECK(dimsrc > axis) << "reduce axis out of bound";
    Shape<dimsrc> src_shape = ShapeCheck<dimsrc, SrcExp>::Check(src_);
    for (int i = 0; i < axis; ++i) {
      this->shape_[i] = src_shape[i];
    }
    this->size_ = src_shape[axis];
    this->trailing_ = 1;
    if (!keepdim) {
      for (int i = axis + 1; i < dimsrc; ++i) {
        this->trailing_ *= src_shape[i];
        this->shape_[i - 1] = src_shape[i];
      }
    } else {
      this->shape_[axis] = 1;
      for (index_t i = axis + 1; i < dimsrc; ++i) {
        this->trailing_ *= src_shape[i];
        this->shape_[i] = src_shape[i];
      }
    }

    this->last_ = src_shape[dimsrc - 1];
    this->last_dst_dim_ = this->shape_[dimdst - 1];
  }
};  // struct ReduceWithAxisExp

/*!
 * \brief reduce out the dimension of src labeled by axis.
 * \param Reducer type of the reducing operation
 * \param mask whether to output the unmask indices
 * \tparam SrcExp source expression
 * \tparam DType data type
 * \tparam etype type of the expression
 */
template<typename Reducer, bool mask, typename SrcExp, typename DType, int etype>
inline ReduceWithAxisExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim, mask,
  ExpInfo<SrcExp>::kDim - 1>
reduce_with_axis(const Exp<SrcExp, DType, etype> &src, int axis) {
  return ReduceWithAxisExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim, mask,
    ExpInfo<SrcExp>::kDim- 1>(src.self(), axis);
}

/*!
* \brief reduce out the dimension of src labeled by axis, keepdim turned on.
* \param Reducer type of the reducing operation
* \param mask whether to output the unmask indices
* \tparam SrcExp source expression
* \tparam DType data type
* \tparam etype type of the expression
*/
template<typename Reducer, bool mask, typename SrcExp, typename DType, int etype>
inline ReduceWithAxisExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim, mask,
  ExpInfo<SrcExp>::kDim>
  reduce_keepdim(const Exp<SrcExp, DType, etype> &src, int axis) {
  return ReduceWithAxisExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim, mask,
    ExpInfo<SrcExp>::kDim>(src.self(), axis);
}

//----------------------
// Execution plan
//----------------------
template<typename Reducer, typename SrcExp, typename DType, int dimsrc, bool mask, int dimdst>
struct Plan<ReduceWithAxisExp<Reducer, SrcExp, DType, dimsrc, mask, dimdst>, DType> {
 public:
  explicit Plan(const ReduceWithAxisExp<Reducer, SrcExp, DType, dimsrc, mask, dimdst> &e)
      : src_(MakePlan(e.src_)), last_dst_dim_(e.last_dst_dim_), trailing_(e.trailing_),
        size_(e.size_), last_(e.last_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    index_t x = (i*last_dst_dim_ + j)/trailing_;
    index_t y = (i*last_dst_dim_ + j)%trailing_;

    if (mask) {
      index_t idx = 0;
      DType res; Reducer::SetInitValue(res);
      for (index_t k = 0; k < size_; ++k) {
        index_t z = (x*size_+k)*trailing_+y;
        DType tmp = res;
        Reducer::Reduce(res, src_.Eval(z/last_, z%last_));
        if (tmp != res) {
          idx = k;
        }
      }
      return static_cast<DType>(static_cast<int>(idx));
    } else {
      DType res; Reducer::SetInitValue(res);
      for (index_t k = 0; k < size_; ++k) {
        index_t z = (x*size_+k)*trailing_+y;
        Reducer::Reduce(res, src_.Eval(z/last_, z%last_));
      }
      return res;
    }
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t last_dst_dim_, trailing_, size_, last_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_REDUCE_WITH_AXIS_H_
