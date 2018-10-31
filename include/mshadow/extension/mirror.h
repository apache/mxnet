/*!
 *  Copyright (c) 2014 by Contributors
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
