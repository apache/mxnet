/*!
 *  Copyright (c) 2014 by Contributors
 * \file spatial_unpool.h
 * \brief support for unpool
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_SPATIAL_UNPOOL_H_
#define MSHADOW_EXTENSION_SPATIAL_UNPOOL_H_
#include <algorithm>
#include "../extension.h"
namespace mshadow {
namespace expr {
/*!
 * \brief unpooling expr reverse operation of pooling, used to pass gradient back
 * \tparam Reducer reduction method during pooling
 * \tparam SrcExp source expression to be pooled from
 * \tparam DType the content data type
 * \tparam srcdim dimension of src
 */
template<typename Reducer, typename SrcExp, typename DType, int srcdim>
struct UnPoolingExp:
      public MakeTensorExp<UnPoolingExp<Reducer, SrcExp, DType, srcdim>,
                           SrcExp, srcdim, DType> {
  /*! \brief source input, corresponds to src in pooling */
  const SrcExp &data_src_;
  /*! \brief result of pooled data, corresponds to result of pooling */
  const SrcExp &data_pooled_;
  /*! \brief gradient data of pooled part, to be propgate down */
  const SrcExp &grad_pooled_;
  /*! \brief shape of pooled expression */
  index_t pshape_y_;
  /*! \brief shape of pooled expression */
  index_t pshape_x_;
  /*! \brief kernel size in height */
  index_t ksize_y_;
  /*! \brief kernel size in width */
  index_t ksize_x_;
  /*! \brief kernel stride in y directory */
  index_t kstride_y_;
  /*! \brief kernel stride in x directory */
  index_t kstride_x_;
  /*! \brief constructor */
  UnPoolingExp(const SrcExp &data_src,
               const SrcExp &data_pooled,
               const SrcExp &grad_pooled,
               index_t ksize_y, index_t ksize_x, index_t kstride_y, index_t kstride_x)
      : data_src_(data_src), data_pooled_(data_pooled),
        grad_pooled_(grad_pooled),
    ksize_y_(ksize_y), ksize_x_(ksize_x),
    kstride_y_(kstride_y), kstride_x_(kstride_x) {
    Shape<srcdim> pshape = ShapeCheck<srcdim, SrcExp>::Check(grad_pooled);
    typedef ShapeCheck<srcdim, SrcExp> ShapeCheckSrcDimSrcExp;
    CHECK_EQ(pshape, ShapeCheckSrcDimSrcExp::Check(data_pooled))
      << "UnPoolingExp: pooled shape mismatch";
    Shape<srcdim> sshape = ShapeCheck<srcdim, SrcExp>::Check(data_src);
    for (int k = 0;  k < srcdim - 2; ++k) {
      CHECK_EQ(pshape[k], sshape[k]) << "UnPoolingExp: pool and src shape mismatch";
    }
    pshape_x_ = pshape[srcdim - 1];
    pshape_y_ = pshape[srcdim - 2];
    this->shape_ = sshape;
  }
};
/*!
 * \brief unpooling gradient for 4D, backprop gradient value back, revserse operation of pooling,
 *   same as unpooling, but allows unequal size of kernel
 * \param data_src  source input, corresponds to src in pooling
 * \param data_pooled result of pooled data, corresponds to result of pooling
 * \param grad_pooled gradient data of pooled part, to be propgate down
 * \param ksize_y kernel height
 * \param ksize_x kernel width
 * \param kstride_y stride in y directory
 * \param kstride_x stride in x directory
 * \return expression corresponding to unpooled 4D Tensor, storing backproped gradient
 * \tparam Reducer reducer type
 * \tparam SrcExp source expression
 * \tparam DType the content data type
 * \tparam etype type of expression
 */
template<typename Reducer, typename SrcExp, typename DType, int etype>
inline UnPoolingExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
unpool(const Exp<SrcExp, DType, etype> &data_src,
       const Exp<SrcExp, DType, etype> &data_pooled,
       const Exp<SrcExp, DType, etype> &grad_pooled,
       index_t ksize_y, index_t ksize_x, index_t kstride_y, index_t kstride_x) {
  return UnPoolingExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
      (data_src.self(), data_pooled.self(), grad_pooled.self(),
       ksize_y, ksize_x, kstride_y, kstride_x);
}
//----------------------
// Execution plan
//----------------------
template<typename Reducer, typename SrcExp, typename DType, int srcdim>
struct Plan<UnPoolingExp<Reducer, SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const UnPoolingExp<Reducer, SrcExp, DType, srcdim> &e)
      : data_src_(MakePlan(e.data_src_)), data_pooled_(MakePlan(e.data_pooled_)),
        grad_pooled_(MakePlan(e.grad_pooled_)), sshape_y_(e.shape_[srcdim - 2]),
        pshape_y_(e.pshape_y_),  pshape_x_(e.pshape_x_),
        ksize_y_(e.ksize_y_), ksize_x_(e.ksize_x_),
        kstride_y_(e.kstride_y_), kstride_x_(e.kstride_x_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    using namespace std;
    const index_t x = j;
    const index_t y = i % sshape_y_;
    const index_t c = i / sshape_y_;
    const DType vsrc = data_src_.Eval(i, j);
    const index_t py_min =
        y < ksize_y_ ? 0 : (y - ksize_y_ + kstride_y_) / kstride_y_;
    const index_t px_min =
        x < ksize_x_ ? 0 : (x - ksize_x_ + kstride_x_) / kstride_x_;
    const index_t py_max = min((y + kstride_y_) / kstride_y_, pshape_y_);
    const index_t px_max = min((x + kstride_x_) / kstride_x_, pshape_x_);

    DType val = static_cast<DType>(0);
    for (index_t py = py_min; py < py_max; ++py) {
      for (index_t px = px_min; px < px_max; ++px) {
        val += Reducer::PartialGrad(vsrc,
                                    data_pooled_.Eval(c * pshape_y_ + py, px)) *
                                    grad_pooled_.Eval(c * pshape_y_ + py, px);
      }
    }

    return val;
  }

 private:
  Plan<SrcExp, DType> data_src_, data_pooled_, grad_pooled_;
  const index_t sshape_y_, pshape_y_, pshape_x_;
  const index_t ksize_y_, ksize_x_;
  const index_t kstride_y_, kstride_x_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_SPATIAL_UNPOOL_H_
