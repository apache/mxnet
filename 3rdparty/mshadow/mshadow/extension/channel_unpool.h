/*!
 *  Copyright (c) 2014 by Contributors
 * \file channel_pool.h
 * \brief support for chpool
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_CHANNEL_UNPOOL_H_
#define MSHADOW_EXTENSION_CHANNEL_UNPOOL_H_
#include <algorithm>
#include "../extension.h"
namespace mshadow {
namespace expr {
/*!
 * \brief channel pooling expression, do reduction over (local nearby) channels,
 *        used to implement local response normalization
 * \tparam Reducer reduction method during pooling
 * \tparam SrcExp source expression to be pooled from
 * \tparam DType the type of elements
 * \tparam srcdim dimension of src
 */
template<typename Reducer, typename SrcExp, typename DType, int srcdim>
struct ChannelUnpoolingExp:
      public MakeTensorExp<ChannelUnpoolingExp<Reducer, SrcExp, DType, srcdim>,
                           SrcExp, srcdim, DType> {
  /*! \brief source input, corresponds to src in pooling */
  const SrcExp &data_src_;
  /*! \brief result of pooled data, corresponds to result of pooling */
  const SrcExp &data_pooled_;
  /*! \brief gradient data of pooled part, to be propgate down */
  const SrcExp &grad_pooled_;
  /*! \brief channel of pooled expression */
  index_t pchannel_;
  /*! \brief kernel size in height */
  index_t nsize_;
  /*! \brief kernel size in width */
  index_t kstride_;
  /*! \brief pad */
  index_t pad_;
  /*! \brief constructor */
  ChannelUnpoolingExp(const SrcExp &data_src,
               const SrcExp &data_pooled,
               const SrcExp &grad_pooled,
               index_t nsize, index_t kstride, index_t pad)
      : data_src_(data_src), data_pooled_(data_pooled),
        grad_pooled_(grad_pooled),
        nsize_(nsize), kstride_(kstride), pad_(pad) {
    Shape<srcdim> pshape = ShapeCheck<srcdim, SrcExp>::Check(grad_pooled);
    typedef ShapeCheck<srcdim, SrcExp> ShapeCheckSrcDimSrcExp;
    CHECK_EQ(pshape, ShapeCheckSrcDimSrcExp::Check(data_pooled))
      << "ChannelUnPoolingExp: data and grad shape mismatch";
    Shape<srcdim> sshape = ShapeCheck<srcdim, SrcExp>::Check(data_src);
    for (int k = 0; k < srcdim; ++k) {
      if (k == 1) {
        continue;
      }
      CHECK_EQ(pshape[k], sshape[k])
        << "ChannelUnPoolingExp: pooled tensor and src tensor shape mismatch"
        << pshape[k]
        << " vs "
        << sshape[k];
    }
    pchannel_ = pshape[1];
    this->shape_ = sshape;
  }
};
/*!
 * \brief  channel unpooling, do unroll over (local nearby) channels
 * \param src source data
 * \param nsize neighbor size
 * \param stride stride of the pooling
 * \param pad number of padding at each side
 * \return expression of pooled result
 * \tparam Reducer reducer type
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam etype type of expression
 */
template<typename Reducer, typename SrcExp, typename DType, int etype>
inline ChannelUnpoolingExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
ch_unpool(const Exp<SrcExp, DType, etype> &data_src,
       const Exp<SrcExp, DType, etype> &data_pooled,
       const Exp<SrcExp, DType, etype> &grad_pooled,
      index_t nsize, index_t stride, index_t pad) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 3>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return ChannelUnpoolingExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
        (data_src.self(), data_pooled.self(), grad_pooled.self(), nsize, stride, pad);
}

template<typename Reducer, typename SrcExp, typename DType, int etype>
inline ChannelUnpoolingExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
ch_unpool(const Exp<SrcExp, DType, etype> &data_src,
       const Exp<SrcExp, DType, etype> &data_pooled,
       const Exp<SrcExp, DType, etype> &grad_pooled, index_t nsize) {
  return ch_unpool(data_src, data_pooled, grad_pooled, nsize, 1, nsize / 2);
}


//----------------------
// Execution plan
//----------------------
template<typename Reducer, typename SrcExp, typename DType, int srcdim>
struct Plan<ChannelUnpoolingExp<Reducer, SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const ChannelUnpoolingExp<Reducer, SrcExp, DType, srcdim> &e)
      : data_src_(e.data_src_), data_pooled_(e.data_pooled_),
        grad_pooled_(e.grad_pooled_), channel_(e.shape_[srcdim - 3]),
        height_(e.shape_[srcdim - 2]), pchannel_(e.pchannel_),
        hnsize_(e.nsize_), stride_(e.kstride_), pad_(e.pad_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    using namespace std;
    const DType vsrc = data_src_.Eval(i, j);
    const index_t y = i % height_;
    i /= height_;
    const index_t c = i % channel_;
    const index_t n = i / channel_;
    const index_t x = j;
    const index_t cstart = c < hnsize_ - pad_ ? 0
                        : (c - (hnsize_ - pad_) + stride_) / stride_;
    const index_t cend = min((c + pad_ + stride_) / stride_, channel_);
    DType val = static_cast<DType>(0);
    for (index_t cc = cstart; cc < cend; ++cc) {
      val += Reducer::PartialGrad(vsrc,
                                  data_pooled_.Eval((n * pchannel_ + cc) * height_ + y, x)) *
                                  grad_pooled_.Eval((n * pchannel_ + cc) * height_ + y, x);
    }
    return val;
  }

 private:
  Plan<SrcExp, DType> data_src_, data_pooled_, grad_pooled_;
  const index_t channel_, height_, pchannel_, hnsize_, stride_, pad_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_CHANNEL_UNPOOL_H_

