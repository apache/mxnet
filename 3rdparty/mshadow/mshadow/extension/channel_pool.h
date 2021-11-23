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
 * \file channel_pool.h
 * \brief support for chpool
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_CHANNEL_POOL_H_
#define MSHADOW_EXTENSION_CHANNEL_POOL_H_
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
struct ChannelPoolingExp:
      public MakeTensorExp<ChannelPoolingExp<Reducer, SrcExp, DType, srcdim>,
                           SrcExp, srcdim, DType> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief neighbor size */
  index_t nsize_;
  /*! \brief stride of pooling */
  index_t stride_;
  /*! \brief pad of pooling of each side */
  index_t pad_;
  index_t src_channel_;
  /*! \brief constructor */
  ChannelPoolingExp(const SrcExp &src, index_t nsize, index_t stride, index_t pad)
      : src_(src), nsize_(nsize), stride_(stride), pad_(pad) {
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
    this->src_channel_ = this->shape_[srcdim - 3];
    CHECK_GE(this->shape_[srcdim - 3], nsize_)
      << "chpool: local size must be smaller than nchannels";
    this->shape_[srcdim - 3] = (this->src_channel_ - nsize + pad * 2 + 1) / stride;
  }
};
/*!
 * \brief  channel pooling, do reduction over (local nearby) channels,
 *         used to implement local response normalization
 * \param src source data
 * \param nsize neighbor size
 * \return expression of pooled result
 * \tparam Reducer reducer type
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam etype type of expression
 */
template<typename Reducer, typename SrcExp, typename DType, int etype>
inline ChannelPoolingExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
chpool(const Exp<SrcExp, DType, etype> &src, index_t nsize) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 3>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  CHECK_EQ(nsize % 2, 1U) << "chpool: if no pad is specified, local size must be odd";
  return ChannelPoolingExp<Reducer, SrcExp,
                           DType, ExpInfo<SrcExp>::kDim>(src.self(), nsize, 1, nsize / 2);
}

template<typename Reducer, typename SrcExp, typename DType, int etype>
inline ChannelPoolingExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
chpool(const Exp<SrcExp, DType, etype> &src, index_t nsize, index_t stride, index_t pad) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 3>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return ChannelPoolingExp<Reducer, SrcExp,
                           DType, ExpInfo<SrcExp>::kDim>(src.self(), nsize, stride, pad);
}

//----------------------
// Execution plan
//----------------------
template<typename Reducer, typename SrcExp, typename DType, int srcdim>
struct Plan<ChannelPoolingExp<Reducer, SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const ChannelPoolingExp<Reducer, SrcExp, DType, srcdim> &e)
      : src_(MakePlan(e.src_)), channel_(e.shape_[srcdim - 3]),
        height_(e.shape_[srcdim - 2]), width_(e.shape_[srcdim - 1]),
        hnsize_(e.nsize_), stride_(e.stride_), pad_(e.pad_),
        src_channel_(e.src_channel_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    using namespace std;
    const index_t y = i % height_;
    i /= height_;
    const index_t c = i % channel_;
    const index_t n = i / channel_;
    const index_t x = j;
    const index_t cstart = c * stride_ < pad_ ? 0  : c * stride_ - pad_;
    const index_t cend   = min(c * stride_ - pad_ + hnsize_, channel_);
    DType res; Reducer::SetInitValue(res);
    for (index_t cc = cstart; cc < cend; ++cc) {
      Reducer::Reduce(res, src_.Eval((n * src_channel_ + cc) * height_ + y, x));
    }
    return res;
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t channel_, height_, width_, hnsize_, stride_, pad_, src_channel_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_CHANNEL_POOL_H_

