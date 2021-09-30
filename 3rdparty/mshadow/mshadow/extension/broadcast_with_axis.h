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
 * \file broadcast_with_axis.h
 * \brief
 * \author Junyuan Xie, Xingjian Shi
*/
#ifndef MSHADOW_EXTENSION_BROADCAST_WITH_AXIS_H_
#define MSHADOW_EXTENSION_BROADCAST_WITH_AXIS_H_

#include <vector>
#include "../extension.h"

namespace mshadow {
namespace expr {

  /*!
  * \brief Broadcasting the tensor in the given axis. If keepdim is off, insert the broadcasting dim after axis. Otherwise broadcasting axis.
  * \tparam SrcExp source expression
  * \tparam DType  data type
  * \tparam dimsrc source dimension
  * \tparam dimdst destination dimension
  */
template<typename SrcExp, typename DType, int dimsrc, int dimdst>
struct BroadcastWithAxisExp:
    public MakeTensorExp<BroadcastWithAxisExp<SrcExp, DType, dimsrc, dimdst>,
                         SrcExp, dimdst, DType> {
  /*! \brief data oprand */
  const SrcExp &src_;
  /*! \brief size of the last dimension of dst */
  index_t dst_last_;
  /*! \brief product of the dimensions after the broadcasting axis */
  index_t trailing_;
  /*! \brief new dimension of the broadcasting axis*/
  index_t size_;
  /*! \brief size of the last dimension of src*/
  index_t last_;
  /*! constructor */
  BroadcastWithAxisExp(const SrcExp &src, const int axis, const index_t size)
    : src_(src), size_(size) {
    bool keepdim = (dimsrc == dimdst);
    Shape<dimsrc> src_shape = ShapeCheck<dimsrc, SrcExp>::Check(src_);
    this->trailing_ = 1;

    if (!keepdim) {
      CHECK(dimsrc > axis && axis >= -1) << "broadcast axis (no keepdim) out of bound, "  <<
        "axis must be between -1 and" << dimsrc - 1 << ", given=" << axis << ".";
      for (int i = 0; i <= axis; ++i) {
        this->shape_[i] = src_shape[i];
      }
      this->shape_[axis + 1] = size_;
      for (int i = axis + 1; i < dimsrc; ++i) {
        this->trailing_ *= src_shape[i];
        this->shape_[i + 1] = src_shape[i];
      }
    } else {
      CHECK(dimdst > axis && axis >= 0) << "broadcast axis (keepdim) out of bound, " <<
        "axis must be between 0 and" << dimdst - 1 << ", given=" << axis << ".";
      CHECK_EQ(src_shape[axis], 1U) << "Size of the dimension of the broadcasting axis must be 1" <<
        " when keepdim is on, src_shape[" << axis << "]=" << src_shape[axis] << ".";
      for (int i = 0; i <= axis - 1; ++i) {
        this->shape_[i] = src_shape[i];
      }
      this->shape_[axis] = size_;
      for (int i = axis + 1; i < dimdst; ++i) {
        this->trailing_ *= src_shape[i];
        this->shape_[i] = src_shape[i];
      }
    }

    this->last_ = src_shape[dimsrc - 1];
    this->dst_last_ = this->shape_[dimdst - 1];
  }
};  // struct BroadcastWithAxisExp

/*!
 * \brief Broadcasting the tensor after given axis.
 * \tparam SrcExp source expression
 * \tparam DType data type
 * \tparam etype type of the expression
 */
template<typename SrcExp, typename DType, int etype>
inline BroadcastWithAxisExp<SrcExp, DType, ExpInfo<SrcExp>::kDim,
  ExpInfo<SrcExp>::kDim + 1>
broadcast_with_axis(const Exp<SrcExp, DType, etype> &src, const int axis, const index_t size) {
  return BroadcastWithAxisExp<SrcExp, DType, ExpInfo<SrcExp>::kDim,
    ExpInfo<SrcExp>::kDim + 1>(src.self(), axis, size);
}

/*!
* \brief Broadcasting the tensor in the given axis (keepdim turned on)
* \tparam SrcExp source expression
* \tparam DType data type
* \tparam etype type of the expression
*/
template<typename SrcExp, typename DType, int etype>
inline BroadcastWithAxisExp<SrcExp, DType, ExpInfo<SrcExp>::kDim,
  ExpInfo<SrcExp>::kDim>
  broadcast_keepdim(const Exp<SrcExp, DType, etype> &src, const int axis, const index_t size) {
  return BroadcastWithAxisExp<SrcExp, DType, ExpInfo<SrcExp>::kDim,
    ExpInfo<SrcExp>::kDim>(src.self(), axis, size);
}

/*!
* \brief Broadcasting the tensor in multiple axes. The dimension of the source tensor
         in the given axes must be 1.
* \tparam SrcExp source expression
* \tparam DType  data type
* \tparam dimsrc source dimension
* \tparam axesnum number of broadcasting dimensions
*/
template<typename SrcExp, typename DType, int dimsrc>
struct BroadcastWithMultiAxesExp :
      public MakeTensorExp<BroadcastWithMultiAxesExp<SrcExp, DType, dimsrc>,
  SrcExp, dimsrc, DType> {
  /*! \brief data oprand */
  const SrcExp &src_;
  /*! \brief size of the last dimension of dst */
  index_t dst_last_;
  /*! \brief number of broadcasting axes*/
  index_t axesnum_;
  /*! \brief product of the dimensions after the broadcasting axses */
  Shape<dimsrc> trailings_;
  /*! \brief new dimension of the broadcasting axes*/
  Shape<dimsrc> sizes_;
  /*! \brief size of the last dimension of src*/
  index_t last_;
  /*! constructor */
  template<typename TShape>
  BroadcastWithMultiAxesExp(const SrcExp &src, const TShape& axes, const TShape& sizes)
    : src_(src) {
    Shape<dimsrc> src_shape = ShapeCheck<dimsrc, SrcExp>::Check(src_);
    CHECK(axes.ndim() == sizes.ndim()) << "ndim of axes and sizes must be equal.";
    this->axesnum_ = axes.ndim();
    CHECK(this->axesnum_ <= dimsrc) << "Number of broadcasting axes must be smaller than"
      "the source ndim, number of axes=" << this->axesnum_ << " dimsrc=" << dimsrc;
    for (index_t i = 0; i < this->axesnum_; i++) {
      CHECK(dimsrc > axes[i]) << "broadcast axis (keepdim) out of bound, " <<
        "all axes must be between 0 and" << dimsrc - 1 << ", given axes[" << i << "] = " << axes[i]
        << ".";
      CHECK_EQ(src_shape[axes[i]], 1U) << "Size of the dimension of the broadcasting axis must be 1"
        << ", src_shape[" << axes[i] << "]=" << src_shape[axes[i]] << ".";
      if (i < this->axesnum_ - 1) {
        CHECK(axes[i] < axes[i + 1]) << "The given axes must be in increasing order.";
      }
    }
    for (index_t i = 0; i < dimsrc; i++) {
      this->shape_[i] = src_shape[i];
      this->sizes_[i] = 1;
      this->trailings_[i] = 1;
    }
    for (index_t i = 0; i < this->axesnum_; i++) {
      this->shape_[axes[i]] = sizes[i];
      this->sizes_[i] = sizes[i];
    }
    for (index_t i = 0; i < this->axesnum_; i++) {
      this->trailings_[i] = 1;
      for (index_t j = axes[i] + 1; j < dimsrc; ++j) {
        this->trailings_[i] *= this->shape_[j];
      }
    }
    this->last_ = src_shape[dimsrc - 1];
    this->dst_last_ = this->shape_[dimsrc - 1];
  }
};  // struct BroadcastWithMultiAxesExp

/*!
* \brief Broadcasting the tensor in the given axis (keepdim turned on)
* \param src source
* \param axes broadcasting axes
* \param sizes sizes of the broadcasting axes
* \tparam SrcExp source expression
* \tparam DType data type
* \tparam etype type of the expression
* \tparam TShape the flexible shape type
*/
template<typename SrcExp, typename DType, int etype, typename TShape>
inline BroadcastWithMultiAxesExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
broadcast_multi_axes(const Exp<SrcExp, DType, etype> &src,
const TShape &axes, const TShape &sizes) {
  return BroadcastWithMultiAxesExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), axes, sizes);
}

/*!
* \brief Broadcasting the tensor to the target shape,
         dimension of different sizes must be 1 in the original tensor.
* \param src source
* \param target_shape shape of the target broadcasting tensor
* \tparam SrcExp source expression
* \tparam DType data type
* \tparam etype type of the expression
* \tparam TShape the flexible shape type
*/
template<typename SrcExp, typename DType, int etype, typename TShape>
inline BroadcastWithMultiAxesExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
broadcast_to(const Exp<SrcExp, DType, etype> &src, const TShape &target_shape) {
  static const size_t dimsrc = ExpInfo<SrcExp>::kDim;
  CHECK_EQ(target_shape.ndim(), dimsrc);
  std::vector<index_t> axes_vec, sizes_vec;
  Shape<dimsrc> src_shape = ShapeCheck<dimsrc, SrcExp>::Check(src.self());
  for (size_t i = 0; i < dimsrc; ++i) {
    if (src_shape[i] != target_shape[i]) {
      CHECK_EQ(src_shape[i], 1U) << "broadcasting axis must have size 1, received shape="
        << src_shape << " target_shape=" << target_shape;
      axes_vec.push_back(i);
      sizes_vec.push_back(target_shape[i]);
    }
  }
  TShape axes = TShape(axes_vec.begin(), axes_vec.end());
  TShape sizes = TShape(sizes_vec.begin(), sizes_vec.end());
  return BroadcastWithMultiAxesExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), axes, sizes);
}

//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int dimsrc, int dimdst>
struct Plan<BroadcastWithAxisExp<SrcExp, DType, dimsrc, dimdst>, DType> {
 public:
  explicit Plan(const BroadcastWithAxisExp<SrcExp, DType, dimsrc, dimdst> &e)
       : src_(MakePlan(e.src_)), dst_last_(e.dst_last_),
         trailing_(e.trailing_), size_(e.size_), last_(e.last_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    index_t x = (i * dst_last_ + j) / trailing_ / size_;
    index_t y = (i * dst_last_ + j) % trailing_;
    index_t z = x * trailing_ + y;
    return src_.Eval(z / last_, z % last_);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t dst_last_, trailing_, size_, last_;
};

template<typename SrcExp, typename DType, int dimsrc>
struct Plan<BroadcastWithMultiAxesExp<SrcExp, DType, dimsrc>, DType> {
 public:
  explicit Plan(const BroadcastWithMultiAxesExp<SrcExp, DType, dimsrc> &e)
    : src_(MakePlan(e.src_)), dst_last_(e.dst_last_), last_(e.last_), axesnum_(e.axesnum_),
    trailings_(e.trailings_), sizes_(e.sizes_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    index_t indx = i * dst_last_ + j;
    for (index_t p = 0; p < dimsrc; ++p) {
      if (p >= axesnum_) {
        break;
      }
      indx = (indx / trailings_[p] / sizes_[p]) * trailings_[p] + (indx % trailings_[p]);
    }
    return src_.Eval(indx / last_, indx % last_);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t dst_last_, last_, axesnum_;
  const Shape<dimsrc> trailings_, sizes_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_BROADCAST_WITH_AXIS_H_
