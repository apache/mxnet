/*!
 *  Copyright (c) 2016 by Contributors
 * \file transpose.h
 * \brief support for transpose
 * \author Junyuan Xie
 */
#ifndef MSHADOW_EXTENSION_TRANSPOSE_H_
#define MSHADOW_EXTENSION_TRANSPOSE_H_
#include <algorithm>
#include "../extension.h"
namespace mshadow {
namespace expr {
/*!
 * \brief transpose axes of a tensor
 * input: Tensor<Device,dim>: ishape
 * output: Tensor<Device,dimdst> oshape[a1],oshape[a2] = ishape[a2],oshape[a1]
 *
 * \tparam SrcExp type of source expression
 * \tparam DType the type of elements
 * \tparam dimsrc source dimension, assert a1 > a2
 * \tparam m_a1 one dimension to be swapped, encoded by dimsrc - a1
 * \tparam a2 second dimension to be swapped, encoded by a2
 */
template<typename SrcExp, typename DType, int dimsrc>
struct TransposeExExp:
      public MakeTensorExp<TransposeExExp<SrcExp, DType, dimsrc>,
                           SrcExp, dimsrc, DType> {
  /*! \brief source expression */
  const SrcExp &src_;
  const Shape<dimsrc> axes_;
  Shape<dimsrc> dst_in_src_stride_;  // Holds the corresponding stride of the dst axes in src
  index_t src_stride_;
  /*! \brief constructor */
  explicit TransposeExExp(const SrcExp &src, Shape<dimsrc> axes) : src_(src), axes_(axes) {
    Shape<dimsrc> src_shape = ShapeCheck<dimsrc, SrcExp>::Check(src);
    src_stride_ = src_shape[dimsrc - 1];
    Shape<dimsrc> src_stride;
    src_stride[dimsrc-1] = 1;
    for (int i = dimsrc-2; i >= 0; --i) src_stride[i] = src_shape[i+1]*src_stride[i+1];
    for (int i = 0; i < dimsrc; ++i) {
      dst_in_src_stride_[i] = src_stride[axes[i]];
      this->shape_[i] = src_shape[axes[i]];
    }
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
template<typename SrcExp, typename DType, int etype>
inline TransposeExExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
transpose(const Exp<SrcExp, DType, etype> &src, Shape<ExpInfo<SrcExp>::kDim> axes) {
  return TransposeExExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), axes);
}

template<typename SrcExp, typename DType, int dimsrc>
struct Plan<TransposeExExp<SrcExp, DType, dimsrc>, DType> {
 public:
  explicit Plan(const TransposeExExp<SrcExp, DType, dimsrc> &e)
      : src_(MakePlan(e.src_)),
        src_stride_(e.src_stride_),
        dst_in_src_stride_(e.dst_in_src_stride_),
        dst_shape_(e.shape_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    index_t idx = j * dst_in_src_stride_[dimsrc - 1];
    #pragma unroll
    for (int k = dimsrc-2; k >= 0; --k) {
      idx += (i % dst_shape_[k]) * dst_in_src_stride_[k];
      i /= dst_shape_[k];
    }
    return src_.Eval(idx/src_stride_, idx%src_stride_);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t src_stride_;
  const Shape<dimsrc> dst_in_src_stride_, dst_shape_;
};

/*!
 * \brief transform contiguous indices of the source tensor to indices of the transposed tensor.
 * input: Tensor<Device, k>: ishape
 * output: Tensor<Device, k>: oshape = ishape
 *
 * \tparam SrcExp type of source expression
 * \tparam DType the type of elements
 * \tparam dimsrc source dimension
 * \tparam etype source type
 */
template<typename SrcExp, typename DType, int dimsrc, int etype>
struct TransposeIndicesExp:
      public Exp<TransposeIndicesExp<SrcExp, DType, dimsrc, etype>, DType, etype> {
  /*! \brief source expression */
  const SrcExp &src_indices_;  // Expression of the source indices
  Shape<dimsrc> src_shape_;  // Holds the corresponding stride of the source axes in dst
  const Shape<dimsrc> axes_;  // The transpose axes
  Shape<dimsrc> src_in_dst_stride_;  // Holds the corresponding stride of the source axes in dst
  /*! \brief constructor */
  explicit TransposeIndicesExp(const SrcExp &src_indices,
                               Shape<dimsrc> src_shape,
                               Shape<dimsrc> axes) : src_indices_(src_indices),
                                                     src_shape_(src_shape), axes_(axes) {
    Shape<dimsrc> dst_shape_;
    Shape<dimsrc> dst_stride_;
    bool axes_checking_flag[dimsrc] = { 0 };
    for (int i = 0; i < dimsrc; ++i) {
      CHECK_LT(static_cast<int>(axes[i]), dimsrc)
        << "Invalid axes input! All elements of axes must be between 0 and " << dimsrc
        << ", find axes=" << axes;
      dst_shape_[i] = src_shape[axes[i]];
      axes_checking_flag[axes[i]] = true;
    }
    // check if the input axes is valid
    for (int i = 0; i < dimsrc; ++i) {
      CHECK_EQ(axes_checking_flag[i], true)
        << "Invalid axes input! All elements of axes must be between 0 and " << dimsrc
        << ", find axes=" << axes;
    }
    dst_stride_[dimsrc - 1] = 1;
    for (int i = dimsrc - 2; i >= 0; --i) dst_stride_[i] = dst_shape_[i+1] * dst_stride_[i+1];
    for (int i = 0; i < dimsrc; ++i) {
      src_in_dst_stride_[axes[i]] = dst_stride_[i];
    }
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
template<typename SrcExp, typename DType, int dimsrc, int etype>
inline TransposeIndicesExp<SrcExp, DType, dimsrc, etype>
transpose_indices(const Exp<SrcExp, DType, etype> &src_indices,
                  Shape<dimsrc> src_shape,
                  Shape<dimsrc> axes) {
  return TransposeIndicesExp<SrcExp, DType, dimsrc, etype>(src_indices.self(), src_shape, axes);
}

template<typename SrcExp, typename DType, int dimsrc, int etype>
struct Plan<TransposeIndicesExp<SrcExp, DType, dimsrc, etype>, DType> {
 public:
  explicit Plan(const TransposeIndicesExp<SrcExp, DType, dimsrc, etype> &e)
      : src_indices_(MakePlan(e.src_indices_)),
        src_in_dst_stride_(e.src_in_dst_stride_),
        src_shape_(e.src_shape_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    index_t src_idx = static_cast<index_t>(src_indices_.Eval(i, j));
    index_t dst_idx = 0;
    #pragma unroll
    for (int k = dimsrc - 1; k >= 0; --k) {
      dst_idx += (src_idx % src_shape_[k]) * src_in_dst_stride_[k];
      src_idx /= src_shape_[k];
    }
    return static_cast<DType>(dst_idx);
  }

 private:
  Plan<SrcExp, DType> src_indices_;
  const Shape<dimsrc> src_in_dst_stride_, src_shape_;
};

//----------------------
// Execution plan
//----------------------
/*! \brief make expression */
template<typename SrcExp, typename DType, int dimsrc, int etype>
inline Plan<TransposeIndicesExp<SrcExp, DType, dimsrc, etype>, DType>
MakePlan(const TransposeIndicesExp<SrcExp, DType, dimsrc, etype> &e) {
  return Plan<TransposeIndicesExp<SrcExp, DType, dimsrc, etype>, DType>(e);
}

template<int dim, typename SrcExp, typename DType, int dimsrc, int etype>
struct ShapeCheck<dim, TransposeIndicesExp<SrcExp, DType, dimsrc, etype> > {
  inline static Shape<dim>
  Check(const TransposeIndicesExp<SrcExp, DType, dimsrc, etype> &t) {
    Shape<dim> s = ShapeCheck<dim, SrcExp>::Check(t.src_indices_);
    return s;
  }
};

template<typename SrcExp, typename DType, int dimsrc, int etype>
struct ExpInfo<TransposeIndicesExp<SrcExp, DType, dimsrc, etype> > {
  static const int kDim = ExpInfo<SrcExp>::kDim;
  static const int kDevMask = ExpInfo<SrcExp>::kDevMask;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_TRANSPOSE_H_
