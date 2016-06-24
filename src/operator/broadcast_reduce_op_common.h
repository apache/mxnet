/*!
* Copyright (c) 2016 by Contributors
* \file broadcast_reduce_op_common.h
* \brief common function used for broadcasting and reducing
* \author Xingjian Shi
*/
#ifndef MXNET_OPERATOR_BROADCAST_REDUCE_OP_COMMON_H_
#define MXNET_OPERATOR_BROADCAST_REDUCE_OP_COMMON_H_
#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <vector>
#include <set>

namespace mxnet {
namespace op {
/*!
* \brief Sort the given axes and removes the duplicate keys to get a vector
* \param param_axis the input axis
* \param max_ndim the maximum ndim
*/
inline std::vector<index_t> ParseAxes_(const TShape& param_axis, index_t max_ndim) {
  std::set<index_t> axes_set_;
  std::vector<index_t> axes;
  for (index_t i = 0; i < param_axis.ndim(); i++) {
    CHECK(param_axis[i] < max_ndim) << "axes must be within the range, ndim of the source="
      << max_ndim << "axis=" << param_axis;
    CHECK_EQ(axes_set_.find(param_axis[i]), axes_set_.end())
      << "Duplicate value in 'axis', received:" << param_axis;
    axes_set_.insert(param_axis[i]);
  }
  for (std::set<index_t>::iterator it = axes_set_.begin(); it != axes_set_.end(); ++it) {
    axes.push_back(*it);
  }
  return axes;
}

/*!
* \brief Check if the axes are continuous + get reducing size. E.g (1, 3) -> false, (1,2,3) -> true
* \param is_contiguous_axes whether the axes is contiguous
* \param reducing_size product of source shape in the given axes
* \param axes
* \param src_shape shape of the source tensor
*/
inline void CheckContiguousAxes_(bool *is_contiguous_axes, index_t *reducing_size,
  const mshadow::TShape &axes, const mshadow::TShape &src_shape) {
  *is_contiguous_axes = true;
  *reducing_size = 1;
  for (index_t i = 0; i < axes.ndim(); ++i) {
    *reducing_size *= src_shape[axes[i]];
    if (i > 0) {
      *is_contiguous_axes = *is_contiguous_axes && (axes[i] == (axes[i - 1] + 1));
      CHECK(axes[i - 1] < axes[i]) << "axes must be in increasing order, received axes=" << axes;
    }
  }
}

template<int dimsrc>
inline void CheckContiguousAxes_(bool *is_contiguous_axes, index_t *reducing_size,
  const mshadow::TShape &axes, const mshadow::Shape<dimsrc> &src_shape) {
  CheckContiguousAxes_(is_contiguous_axes, reducing_size, axes,
    TShape(src_shape.shape_, src_shape.shape_ + dimsrc));
}

inline TShape GetBroadcastingAxes_(const mshadow::TShape &src_shape,
  const mshadow::TShape &target_shape) {
  std::vector<index_t> axes_vec;
  CHECK_EQ(target_shape.ndim(), src_shape.ndim());
  for (index_t i = 0; i < src_shape.ndim(); ++i) {
    if (src_shape[i] != target_shape[i]) {
      CHECK_EQ(src_shape[i], 1) << "broadcastsing axis must have size 1, received src_shape="
        << src_shape << " target_shape=" << target_shape;
      axes_vec.push_back(i);
    }
  }
  TShape axes = TShape(axes_vec.begin(), axes_vec.end());
  return axes;
}

/*!
* \brief a reduce over multiple axes and assign to the output tensor.
* \param out output tensor, must have dim 1
* \param src the source expression
* \param axes the given axes, should be in increasing order
* \tparam Reducer type of the reducing operation
* \tparam xpu
* \tparam SrcExp the src expression template
* \tparam etype type of expression
*/
template<typename Reducer, typename xpu, typename SrcExp, typename DType>
void ReduceAxesAssign(mshadow::Tensor<xpu, 1, DType> out, const OpReqType req,
  const TShape &axes, const SrcExp &src_) {
  using namespace mshadow;
  using namespace mshadow::expr;
  static const int dimsrc = ExpInfo<SrcExp>::kDim;
  CHECK(axes.ndim() <= dimsrc);
  Shape<dimsrc> src_shape = ShapeCheck<dimsrc, SrcExp>::Check(src_);

  // 1. Check if the axes has size 0, if so, no reducing is needed.
  if (0 == axes.ndim()) {
    ASSIGN_DISPATCH(out, req, reshape(src_, Shape1(src_shape.ProdShape(0, dimsrc))));
    return;
  }

  // 2. Check if we want to reduce over contiguous axes and get the reducing size.
  //  e.g. (1,2,3) --> contiguous, (1,3) --> noncontiguous
  bool is_contiguous_axes = true;
  index_t reducing_size = 1;
  CheckContiguousAxes_(&is_contiguous_axes, &reducing_size, axes, src_shape);

  // 3. For contiguous axes, we can always reshape them to (leading, reducing_size, trailing)
  //  and we can then simplify the combination of mshadow symbols.
  if (is_contiguous_axes) {
    index_t leading = 1;
    index_t trailing = 1;
    for (index_t i = 0; i < dimsrc; ++i) {
      if (i < axes[0]) {
        leading *= src_shape[i];
      } else if (i > axes[axes.ndim() - 1]) {
        trailing *= src_shape[i];
      }
    }
    if (1 == leading) {
      ASSIGN_DISPATCH(out, req,
        (reduce_except_dim<1, Reducer>(reshape(src_, Shape2(reducing_size, trailing)))));
    } else {
      ASSIGN_DISPATCH(out, req, (reduce_except_dim<1, Reducer>(
        reshape(swapaxis<1, 0>(reshape(src_, Shape3(leading, reducing_size, trailing))),
        Shape2(reducing_size, leading * trailing)))));
    }
    return;
  }
  // 4. For non-contiguous axes, we need to push axes to the front of the shape vector then reduce.
  //   E.g axes = (1, 2), dim = 6 => transpose_shape = (1, 2, 0, 3, 4, 5)
  Shape<dimsrc> transpose_shape = src_shape;
  index_t remaining_size = 1;
  for (index_t i = 0; i < axes.ndim(); ++i) {
    transpose_shape[i] = axes[i];
    if (i > 0) {
      for (index_t j = axes[i - 1] + 1; j < axes[i]; ++j) {
        transpose_shape[axes.ndim() - i + j] = j;
        remaining_size *= src_shape[j];
      }
    }
    if (axes.ndim() - 1 == i) {
      for (index_t j = axes[axes.ndim() - 1] + 1; j < dimsrc; ++j) {
        transpose_shape[j] = j;
        remaining_size *= src_shape[j];
      }
    }
    if (0 == i) {
      for (index_t j = 0; j < axes[0]; ++j) {
        transpose_shape[axes.ndim() - i + j] = j;
        remaining_size *= src_shape[j];
      }
    }
  }
  ASSIGN_DISPATCH(out, req,
    (reduce_except_dim<1, Reducer>(reshape(transpose(src_, transpose_shape),
    Shape2(reducing_size, remaining_size)))));
}

/*!
* \brief a reduce to the given shape and assign to the output tensor.
* \param out output tensor, must have dim 1
* \param src the source expression
* \param target_shape shape of the target tensor, must have size 1 for the reduction axes
* \tparam Reducer type of the reducing operation
* \tparam xpu
* \tparam SrcExp the src expression template
* \tparam etype type of expression
*/
template<typename Reducer, typename xpu, typename SrcExp, typename DType>
void ReduceToAssign(mshadow::Tensor<xpu, 1, DType> out, const OpReqType req,
  const TShape &target_shape, const SrcExp &src_) {
  using namespace mshadow;
  using namespace mshadow::expr;
  static const int dimsrc = ExpInfo<SrcExp>::kDim;
  Shape<dimsrc> src_shape = ShapeCheck<dimsrc, SrcExp>::Check(src_);
  TShape axes = GetBroadcastingAxes_(target_shape,
    TShape(src_shape.shape_, src_shape.shape_ + dimsrc));
  ReduceAxesAssign<Reducer>(out, req, axes, src_);
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BROADCAST_REDUCE_OP_COMMON_H_
