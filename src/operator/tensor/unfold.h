
/*!
 *  Copyright (c) 2014 by Contributors
 */
#ifndef MXNET_OPERATOR_TENSOR_UNFOLD_H_
#define MXNET_OPERATOR_TENSOR_UNFOLD_H_
#include <dmlc/logging.h>
#include <mshadow/tensor.h>
#include <vector>


namespace mxnet {
namespace op {

using namespace mshadow;
using namespace mshadow::expr;

template <typename T>
inline std::vector<std::vector<T> > cart_product
  (const std::vector<std::vector<T> > &v) {
  std::vector<std::vector<T> > s = {{}};
  for (const auto &u : v) {
    std::vector<std::vector<T> > r;
    for (const auto y : u) {
      for (const auto &prod : s) {
        r.push_back(prod);
        r.back().push_back(y);
      }
    }
    s = std::move(r);
  }
  return s;
}

template<int ndim>
MSHADOW_XINLINE int ravel_multi_index
  (const Shape<ndim>& coord, const Shape<ndim>& strides);

template <int order, typename DType>
inline DType TensorAt(const Tensor<cpu, order, DType> &t,
  const std::vector<index_t> &coord);

template <int order, typename DType>
inline void Unfold
  (Tensor<cpu, 2, DType> unfolding,
  const Tensor<cpu, order, DType> &t,
  int mode,
  Stream<cpu> *stream = NULL) {
  // Make array of index value ranges along each dimension
  std::vector<std::vector<index_t> > v;
  for (int id_mode = 0; id_mode < order; ++id_mode) {
    if (id_mode == mode)
      continue;

    std::vector<index_t> u(t.size(id_mode));
    std::iota(u.begin(), u.end(), 0);
    v.push_back(u);
  }

  // Cartesian product of index value ranges along each dimension
  std::vector<std::vector<index_t> > coords = cart_product(v);

  // Unfold
  Shape<order> strides = t.shape_;
  strides[order - 1] = t.stride_;
  
  Shape<order> coord;

  for (index_t i = 0; i < t.size(mode); ++i) {
    for (index_t j = 0; j < (index_t) coords.size(); ++j) {
      std::vector<index_t> coord_ = coords[j];
      coord_.insert(coord_.begin() + mode, i);

      for (int w = 0; w < (int) coord_.size(); ++w)
        coord[w] = coord_[w];

      unfolding[i][j] = t.dptr_[ravel_multi_index(coord, strides)];
      // unfolding[i][j] = TensorAt(t, coord_); 
    }
  }
}

template <int order, typename DType>
inline DType TensorAt(const Tensor<cpu, order, DType> &t,
  const std::vector<index_t> &coord) {
  std::vector<index_t> sub_coord(coord.begin() + 1, coord.end());
  return TensorAt(t[coord[0]], sub_coord);
}

template <>
inline float TensorAt<1, float>(const Tensor<cpu, 1, float> &t,
  const std::vector<index_t> &coord) {
  return t[coord[0]];
}

template <>
inline double TensorAt<1, double>(const Tensor<cpu, 1, double> &t,
  const std::vector<index_t> &coord) {
  return t[coord[0]];
}

template<int ndim>
MSHADOW_XINLINE int ravel_multi_index
  (const Shape<ndim>& coord, const Shape<ndim>& strides) {
  int ret = 0;
  for (int i = 0; i < ndim; ++i) {
    ret = ret * strides[i] + coord[i];
  }
  return ret;
}

}  // op
}  // mxnet

#endif
