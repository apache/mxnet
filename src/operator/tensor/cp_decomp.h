/*!
 *  Copyright (c) 2014 by Contributors
 */
#ifndef MXNET_OPERATOR_TENSOR_CP_DECOMP_H_
#define MXNET_OPERATOR_TENSOR_CP_DECOMP_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mshadow/tensor.h>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <numeric>
#include <iostream>
#include "./unfold.h"
#include "./cp_decomp_linalg.h"
#include "./broadcast_reduce-inl.h"

namespace mxnet {
namespace op {

using namespace mshadow;
using namespace mshadow::expr;
using namespace mxnet::op::cp_decomp;

template <typename DType>
inline void print1DTensor_(const Tensor<cpu, 1, DType> &t);

template <typename DType>
inline void print2DTensor_(const Tensor<cpu, 2, DType> &t);

template <int order, typename DType>
inline DType CPDecompReconstructionError
  (const Tensor<cpu, order, DType> &t,
  const Tensor<cpu, 1, DType> &eigvals,
  const std::vector<Tensor<cpu, 2, DType> > &factors_T);

template <typename RandomEngine, typename DType>
inline int CPDecompInitFactors
  (std::vector<Tensor<cpu, 2, DType> > factors_T,
  bool orthonormal,
  RandomEngine &generator);

template <typename DType>
inline int CPDecompUpdate
  (Tensor<cpu, 1, DType> eigvals,
  std::vector<Tensor<cpu, 2, DType> > factors,
  const Tensor<cpu, 2, DType> &unfolding,
  int mode,
  std::vector<TensorContainer<cpu, 2, DType> > kr_prod,
  TensorContainer<cpu, 2, DType> hd_prod,
  Stream<cpu> *stream = NULL);

template <typename DType>
inline bool CPDecompConverged
  (const Tensor<cpu, 1, DType> &eigvals,
  const std::vector<Tensor<cpu, 2, DType> > &factors_T,
  const Tensor<cpu, 1, DType> &oldEigvals,
  const std::vector<Tensor<cpu, 2, DType> > &oldFactors_T,
  DType eps);

template <int order, typename DType>
inline int CPDecomp
  (Tensor<cpu, 1, DType> eigvals,
  std::vector<Tensor<cpu, 2, DType> > factors_T,
  const Tensor<cpu, order, DType> &in,
  int k,
  DType eps = 1e-6,
  int max_iter = 100,
  int restarts = 5,
  bool init_orthonormal_factors = true,
  Stream<cpu> *stream = NULL) {
  CHECK_GE(order, 2);

  CHECK_EQ((int) eigvals.size(0), k);
  CHECK_EQ((int) factors_T.size(), order);
  CHECK_GE(k, 1);
  for (int i = 0; i < order; ++i) {
    CHECK_EQ((int) factors_T[i].size(0), k);
    CHECK_EQ(factors_T[i].size(1), in.size(i));
  }
  CHECK_GE(restarts, 1);

  // Return value
  int status;

  // in is unfolded mode-1 tensor
  // Transform it into mode-2, mode-3 tensors
  std::vector<Tensor<cpu, 2, DType> > unfoldings;
  const int tensor_size = in.shape_.Size();
  for (int id_mode = 0; id_mode < order; ++id_mode) {
    unfoldings.emplace_back
      (Shape2(in.size(id_mode), tensor_size / in.size(id_mode)));
    AllocSpace(&unfoldings[id_mode]);

    Unfold(unfoldings[id_mode], in, id_mode);
  }

  // Allocate space for old factor matrices A, B, C, etc,
  // transposed as well, with the same shapes as factors_T
  Tensor<cpu, 1, DType> currEigvals(Shape1(k)), oldEigvals(Shape1(k));
  AllocSpace(&currEigvals);
  AllocSpace(&oldEigvals);

  std::vector<Tensor<cpu, 2, DType> > currFactors_T;
  std::vector<Tensor<cpu, 2, DType> > oldFactors_T;
  for (int id_mode = 0; id_mode < order; ++id_mode) {
    currFactors_T.emplace_back(factors_T[id_mode].shape_);
    oldFactors_T.emplace_back(factors_T[id_mode].shape_);
    AllocSpace(&currFactors_T[id_mode]);
    AllocSpace(&oldFactors_T[id_mode]);
  }

  // The intermediate tensors are reused for efficiency
  // We store the transpose of all intermediate and final
  // Khatri-Rao products for convenience of computation
  //
  // As across the modes, the intermediate Khatri-Rao products
  // will eventually take different shapes, kr_prod_T is
  // first indexed by mode, then by the number of Khatri-Rao
  // products already done
  std::vector<std::vector<TensorContainer<cpu, 2, DType> > > kr_prod_T;
  int kr_length;
  for (int id_mode = 0; id_mode < order; ++id_mode) {
    kr_length = 1;

    std::vector<TensorContainer<cpu, 2, DType> > kr_prod_T_;
    kr_prod_T_.emplace_back(Shape2(k, kr_length));
    kr_prod_T_[0] = 1;

    for (int q = order - 1; q >= 0; --q) {
      if (q == id_mode)
        continue;

      kr_length *= in.size(q);
      kr_prod_T_.emplace_back(Shape2(k, kr_length));
    }

    kr_prod_T.push_back(kr_prod_T_);
  }

  // Hadamard product
  TensorContainer<cpu, 2, DType> hd_prod(Shape2(k, k));

  // Initialise random generator
  std::random_device rnd_device;
  std::mt19937 generator(rnd_device());

  // Starting multi-runs of ALS
  DType reconstructionError = std::numeric_limits<DType>::infinity();
  DType currReconstructionError;
  for (int id_run = 0; id_run < restarts; ++id_run) {
    // Randomly initialise factor matrices
    status = CPDecompInitFactors(currFactors_T, 
      init_orthonormal_factors, generator);
    if (status != 0) {
      std::cerr << "Init error\n";
      continue;
    }

    // ALS
    int iter = 0;
    while (iter < max_iter
        && status == 0
        && (iter == 0 || !CPDecompConverged(currEigvals, currFactors_T,
                                     oldEigvals, oldFactors_T, eps))) {
      Copy(oldEigvals, currEigvals);
      for (int id_mode = 0; id_mode < order; ++id_mode)
        Copy(oldFactors_T[id_mode], currFactors_T[id_mode]);

      for (int id_mode = 0; id_mode < order; ++id_mode) {
        status = CPDecompUpdate
          (currEigvals, currFactors_T,
          unfoldings[id_mode], id_mode,
          kr_prod_T[id_mode], hd_prod,
          stream);
        if (status != 0) {
          std::cerr << "Iter " << iter << " Update error\n";
          break;
        }
      }

      ++iter;
    }

    if (status != 0)
      continue;

    currReconstructionError = CPDecompReconstructionError
      (in, currEigvals, currFactors_T);
    print1DTensor_(currEigvals);
    std::cerr << "Reconstruction error: " << currReconstructionError << "\n";
    if (currReconstructionError < reconstructionError) {
      Copy(eigvals, currEigvals);
      for (int id_mode = 0; id_mode < order; ++id_mode)
        Copy(factors_T[id_mode], currFactors_T[id_mode]);
      reconstructionError = currReconstructionError;
    }
  }

  // Free up space
  for (int id_mode = 0; id_mode < order; ++id_mode) {
    FreeSpace(&unfoldings[id_mode]);
    FreeSpace(&currFactors_T[id_mode]);
    FreeSpace(&oldFactors_T[id_mode]);
  }

  FreeSpace(&currEigvals);
  FreeSpace(&oldEigvals);

  if (reconstructionError < std::numeric_limits<DType>::infinity())
    return 0;
  else
    return 1;
}

template <int order, typename DType>
inline DType CPDecompReconstructionError
  (const Tensor<cpu, order, DType> &t,
  const Tensor<cpu, 1, DType> &eigvals,
  const std::vector<Tensor<cpu, 2, DType> > &factors_T) {
  int k = eigvals.size(0);

  Shape<order> strides = t.shape_;
  strides[order - 1] = t.stride_;

  Shape<order> coord;
  DType sum_sq = 0, delta;
  DType reconstructedElement, c;
  for (int flat_id = 0; flat_id < (int) t.shape_.Size(); ++flat_id) {
    coord = mxnet::op::broadcast::unravel(flat_id, t.shape_);

    reconstructedElement = 0;
    for (int i = 0; i < k; ++i) {
      c = eigvals[i];
      for (int id_mode = 0; id_mode < order; ++id_mode)
        c *= factors_T[id_mode][i][coord[id_mode]];

      reconstructedElement += c;
    }

    delta = t.dptr_[ravel_multi_index(coord, strides)] - reconstructedElement;
    sum_sq += delta * delta;
  }

  return std::sqrt(sum_sq);
}

template <typename RandomEngine, typename DType>
inline int CPDecompInitFactors
  (std::vector<Tensor<cpu, 2, DType> > factors_T,
  bool orthonormal, 
  RandomEngine &generator) {
  int status = 0;

  int order = (int) factors_T.size();
  int k = factors_T[0].size(0);
  for (const auto &mat : factors_T)
    CHECK_EQ((int) mat.size(0), k);

  // TODO(jli05): implement seed for random generator
  std::normal_distribution<DType> normal(0.0, 1.0);

  for (int id_mode = 0; id_mode < order; ++id_mode) {
    for (int i = 0; i < k; ++i) {
      for (int j = 0; j < (int) factors_T[id_mode].size(1); ++j)
        factors_T[id_mode][i][j] = normal(generator);
    }

    if (orthonormal) {
      status = orgqr<cpu, DType>((int) factors_T[id_mode].size(1), k, k,
          factors_T[id_mode].dptr_, factors_T[id_mode].stride_);
      if (status != 0)
        return status;
    }
  }

  return status;
}

template <typename DType>
inline int CPDecompUpdate
  (Tensor<cpu, 1, DType> eigvals,
  std::vector<Tensor<cpu, 2, DType> > factors_T,
  const Tensor<cpu, 2, DType> &unfolding,
  int mode,
  std::vector<TensorContainer<cpu, 2, DType> > kr_prod_T,
  TensorContainer<cpu, 2, DType> hd_prod,
  Stream<cpu> *stream) {
  int order = (int) factors_T.size();
  int k = eigvals.size(0);

  for (auto &m : factors_T)
    CHECK_EQ((int) m.size(0), k);

  // Return value
  int info;

  // Compute khatri-rao product of C\odot B ...
  // and hadamard product of C^T C * B^T B ...
  int kr_length = 1;
  int id_kr_prod = 1;
  int d;

  hd_prod = 1;

  for (int id_mode = order - 1; id_mode >= 0; --id_mode) {
    if (id_mode == mode)
      continue;

    d = factors_T[id_mode].size(1);
    for (int i = 0; i < k; ++i) {
      expr::BLASEngine<cpu, DType>::SetStream
        (kr_prod_T[id_kr_prod][i].stream_);
      expr::BLASEngine<cpu, DType>::ger
        (kr_prod_T[id_kr_prod][i].stream_,
        d, kr_length,
        1,
        factors_T[id_mode][i].dptr_, 1,
        kr_prod_T[id_kr_prod - 1][i].dptr_, 1,
        kr_prod_T[id_kr_prod][i].dptr_, d);
    }
    kr_length *= d;
    ++id_kr_prod;

    hd_prod = hd_prod *
        implicit_dot(factors_T[id_mode], factors_T[id_mode].T());
  }

  TensorContainer<cpu, 2, DType> rhs_T(Shape2(k, unfolding.size(0)));
  rhs_T = implicit_dot(kr_prod_T[order - 1], unfolding.T());

  // In order to compute rhs pinv(hd_prod) we try to solve for X
  // such that
  //
  //     hd_prod X^T = rhs^T
  //
  // and update factors_T[mode] to be X^T

  info = posv<cpu, DType>(k, unfolding.size(0),
      hd_prod.dptr_, hd_prod.stride_,
      rhs_T.dptr_, rhs_T.stride_);
  if (info != 0) {
    return info;
  }
  Copy(factors_T[mode], rhs_T);

  for (int j = 0; j < k; ++j) {
    // Compute the L2-norm of Column j of factors[mode]
    eigvals[j] = nrm2<cpu, DType>(factors_T[mode].size(1),
        factors_T[mode][j].dptr_, 1);

    // Normalise Column j of factors[mode]
    factors_T[mode][j] = factors_T[mode][j] / eigvals[j];
  }

  return 0;
}

template <typename DType>
inline bool CPDecompConverged
  (const Tensor<cpu, 1, DType> &eigvals,
  const std::vector<Tensor<cpu, 2, DType> > &factors_T,
  const Tensor<cpu, 1, DType> &oldEigvals,
  const std::vector<Tensor<cpu, 2, DType> > &oldFactors_T,
  DType eps) {
  int k = eigvals.size(0);

  TensorContainer<cpu, 1, DType> eigval_diff(eigvals.shape_);
  eigval_diff = eigvals - oldEigvals;
  if (nrm2<cpu, DType>(k, eigval_diff.dptr_, 1)
      > eps * nrm2<cpu, DType>(k, oldEigvals.dptr_, 1))
    return false;

  int d;
  for (int p = 0; p < (int) factors_T.size(); ++p) {
    d = factors_T[p].size(1);
    TensorContainer<cpu, 2, DType> factors_diff(factors_T[p].shape_);
    factors_diff = factors_T[p] - oldFactors_T[p];

    for (int i = 0; i < k; ++i) {
      if (nrm2<cpu, DType>(d, factors_diff[i].dptr_, 1)
          > eps * nrm2<cpu, DType>(d, oldFactors_T[p][i].dptr_, 1))
        return false;
    }
  }

  return true;
}


template <typename DType>
inline void print1DTensor_(const Tensor<cpu, 1, DType> &t) {
  for (int i = 0; i < (int) t.size(0); ++i)
    std::cerr << t[i] << " ";
  std::cerr << "\n";
}

template <typename DType>
inline void print2DTensor_(const Tensor<cpu, 2, DType> &t) {
  for (int i = 0; i < (int) t.size(0); ++i) {
    for (int j = 0; j < (int) t.size(1); ++j)
      std::cerr << t[i][j] << " ";
    std::cerr << "\n";
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_CP_DECOMP_H_
