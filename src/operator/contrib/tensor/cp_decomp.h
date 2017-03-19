/*!
 *  Copyright (c) 2014 by Contributors
 *  \file cp_decomp.h
 *  \brief Core function performing CP Decomposition
 *  \author Jencir Lee
 */
#ifndef MXNET_OPERATOR_CONTRIB_TENSOR_CP_DECOMP_H_
#define MXNET_OPERATOR_CONTRIB_TENSOR_CP_DECOMP_H_
#include <mshadow/tensor.h>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <utility>
#include <numeric>
#include <iostream>
#include "./unfold.h"
#include "./cp_decomp_linalg.h"
#include "../../tensor/broadcast_reduce-inl.h"

namespace mxnet {
namespace op {

using namespace mshadow;
using namespace mshadow::expr;
using namespace mxnet::op::cp_decomp;

template <typename DType>
inline void print1DTensor_(const Tensor<cpu, 1, DType> &t);

template <typename DType>
inline void print2DTensor_(const Tensor<cpu, 2, DType> &t);

/*!
 * \brief Evaluates the reconstruction error of the CP Decomposition
 *
 * Reconstruction error = \lVert t - [eigvals; factors] \rVert_2
 *
 * \param t input tensor
 * \param eigvals eigen-value vector of the CP Decomposition
 * \param factors_T array of transposed factor matrices
 * \return the reconstruction error
 */
template <int order, typename DType>
inline DType CPDecompReconstructionError
  (const Tensor<cpu, order, DType> &t,
  const Tensor<cpu, 1, DType> &eigvals,
  const std::vector<Tensor<cpu, 2, DType> > &factors_T);

/*!
 * \brief Randomly initialise the transposed factor matrices for the CP Decomposition
 *
 * All factor matrices will be filled with random numbers from the standard Gaussian distribution. Optionally they could be orthogonalised.
 * This function is called by CPDecomp().
 *
 * \param factors_T array of transposed factor matrices to be initialised
 * \param orthonormal whether to orthogonalise the factor matrices
 * \param generator C++ random generator from <random>
 * \return 0 for success, non-zero otherwise
 */
template <typename RandomEngine, typename DType>
inline int CPDecompInitFactors
  (std::vector<Tensor<cpu, 2, DType> > factors_T,
  bool orthonormal,
  RandomEngine *generator);

/*!
 * \brief Update one factor matrix during one step of the ALS algorithm for the CP Decomposition
 *
 * Given the unfolded tensor and all the remaining factor matrices, this function solves a least-squre problem
 *
 *    min \lVert unfolding - [I; factors] \rVert_2
 *
 * then normalise all the columns of the newly updated factor matrix and record the norms of the columns into eigvals.
 *
 * This function is called by CPDecomp().
 *
 * \param eigvals eigen-value vector to be updated
 * \param unfolding unfolded tensor along the specified mode
 * \param mode the dimension for which we update the factor matrix
 * \param kr_prod stores the intermediate Khatri-Rao products
 * \param hd_prod stores the intermediate Hadamard product
 * \param stream calculation stream (for GPU)
 * \return 0 if success, non-zero otherwise
 */
template <typename DType>
inline int CPDecompUpdate
  (Tensor<cpu, 1, DType> eigvals,
  std::vector<Tensor<cpu, 2, DType> > factors,
  const Tensor<cpu, 2, DType> &unfolding,
  int mode,
  std::vector<TensorContainer<cpu, 2, DType> > kr_prod,
  TensorContainer<cpu, 2, DType> hd_prod,
  Stream<cpu> *stream = NULL);

/*!
 * \brief Evaluates if the ALS algorithm has converged
 *
 * Evaluates if across two steps t, t+1 of the ALS algorithm, the norm of the change of a vector is less than or equal to eps times the norm of the vector at time t,
 *
 *   \lVert v^{(t+1)}-v^{(t)} \rVert_2 \le eps \lVert v^{(t)} \rVert_2,
 *
 * for the eigen-value vector and all the columns of all the factor matrices.
 *
 * \param eigvals eigen-value vector at time t+1
 * \param factors_T transposed factor matrices at time t+1
 * \param oldEigvals eigen-value vector at time t
 * \param oldFactors_T transposed factor matrices at time t+1
 * \param eps relative error threshold
 * \return if all vectors converged
 */
template <typename DType>
inline bool CPDecompConverged
  (const Tensor<cpu, 1, DType> &eigvals,
  const std::vector<Tensor<cpu, 2, DType> > &factors_T,
  const Tensor<cpu, 1, DType> &oldEigvals,
  const std::vector<Tensor<cpu, 2, DType> > &oldFactors_T,
  DType eps);

/*!
 * \brief Sort the eigenvalue vector in descending order and re-arrange the columns in factor matrices accordingly
 *
 * \param eigvals eigenvalue vector obtained from CPDecomp
 * \param factors_T transposed factor matrices obtained from CPDecomp
 */
template <typename DType>
inline void CPDecompSortResults
  (Tensor<cpu, 1, DType> eigvals,
  std::vector<Tensor<cpu, 2, DType> > factors_T);

/*!
 * \brief Perform CANDECOMP/PARAFAC Decomposition on CPU
 *
 * This function performs CP Decompsition for input tensor of arbitrary order and shape on CPU. At success, it populates `eigvals` and `factors_T` with the eigen-value vector sorted in descending order and transposed factor matrices along each dimension, and returns 0; otherwise returns 1.
 *
 * Internally it uses an iterative algorithm with random initial matrices and may not necessarily converge to the same solution from run to run.
 *
 * \param eigvals eigen-value vector to be populated
 * \param factors_T array of the transposed factor matrices to be populated
 * \param in input tensor
 * \param k rank for the CP Decomposition
 * \param eps relative error thershold for checking convergence, default: 1e-6
 * \param max_iter maximum iterations for each run of the ALS algorithm, default: 100
 * \param restarts number of runs for the ALS algorithm, default: 5
 * \param init_orthonormal_factors whether to initialise the factor matrices with orthonormal columns, default: true
 * \param stream calculation stream (for GPU)
 * \return 0 if success, 1 otherwise
 */
template <int order, typename DType>
inline int CPDecomp
  (Tensor<cpu, 1, DType> eigvals,
  std::vector<Tensor<cpu, 2, DType> > factors_T,
  const Tensor<cpu, order, DType> &in,
  int k,
  DType eps = 1e-3,
  int max_iter = 100,
  int restarts = 10,
  bool init_orthonormal_factors = true,
  Stream<cpu> *stream = NULL) {
  CHECK_GE(order, 2);
  CHECK_GE(k, 1);

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

  // Unfold the input tensor along specified mode
  // unfoldings[id_mode] is a matrix of shape
  // (in.size(id_mode), tensor_size / in.size(id_mode))
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
      init_orthonormal_factors, &generator);
    if (status != 0) {
#if DEBUG
      std::cerr << "Init error\n";
#endif
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
#if DEBUG
          std::cerr << "Iter " << iter << " Update error\n";
#endif
          break;
        }
      }

      ++iter;
    }

    if (status != 0)
      continue;

    currReconstructionError = CPDecompReconstructionError
      (in, currEigvals, currFactors_T);
#if DEBUG
    print1DTensor_(currEigvals);
    std::cerr << "Reconstruction error: " << currReconstructionError << "\n";
#endif
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

  if (reconstructionError < std::numeric_limits<DType>::infinity()) {
    CPDecompSortResults(eigvals, factors_T);
    return 0;
  } else {
    return 1;
  }
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
  for (int flat_id = 0;
      flat_id < static_cast<int>(t.shape_.Size()); ++flat_id) {
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
  RandomEngine *generator) {
  int status = 0;

  int order = static_cast<int>(factors_T.size());
  int k = factors_T[0].size(0);
  for (const auto &mat : factors_T)
    CHECK_EQ((int) mat.size(0), k);

  // TODO(jli05): implement seed for random generator
  std::normal_distribution<DType> normal(0.0, 1.0);

  for (int id_mode = 0; id_mode < order; ++id_mode) {
    for (int i = 0; i < k; ++i) {
      for (int j = 0; j < static_cast<int>(factors_T[id_mode].size(1)); ++j)
        factors_T[id_mode][i][j] = normal(*generator);
    }

    if (orthonormal) {
      status = orgqr<cpu, DType>(static_cast<int>(factors_T[id_mode].size(1)),
          k, k, factors_T[id_mode].dptr_, factors_T[id_mode].stride_);
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
  int order = static_cast<int>(factors_T.size());
  int k = eigvals.size(0);

  for (auto &m : factors_T)
    CHECK_EQ(static_cast<int>(m.size(0)), k);

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
  for (int p = 0; p < static_cast<int>(factors_T.size()); ++p) {
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

// Argsort 1D Tensor in descending order
template <typename DType>
std::vector<int> sort_indexes(const Tensor<cpu, 1, DType> &v) {
  // initialize original index locations
  std::vector<int> idx(v.size(0));
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](int i1, int i2) { return v[i1] > v[i2]; });

  return idx;
}

template <typename DType>
inline void CPDecompSortResults
  (Tensor<cpu, 1, DType> eigvals,
  std::vector<Tensor<cpu, 2, DType> > factors_T) {
  int order = factors_T.size();
  int k = eigvals.size(0);

  TensorContainer<cpu, 1, DType> eigvals_(eigvals.shape_);
  std::vector<TensorContainer<cpu, 2, DType> > factors_T_;

  Copy(eigvals_, eigvals);
  for (int id_mode = 0; id_mode < order; ++id_mode) {
    factors_T_.emplace_back(factors_T[id_mode].shape_);
    Copy(factors_T_[id_mode], factors_T[id_mode]);
  }

  std::vector<int> idx = sort_indexes(eigvals);

  for (int i = 0; i < k; ++i) {
    eigvals_[i] = eigvals[idx[i]];
    for (int id_mode = 0; id_mode < order; ++id_mode) {
      for (int j = 0; j < static_cast<int>(factors_T[id_mode].size(1)); ++j)
        factors_T_[id_mode][i][j] = factors_T[id_mode][idx[i]][j];
    }
  }

  Copy(eigvals, eigvals_);
  for (int id_mode = 0; id_mode < order; ++id_mode) {
    Copy(factors_T[id_mode], factors_T_[id_mode]);
  }
}



template <typename DType>
inline void print1DTensor_(const Tensor<cpu, 1, DType> &t) {
  for (int i = 0; i < static_cast<int>(t.size(0)); ++i)
    std::cerr << t[i] << " ";
  std::cerr << "\n";
}

template <typename DType>
inline void print2DTensor_(const Tensor<cpu, 2, DType> &t) {
  for (int i = 0; i < static_cast<int>(t.size(0)); ++i) {
    for (int j = 0; j < static_cast<int>(t.size(1)); ++j)
      std::cerr << t[i][j] << " ";
    std::cerr << "\n";
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_TENSOR_CP_DECOMP_H_
