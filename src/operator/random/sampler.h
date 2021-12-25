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
 * \file sampler.h
 * \brief implementations of random sampling functors.
 */

#ifndef MXNET_OPERATOR_RANDOM_SAMPLER_H_
#define MXNET_OPERATOR_RANDOM_SAMPLER_H_

#include <algorithm>

using namespace mshadow;
using namespace mxnet::op::mxnet_op;
using namespace mxnet::common::random;

namespace mxnet {
namespace op {

/*!
 * \brief Launch a generic kernel with parallel random generator.
 * \tparam gen random generator
 * \tparam N Number of iterations
 * \tparam Args Varargs type to eventually pass to the OP::Map() function
 */
template <typename OP, typename xpu, typename GType, typename... Args>
inline static void LaunchRNG(mshadow::Stream<xpu>* s,
                             common::random::RandGenerator<xpu, GType>* gen,
                             const index_t N,
                             Args... args) {
  // minimal check to avoid division by zero, below.
  // if `N` is zero the map operation is a no-op in any case.
  if (N <= 0) {
    return;
  }
  const index_t nloop = (N + RandGenerator<xpu>::kMinNumRandomPerThread - 1) /
                        RandGenerator<xpu>::kMinNumRandomPerThread;
  const index_t nthread =
      std::min(nloop, static_cast<index_t>(RandGenerator<xpu>::kNumRandomStates));
  const index_t step = (N + nthread - 1) / nthread;
  Kernel<OP, xpu>::Launch(s, nthread, *gen, N, step, args...);
}

#define RNG_KERNEL_LOOP(xpu, GType, thread_id, gen, N, step, ...)    \
  const index_t start = thread_id * step;                            \
  const index_t end   = start + step;                                \
  typename RandGenerator<xpu, GType>::Impl genImpl(&gen, thread_id); \
  for (index_t i = start; i < end && i < N; ++i) {                   \
    { __VA_ARGS__ }                                                  \
  }

template <typename xpu>
struct SampleUniformKernel {
  template <typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t id,
                                  RandGenerator<xpu, OType> gen,
                                  const index_t N,
                                  const index_t step,
                                  index_t nParm,
                                  index_t nSample,
                                  const IType* lower,
                                  const IType* upper,
                                  OType* out) {
    RNG_KERNEL_LOOP(xpu, OType, id, gen, N, step, {
      index_t nBatch(1 + (nSample - 1) / nParm);
      out[i] =
          OType(lower[i / nBatch] + (upper[i / nBatch] - lower[i / nBatch]) * genImpl.uniform());
    });
  }
};

template <typename xpu>
struct UniformSampler {
  template <typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& lower,
                                   const Tensor<xpu, 1, IType>& upper,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGenerator<xpu, OType>* pgen,
                                   Stream<xpu>* s) {
    LaunchRNG<SampleUniformKernel<xpu>, xpu>(
        s, pgen, out.size(0), lower.size(0), out.size(0), lower.dptr_, upper.dptr_, out.dptr_);
  }
};

template <typename xpu>
struct SampleRandIntKernel {
  template <typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t id,
                                  RandGenerator<xpu, OType> gen,
                                  const index_t N,
                                  const index_t step,
                                  index_t nParm,
                                  index_t nSample,
                                  const IType* lower,
                                  const IType* upper,
                                  OType* out) {
    RNG_KERNEL_LOOP(xpu, OType, id, gen, N, step, {
      index_t nBatch(1 + (nSample - 1) / nParm);
      if (sizeof(IType) == sizeof(int64_t))
        out[i] = OType(lower[i / nBatch] +
                       genImpl.rand_int64() % (upper[i / nBatch] - lower[i / nBatch]));
      else
        out[i] =
            OType(lower[i / nBatch] + genImpl.rand() % (upper[i / nBatch] - lower[i / nBatch]));
    });
  }
};

template <typename xpu>
struct RandIntSampler {
  template <typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& lower,
                                   const Tensor<xpu, 1, IType>& upper,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGenerator<xpu, OType>* pgen,
                                   Stream<xpu>* s) {
    LaunchRNG<SampleRandIntKernel<xpu>, xpu>(
        s, pgen, out.size(0), lower.size(0), out.size(0), lower.dptr_, upper.dptr_, out.dptr_);
  }
};

template <typename xpu>
struct SampleNormalKernel {
  template <typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t id,
                                  RandGenerator<xpu, OType> gen,
                                  const index_t N,
                                  const index_t step,
                                  index_t nParm,
                                  index_t nSample,
                                  const IType* mean,
                                  const IType* std,
                                  OType* out) {
    RNG_KERNEL_LOOP(xpu, OType, id, gen, N, step, {
      index_t nBatch(1 + (nSample - 1) / nParm);
      out[i] = OType(genImpl.normal() * std[i / nBatch] + mean[i / nBatch]);
    });
  }
};

template <typename xpu>
struct NormalSampler {
  template <typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& mean,
                                   const Tensor<xpu, 1, IType>& std,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGenerator<xpu, OType>* pgen,
                                   Stream<xpu>* s) {
    LaunchRNG<SampleNormalKernel<xpu>, xpu>(
        s, pgen, out.size(0), mean.size(0), out.size(0), mean.dptr_, std.dptr_, out.dptr_);
  }
};

template <typename xpu>
struct SampleExponentialKernel {
  template <typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t id,
                                  RandGenerator<xpu, OType> gen,
                                  const index_t N,
                                  const index_t step,
                                  index_t nParm,
                                  index_t nSample,
                                  const IType* lambda,
                                  OType* out) {
    RNG_KERNEL_LOOP(xpu, OType, id, gen, N, step, {
      index_t nBatch(1 + (nSample - 1) / nParm);
      out[i] = OType(-log(1.0 - genImpl.uniform()) / lambda[i / nBatch]);
    });
  }
};

template <typename xpu>
struct ExponentialSampler {
  template <typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& lambda,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGenerator<xpu, OType>* pgen,
                                   Stream<xpu>* s) {
    LaunchRNG<SampleExponentialKernel<xpu>, xpu>(
        s, pgen, out.size(0), lambda.size(0), out.size(0), lambda.dptr_, out.dptr_);
  }
};

template <typename xpu, typename IType, typename OType>
MSHADOW_XINLINE OType SampleGamma(IType a, IType b, typename RandGenerator<xpu, OType>::Impl* gen) {
  // Generate one sample of the gamma distribution
  OType sample;
  OType d = a < 1 ? a + 2.0 / 3.0 : a - 1.0 / 3.0;
  OType k = sqrt(9.0 * d);
  OType c = 1.0 / k;
  while (1) {
    OType Z = gen->normal();
    if (Z > -k) {
      OType x = 1.0 + c * Z;
      OType V = x * x * x;
      if (log(1.0 - gen->uniform()) < 0.5 * Z * Z + d * (1.0 - V + log(V))) {
        sample = d * V * b;
        break;
      }
    }
  }
  return a < 1 ? sample * pow(gen->uniform(), OType(1.0 / a)) : sample;
}

template <typename xpu>
struct SampleGammaKernel {
  template <typename IType, typename OType, typename FType>
  MSHADOW_XINLINE static void Map(index_t id,
                                  RandGenerator<xpu, FType> gen,
                                  const index_t N,
                                  const index_t step,
                                  index_t nParm,
                                  index_t nSample,
                                  const IType* alpha,
                                  const IType* beta,
                                  OType* out) {
    RNG_KERNEL_LOOP(xpu, FType, id, gen, N, step, {
      index_t nBatch(1 + (nSample - 1) / nParm);
      out[i] = OType(SampleGamma<xpu, IType, FType>(alpha[i / nBatch], beta[i / nBatch], &genImpl));
    });
  }
};

template <typename xpu>
struct GammaSampler {
  template <typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& alpha,
                                   const Tensor<xpu, 1, IType>& beta,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGenerator<xpu, OType>* pgen,
                                   Stream<xpu>* s) {
    typedef
        typename std::conditional<std::is_floating_point<OType>::value, OType, float>::type FType;
    RandGenerator<xpu, FType>* gen = reinterpret_cast<RandGenerator<xpu, FType>*>(pgen);
    LaunchRNG<SampleGammaKernel<xpu>, xpu>(
        s, gen, out.size(0), alpha.size(0), out.size(0), alpha.dptr_, beta.dptr_, out.dptr_);
  }
};

template <typename xpu>
MSHADOW_XINLINE int SamplePoisson(float lambda, typename RandGenerator<xpu, float>::Impl* gen) {
  // Generate one sample of the poisson distribution. Intentionally written
  // towards a specific type (float) for internal computation which is sufficient
  // for accurate enough computation.
  if (lambda < 12.0) {
    float t = expf(-lambda);
    int x   = 0;
    for (float prod = gen->uniform(); prod > t; prod *= gen->uniform()) {
      x += 1;
    }
    return x;
  } else {
    // Approximation for high lambda according to:
    // Numerical Recipes in C: The Art of Scientific Computing
    // Cambridge University Press
    const float pi(3.1415926);
    const float sq(sqrt(2.0 * lambda));
    const float loglambda(log(lambda));
    const float g(lambda * loglambda - lgammaf(lambda + 1.0));
    float em(0), t(0), y(0);
    do {
      do {
        y  = tanf(pi * gen->uniform());
        em = sq * y + lambda;
      } while (em < 0.0);
      em = floorf(em);
      t  = 0.9 * (1.0 + y * y) * expf(em * loglambda - lgammaf(em + 1.0) - g);
    } while (gen->uniform() > t);
    return static_cast<int>(em);
  }
}

template <typename xpu>
struct SamplePoissonKernel {
  template <typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t id,
                                  RandGenerator<xpu, float> gen,
                                  const index_t N,
                                  const index_t step,
                                  index_t nParm,
                                  index_t nSample,
                                  const IType* lambda,
                                  OType* out) {
    RNG_KERNEL_LOOP(xpu, float, id, gen, N, step, {
      index_t nBatch(1 + (nSample - 1) / nParm);
      out[i] = OType(SamplePoisson<xpu>(lambda[i / nBatch], &genImpl));
    });
  }
};

template <typename xpu>
struct PoissonSampler {
  template <typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& lambda,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGenerator<xpu, OType>* pgen,
                                   Stream<xpu>* s) {
    RandGenerator<xpu, float>* gen = reinterpret_cast<RandGenerator<xpu, float>*>(pgen);
    LaunchRNG<SamplePoissonKernel<xpu>, xpu>(
        s, gen, out.size(0), lambda.size(0), out.size(0), lambda.dptr_, out.dptr_);
  }
};

MSHADOW_XINLINE double stirling_approximation(double k) {
  static const double table[] = {0.08106146679532726,
                                 0.04134069595540929,
                                 0.02767792568499834,
                                 0.02079067210376509,
                                 0.01664469118982119,
                                 0.01387612882307075,
                                 0.01189670994589177,
                                 0.01041126526197209,
                                 0.009255462182712733,
                                 0.008330563433362871};

  if (k <= 9)
    return table[static_cast<int>(k)];

  return (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / ((k + 1) * (k + 1))) / (((k + 1) * (k + 1)))) /
         (k + 1);
}

// The algorithm is explained in https://www.tandfonline.com/doi/abs/10.1080/00949659308811496
template <typename xpu, typename IType, typename OType>
MSHADOW_XINLINE OType _sample_binomial_btrd(IType N,
                                            IType p,
                                            typename RandGenerator<xpu, float>::Impl* gen) {
  OType m   = floor((N + 1) * p);
  OType r   = p / (1 - p);
  OType nr  = (N + 1) * r;
  OType npq = N * p * (1 - p);

  OType b     = 1.15 + 2.53 * sqrt(npq);
  OType a     = -0.0873 + 0.0248 * b + 0.01 * p;
  OType c     = N * p + 0.5;
  OType alpha = (2.83 + 5.1 / b) * sqrt(npq);

  OType v_r      = 0.92 - 4.2 / b;
  OType u_r__v_r = 0.86 * v_r;

  while (true) {
    OType v = gen->uniform();
    if (v <= u_r__v_r) {
      OType u = v / v_r - 0.43;

      return floor((2 * a / (0.5 - abs(u)) + b) * u + c);
    }

    OType u;
    if (v >= v_r) {
      u = gen->uniform() - 0.5;
    } else {
      u           = v / v_r - 0.93;
      OType sgn_u = ((0 < u) - (u < 0));
      u           = sgn_u * 0.5 - u;

      v = gen->uniform() * v_r;
    }

    OType us = 0.5 - abs(u);
    OType k  = floor((2 * a / us + b) * u + c);
    if (k < 0 || k > N) {
      continue;
    }

    v = v * alpha / (a / (us * us) + b);

    OType km = abs(k - m);
    if (km <= 15) {
      OType f = 1;
      for (double i = m; i < k; ++i)
        f = f * (nr / i - r);
      for (double i = k; i < m; ++i)
        v = v * (nr / i - r);

      if (v <= f) {
        return k;
      }

      continue;
    }

    v         = log(v);
    OType rho = (km / npq) * (((km / 3 + 0.625) * km + 1.0 / 6) / npq + 0.5);
    OType t   = -km * km / (2 * npq);
    if (v < t - rho) {
      return k;
    }

    if (v > t + rho) {
      continue;
    }

    OType nm = N - m + 1;
    OType h  = (m + 0.5) * log((m + 1) / (r * nm)) + stirling_approximation(m) +
              stirling_approximation(N - m);

    OType nk  = N - k + 1;
    OType tmp = h + (N + 1) * log(nm / nk) + (k + 0.5) * log(nk * r / (k + 1)) -
                stirling_approximation(k) - stirling_approximation(N - k);
    if (v <= tmp) {
      return k;
    }
  }
}

template <typename xpu, typename IType, typename OType>
MSHADOW_XINLINE OType _sample_binomial_inversion(IType n,
                                                 IType p,
                                                 typename RandGenerator<xpu, float>::Impl* gen) {
  OType N = static_cast<OType>(n);
  OType q = static_cast<OType>(p);
  if (q > 0.5)
    q = 1 - q;

  OType s = 1 - q;

  OType A = 1;
  OType B = q / s;
  OType C = (N + 1) * B;
  OType D = A;
  OType X = 0;

  OType U = gen->uniform();
  OType V = U / pow(s, N);

  do {
    if (V <= A)
      break;
    X = X + 1;
    D = D * (C / X - B);
    A = A + D;
  } while (X < N);

  if (p > 0.5)
    return N - X;

  return X;
}

template <typename xpu, typename IType, typename OType>
MSHADOW_XINLINE OType SampleBinomial(IType n,
                                     IType p,
                                     typename RandGenerator<xpu, float>::Impl* gen) {
  // Generate one sample of the binomial distribution
  if (p >= 1) {
    return static_cast<OType>(n);
  }

  if (p <= 0.5) {
    if (n * p >= 10.0) {
      return _sample_binomial_btrd<xpu, IType, OType>(n, p, gen);
    } else {
      return _sample_binomial_inversion<xpu, IType, OType>(n, p, gen);
    }
  } else {
    IType q = 1.0 - p;
    if (n * q >= 10.0) {
      return n - _sample_binomial_btrd<xpu, IType, OType>(n, q, gen);
    } else {
      return n - _sample_binomial_inversion<xpu, IType, OType>(n, q, gen);
    }
  }
}

template <typename xpu>
struct SampleBinomialKernel {
  template <typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t id,
                                  RandGenerator<xpu, float> gen,
                                  const index_t N,
                                  const index_t step,
                                  index_t nParm,
                                  index_t nSample,
                                  const IType* n,
                                  const IType* p,
                                  OType* out) {
    RNG_KERNEL_LOOP(xpu, float, id, gen, N, step, {
      index_t nBatch(1 + (nSample - 1) / nParm);
      out[i] = SampleBinomial<xpu, IType, OType>(n[i / nBatch], p[i / nBatch], &genImpl);
    });
  }
};

template <typename xpu>
struct BinomialSampler {
  template <typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& n,
                                   const Tensor<xpu, 1, IType>& p,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGenerator<xpu, OType>* pgen,
                                   Stream<xpu>* s) {
    RandGenerator<xpu, float>* gen = reinterpret_cast<RandGenerator<xpu, float>*>(pgen);
    LaunchRNG<SampleBinomialKernel<xpu>, xpu>(
        s, gen, out.size(0), n.size(0), out.size(0), n.dptr_, p.dptr_, out.dptr_);
  }
};

template <typename xpu>
struct SampleNegativeBinomialKernel {
  template <typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t id,
                                  RandGenerator<xpu, float> gen,
                                  const index_t N,
                                  const index_t step,
                                  index_t nParm,
                                  index_t nSample,
                                  const IType* k,
                                  const IType* p,
                                  OType* out) {
    RNG_KERNEL_LOOP(xpu, float, id, gen, N, step, {
      index_t nBatch(1 + (nSample - 1) / nParm);
      float alpha  = k[i / nBatch];
      float prob   = p[i / nBatch];
      float beta   = (1.0 - prob) / prob;
      float lambda = SampleGamma<xpu, IType, float>(alpha, beta, &genImpl);
      out[i]       = OType(SamplePoisson<xpu>(lambda, &genImpl));
    });
  }
};

template <typename xpu>
struct NegativeBinomialSampler {
  template <typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& k,
                                   const Tensor<xpu, 1, IType>& p,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGenerator<xpu, OType>* pgen,
                                   Stream<xpu>* s) {
    RandGenerator<xpu, float>* gen = reinterpret_cast<RandGenerator<xpu, float>*>(pgen);
    LaunchRNG<SampleNegativeBinomialKernel<xpu>, xpu>(
        s, gen, out.size(0), k.size(0), out.size(0), k.dptr_, p.dptr_, out.dptr_);
  }
};

template <typename xpu>
struct SampleGeneralizedNegativeBinomialKernel {
  template <typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t id,
                                  RandGenerator<xpu, float> gen,
                                  const index_t N,
                                  const index_t step,
                                  index_t nParm,
                                  index_t nSample,
                                  const IType* mu,
                                  const IType* alpha,
                                  OType* out) {
    RNG_KERNEL_LOOP(xpu, float, id, gen, N, step, {
      index_t nBatch(1 + (nSample - 1) / nParm);
      float lambda =
          alpha[i / nBatch] == 0 ?
              static_cast<float>(mu[i / nBatch]) :
              SampleGamma<xpu, IType, float>(
                  IType(1) / alpha[i / nBatch], alpha[i / nBatch] * mu[i / nBatch], &genImpl);
      out[i] = OType(SamplePoisson<xpu>(lambda, &genImpl));
    });
  }
};

template <typename xpu>
struct GeneralizedNegativeBinomialSampler {
  template <typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& mu,
                                   const Tensor<xpu, 1, IType>& alpha,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGenerator<xpu, OType>* pgen,
                                   Stream<xpu>* s) {
    RandGenerator<xpu, float>* gen = reinterpret_cast<RandGenerator<xpu, float>*>(pgen);
    LaunchRNG<SampleGeneralizedNegativeBinomialKernel<xpu>, xpu>(
        s, gen, out.size(0), mu.size(0), out.size(0), mu.dptr_, alpha.dptr_, out.dptr_);
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_RANDOM_SAMPLER_H_
