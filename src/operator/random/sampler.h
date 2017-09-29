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
 * \brief implementations of sampling functors.
 */

using namespace mshadow;
using namespace mxnet::op::mxnet_op;

#include "./log_factorial.h"

template<typename xpu, typename DType>
class RandGenerator;

template<typename DType>
class RandGenerator<cpu, DType> {
public:
  std::mt19937 engine;
  std::uniform_real_distribution<DType> uniformNum;
  std::normal_distribution<DType> normalNum;
  RandGenerator(int seed): engine(seed) {}
  MSHADOW_XINLINE int rand() { return engine(); }
  MSHADOW_XINLINE DType uniform() { return uniformNum(engine); }
  MSHADOW_XINLINE DType normal() { return normalNum(engine); }
};

#ifdef __CUDACC__
#include <curand.h>
#include <curand_kernel.h>

template<>
class RandGenerator<gpu, float> {
public:
  curandState_t state;
  __device__ RandGenerator(int seed) { curand_init(seed, 0, 0, &state); }
  MSHADOW_FORCE_INLINE __device__ int rand() { return curand(&state); }
  MSHADOW_FORCE_INLINE __device__ float uniform() { return curand_uniform(&state); }
  MSHADOW_FORCE_INLINE __device__ float normal() { return curand_normal(&state); }
};

template<>
class RandGenerator<gpu, double> {
public:
  curandState_t state;
  __device__ RandGenerator(int seed) { curand_init(seed, 0, 0, &state); }
  MSHADOW_FORCE_INLINE __device__ int rand() { return curand(&state); }
  MSHADOW_FORCE_INLINE __device__ double uniform() { return curand_uniform_double(&state); }
  MSHADOW_FORCE_INLINE __device__ double normal() { return curand_normal_double(&state); }
};
#endif

template<typename xpu>
MSHADOW_XINLINE int DefaultThreadNum();

template<>
MSHADOW_XINLINE int DefaultThreadNum<cpu>() { 
  return omp_get_num_threads();
}
template<>
MSHADOW_XINLINE int DefaultThreadNum<gpu>() { 
  return 1024;
}

// In the following:
//   nParm is the number of parameters supplied (length of the PType arrays)
//   nSample is the number of samples that should be drawn for each parameter/parameter set
//   out is the final output. It is an array of size nParms*nSamples 
//   nSeed is the number of seeds supplied
//   seed is the array holding the seeds


template<typename xpu>
struct SampleUniformKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, const index_t nParm, const index_t nSample,
                                  const index_t nSeed, const IType *lower, const IType *upper,
                                  OType *out, const int seed) {
    RandGenerator<xpu, OType> gen(seed + i);
    index_t N(nParm * nSample);
    for ( index_t j = i; j < N; j += nSeed ) {
      out[j] = lower[j/nSample] + (upper[j/nSample] - lower[j/nSample]) * gen.uniform();
    }
  }
};

template<typename xpu>
struct UniformSampler {
  int OptThreadNum() { return DefaultThreadNum<xpu>(); }
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const index_t nParm, const index_t nSample, const index_t nSeed,
                              const IType *lower, const IType *upper, OType *out,
                              const int seed, Stream<xpu> *s) {
    Kernel<SampleUniformKernel<xpu>, xpu>
        ::Launch(s, nSeed, nParm, nSample, nSeed, lower, upper, out, seed);
  }
};

template<typename xpu>
struct SampleNormalKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, const index_t nParm, const index_t nSample,
                                  const index_t nSeed, const IType *mean, const IType *std,
                                  OType *out, const int seed) {
    RandGenerator<xpu, OType> gen(seed + i);
    index_t N(nParm * nSample);
    for( index_t j = i; j < N; j += nSeed ) {
      out[j] = (gen.normal() * std[j/nSample]) + mean[j/nSample];
    }
  }
};

template<typename xpu>
struct NormalSampler {
  int OptThreadNum() { return DefaultThreadNum<xpu>(); }
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const index_t nParm, const index_t nSample, const index_t nSeed,
                              const IType *mean, const IType *std, OType *out,
                              const int seed, Stream<xpu> *s) {
    Kernel<SampleNormalKernel<xpu>, xpu>
        ::Launch(s, nSeed, nParm, nSample, nSeed, mean, std, out, seed);
  }
};

struct SampleExponentialKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, const index_t nSample, const IType *lambda, OType *out) {
    out[i] = - log(out[i]) / lambda[i/nSample];
  }
};

template<typename xpu>
struct ExponentialSampler {
  index_t OptThreadNum() { return DefaultThreadNum<xpu>(); }
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const index_t nParm, const index_t nSample, const index_t nSeed,
                              const IType *lambda, OType *out,
                              const int seed, Stream<xpu> *s) {
    UniformSampler<xpu> Sampler;
    OType lower = OType(0);
    OType upper = OType(1);
    Sampler.Sample(1, nParm*nSample, nSeed, &lower, &upper, out, seed, s);
    Kernel<SampleExponentialKernel, xpu>
        ::Launch(s, nParm*nSample, nSample, lambda, out);
  }
};

// Separate function as we need it also for negative binomials.
template<typename xpu, typename IType, typename OType>
MSHADOW_XINLINE OType SampleGamma(IType a, IType b, RandGenerator<xpu, OType>& gen) {
  // Generate one sample of the gamma distribution
  OType sample;
  OType d = a < 1 ? a + 2.0 / 3 : a - 1.0 / 3;
  OType k = sqrt(9 * d);
  OType c = 1.0 / k;
  while (1) {
    OType Z = gen.normal();
    if (Z > -k) {
      OType x = 1 + c * Z;
      OType V = x * x * x;
      if (log(gen.uniform()) < 0.5 * Z * Z + d * (1 - V + log(V))) {
        sample = d * V * b;
        break;
      }
    }
  }
  return a < 1 ? sample * pow(gen.uniform(), OType(1.0 / a)) : sample;
}

template<typename xpu>
struct SampleGammaKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, const index_t nParm, const index_t nSample,
                                  const index_t nSeed, const IType *alpha, const IType *beta,
                                  OType *out, const int seed) {
    RandGenerator<xpu, OType> gen(seed + i);
    index_t N(nParm * nSample);
    for( index_t j = i; j < N; j += nSeed ) {
      out[j] = SampleGamma(alpha[j/nSample], beta[j/nSample], gen);
    }
  }
};

template<typename xpu>
struct GammaSampler {
  int OptThreadNum() { return DefaultThreadNum<xpu>(); }
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const index_t nParm, const index_t nSample, const index_t nSeed,
                              const IType *alpha, const IType *beta, OType *out,
                              const int seed, Stream<xpu> *s) {
    Kernel<SampleGammaKernel<xpu>, xpu>
        ::Launch(s, nSeed, nParm, nSample, nSeed, alpha, beta, out, seed);
  }
};

// Seperate function as we need it also for negative binomials.
template<typename xpu, typename IType, typename OType>
MSHADOW_XINLINE int SamplePoisson(IType lambda, RandGenerator<xpu, OType>& gen) {
  // Generate one sample of the poisson distribution
  // Reference : www.johndcook.com/blog/2010/06/14/generating-poisson-random-values
  // Involves computing log factorial which requires tabulating based on Stirling's approximation
  if (lambda < 30) {
    OType t = expf(-lambda);
    int x = 0;
    for ( OType prod = gen.uniform(); prod > t; prod *= gen.uniform() ) { x += 1; }
    return x;
  }
  else {
    OType c = 0.767 - 3.36 / lambda;
    OType beta = kPi / sqrt(3.0 * lambda);
    OType alpha = beta * lambda;
    OType k = log(c) - lambda - log(beta);
    for (int cnt = 1; cnt < 1000; cnt ++) {
      OType u = gen.uniform();
      if (u == 1.0) { continue; }
      OType x = (alpha - log((1.0 - u)/u))/beta;
      int n = floor(x + 0.5);
      if (n < 0) { continue; }
      OType v = gen.uniform();
      OType y = alpha - beta * x;
      OType lhs = y + log(v/(1.0 + expf(y))/(1.0 + expf(y)));
      OType rhs = k + n * log(float(lambda)) - LogFactorial(n);
      if (lhs <= rhs) { return n; }
    }
  return 0;
  }
}

template<typename xpu>
struct SamplePoissonKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, const index_t nParm, const index_t nSample,
                                  const index_t nSeed, const IType *lambda,
                                  OType *out, const int seed) {
    RandGenerator<xpu, float> gen(seed + i);
    index_t N(nParm * nSample);
    for( index_t j = i; j < N; j += nSeed ) {
      out[j] = OType(SamplePoisson(lambda[j/nSample], gen));
    }
  }
};

template<typename xpu>
struct PoissonSampler {
  int OptThreadNum() { return DefaultThreadNum<xpu>(); }
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const index_t nParm, const index_t nSample, const index_t nSeed,
                              const IType *lambda, OType *out,
                              const int seed, Stream<xpu> *s) {
    Kernel<SamplePoissonKernel<xpu>, xpu>
        ::Launch(s, nSeed, nParm, nSample, nSeed, lambda, out, seed);
  }
};

template<typename xpu>
struct SampleNegativeBinomialKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, const index_t nParm, const index_t nSample,
                                  const index_t nSeed, const IType *k, const IType *p,
                                  OType *out, const int seed) {
    RandGenerator<xpu, float> gen(seed + i);
    index_t N(nParm * nSample);
    for( index_t j = i; j < N; j += nSeed ) {
      IType alpha = k[j/nSample];
      IType prob = p[j/nSample];
      IType beta = (1.0 - prob) / prob;
      IType lambda = SampleGamma(alpha, beta, gen);
      out[j] = OType(SamplePoisson(lambda, gen));
    }
  }
};

template<typename xpu>
struct NegativeBinomialSampler {
  int OptThreadNum() { return DefaultThreadNum<xpu>(); }
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const index_t nParm, const index_t nSample, const index_t nSeed,
                              const IType *k, const IType *p, OType *out,
                              const int seed, Stream<xpu> *s) {
    Kernel<SampleNegativeBinomialKernel<xpu>, xpu>
        ::Launch(s, nSeed, nParm, nSample, nSeed, k, p, out, seed);
  }
};

template<typename xpu>
struct SampleGeneralizedNegativeBinomialKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, const index_t nParm, const index_t nSample,
                                  const index_t nSeed, const IType *mu, const IType *alpha,
                                  OType *out, const int seed) {
    RandGenerator<xpu, float> gen(seed + i);
    index_t N(nParm * nSample);
    for( index_t j = i; j < N; j += nSeed ) {
      float lambda = alpha[j/nSample] == 0 ? float(mu[j/nSample]) : SampleGamma(
          IType(1) / alpha[j/nSample], alpha[j/nSample] * mu[j/nSample], gen);
      out[j] = OType(SamplePoisson(lambda, gen));
    }
  }
};

template<typename xpu>
struct GeneralizedNegativeBinomialSampler {
  int OptThreadNum() { return DefaultThreadNum<xpu>(); }
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const index_t nParm, const index_t nSample, const index_t nSeed,
                              const IType *mu, const IType *alpha, OType *out,
                              const int seed, Stream<xpu> *s) {
    Kernel<SampleGeneralizedNegativeBinomialKernel<xpu>, xpu>
        ::Launch(s, nSeed, nParm, nSample, nSeed, mu, alpha, out, seed);
  }
};

