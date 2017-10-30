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

#ifdef __CUDACC__
#include <curand.h>
#include <curand_kernel.h>
#endif  // __CUDACC__

using namespace mshadow;
using namespace mxnet::op::mxnet_op;

namespace mxnet {
namespace op {

// Elementary random number generation for int/uniform/gaussian in CPU and GPU.
// Will use float data type whenever instantiated for half_t or any other non
// standard real type.
template<typename xpu, typename DType>
class RandGenerator;

template<typename DType>
class RandGenerator<cpu, DType> {
 public:
  typedef typename std::conditional<std::is_floating_point<DType>::value,
                                    DType, float>::type FType;
  std::mt19937 engine;
  std::uniform_real_distribution<FType> uniformNum;
  std::normal_distribution<FType> normalNum;
  explicit RandGenerator(unsigned int seed): engine(seed) {}
  MSHADOW_XINLINE int rand() { return engine(); }
  MSHADOW_XINLINE FType uniform() { return uniformNum(engine); }
  MSHADOW_XINLINE FType normal() { return normalNum(engine); }
};

#ifdef __CUDACC__

// uniform number generation in Cuda made consistent with stl (include 0 but exclude 1)
// by using 1.0-curand_uniform(). Needed as some samplers below won't be able to deal with
// one of the boundary cases.
template<typename DType>
class RandGenerator<gpu, DType> {
 public:
  curandState_t state;
  __device__ RandGenerator(unsigned int seed) { curand_init(seed, 0, 0, &state); }
  MSHADOW_FORCE_INLINE __device__ int rand() { return curand(&state); }
  MSHADOW_FORCE_INLINE __device__ float uniform()
                              { return static_cast<float>(1.0) - curand_uniform(&state); }
  MSHADOW_FORCE_INLINE __device__ float normal() { return curand_normal(&state); }
};

template<>
class RandGenerator<gpu, double> {
 public:
  curandState_t state;
  __device__ RandGenerator(unsigned int seed) { curand_init(seed, 0, 0, &state); }
  MSHADOW_FORCE_INLINE __device__ int rand() { return curand(&state); }
  MSHADOW_FORCE_INLINE __device__ double uniform()
                            { return static_cast<double>(1.0) - curand_uniform_double(&state); }
  MSHADOW_FORCE_INLINE __device__ double normal() { return curand_normal_double(&state); }
};

#endif  // __CUDACC__

// Number of seeds/threads when sampling on cpu/gpu.
template<typename xpu>
MSHADOW_XINLINE index_t OptSampleSeedNum(index_t N);
template<>
MSHADOW_XINLINE index_t OptSampleSeedNum<cpu>(index_t N) {
  return omp_get_num_threads();
}
template<>
MSHADOW_XINLINE index_t OptSampleSeedNum<gpu>(index_t N) {
  return N;
}

template<typename xpu>
struct SampleUniformKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, index_t nParm, index_t nSample, index_t nSeed,
                     const IType *lower, const IType *upper, OType *out, const unsigned *seed) {
    index_t nBatch(nSample/nParm), nChunk((nSample+nSeed-1)/nSeed),
            start(i*nChunk), end((i+1)*nChunk < nSample ? (i+1)*nChunk : nSample);
    RandGenerator<xpu, OType> gen(seed[i]);
    for ( index_t j = start; j < end; ++j ) {
      out[j] = OType(lower[j/nBatch] + (upper[j/nBatch] - lower[j/nBatch]) * gen.uniform());
    }
  }
};

template<typename xpu>
struct UniformSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& lower,
                                   const Tensor<xpu, 1, IType>& upper,
                                   const Tensor<xpu, 1, OType>& out,
                                   const Tensor<xpu, 1, unsigned>& seed,
                                         Stream<xpu> *s) {
    Kernel<SampleUniformKernel<xpu>, xpu>
      ::Launch(s, seed.size(0), lower.size(0), out.size(0), seed.size(0),
               lower.dptr_, upper.dptr_, out.dptr_, seed.dptr_);
  }
};

template<typename xpu>
struct SampleNormalKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, index_t nParm, index_t nSample, index_t nSeed,
                            const IType *mean, const IType *std, OType *out, const unsigned *seed) {
    index_t nBatch(nSample/nParm), nChunk((nSample+nSeed-1)/nSeed),
            start(i*nChunk), end((i+1)*nChunk < nSample ? (i+1)*nChunk : nSample);
    RandGenerator<xpu, OType> gen(seed[i]);
    for ( index_t j = start; j < end; ++j ) {
      out[j] = OType(gen.normal() * std[j/nBatch] + mean[j/nBatch]);
    }
  }
};

template<typename xpu>
struct NormalSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& mean,
                                   const Tensor<xpu, 1, IType>& std,
                                   const Tensor<xpu, 1, OType>& out,
                                   const Tensor<xpu, 1, unsigned>& seed,
                                         Stream<xpu> *s) {
    Kernel<SampleNormalKernel<xpu>, xpu>
      ::Launch(s, seed.size(0), mean.size(0), out.size(0), seed.size(0),
               mean.dptr_, std.dptr_, out.dptr_, seed.dptr_);
  }
};

template<typename xpu>
struct SampleExponentialKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, index_t nParm, index_t nSample, index_t nSeed,
                                  const IType *lambda, OType *out, const unsigned *seed) {
    index_t nBatch(nSample/nParm), nChunk((nSample+nSeed-1)/nSeed),
            start(i*nChunk), end((i+1)*nChunk < nSample ? (i+1)*nChunk : nSample);
    RandGenerator<xpu, OType> gen(seed[i]);
    for ( index_t j = start; j < end; ++j ) {
      out[j] = OType(-log(1.0-gen.uniform()) / lambda[j/nBatch]);
    }
  }
};

template<typename xpu>
struct ExponentialSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& lambda,
                                   const Tensor<xpu, 1, OType>& out,
                                   const Tensor<xpu, 1, unsigned>& seed,
                                         Stream<xpu> *s) {
    Kernel<SampleExponentialKernel<xpu>, xpu>
      ::Launch(s, seed.size(0), lambda.size(0), out.size(0), seed.size(0),
               lambda.dptr_, out.dptr_, seed.dptr_);
  }
};

template<typename xpu, typename IType, typename OType>
MSHADOW_XINLINE OType SampleGamma(IType a, IType b, RandGenerator<xpu, OType> *gen) {
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
      if (log(1.0-gen->uniform()) < 0.5 * Z * Z + d * (1.0 - V + log(V))) {
        sample = d * V * b;
        break;
      }
    }
  }
  return a < 1 ? sample * pow(gen->uniform(), OType(1.0 / a)) : sample;
}

template<typename xpu>
struct SampleGammaKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, index_t nParm, index_t nSample, index_t nSeed,
                      const IType *alpha, const IType *beta, OType *out, const unsigned *seed) {
    index_t nBatch(nSample/nParm), nChunk((nSample+nSeed-1)/nSeed),
            start(i*nChunk), end((i+1)*nChunk < nSample ? (i+1)*nChunk : nSample);
    typedef typename std::conditional<std::is_floating_point<OType>::value,
                                     OType, float>::type FType;
    RandGenerator<xpu, FType> gen(seed[i]);
    for ( index_t j = start; j < end; ++j ) {
      out[j] = OType(SampleGamma(alpha[j/nBatch], beta[j/nBatch], &gen));
    }
  }
};

template<typename xpu>
struct GammaSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& alpha,
                                   const Tensor<xpu, 1, IType>& beta,
                                   const Tensor<xpu, 1, OType>& out,
                                   const Tensor<xpu, 1, unsigned>& seed,
                                         Stream<xpu> *s) {
    Kernel<SampleGammaKernel<xpu>, xpu>
      ::Launch(s, seed.size(0), alpha.size(0), out.size(0), seed.size(0),
               alpha.dptr_, beta.dptr_, out.dptr_, seed.dptr_);
  }
};

template<typename xpu>
MSHADOW_XINLINE int SamplePoisson(float lambda, RandGenerator<xpu, float> *gen) {
  // Generate one sample of the poisson distribution. Intentionally written
  // towards a specific type (float) for internal computation which is sufficient
  // for accurate enough computation.
  if ( lambda < 12.0 ) {
    float t = expf(-lambda);
    int x = 0;
    for ( float prod = gen->uniform(); prod > t; prod *= gen->uniform() ) { x += 1; }
    return x;
  } else {
    // Approximation for high lambda according to:
    // Numerical Recipes in C: The Art of Scientific Computing
    // Cambridge University Press
    const float pi(3.1415926);
    const float sq(sqrt(2.0*lambda));
    const float loglambda(log(lambda));
    const float g(lambda*loglambda-lgammaf(lambda+1.0));
    float em(0), t(0), y(0);
    do {
      do {
        y = tanf(pi * gen->uniform());
        em = sq * y + lambda;
      } while (em < 0.0);
      em = floorf(em);
      t = 0.9 * (1.0 + y * y) * expf(em * loglambda - lgammaf(em + 1.0) - g);
    } while (gen->uniform() > t);
    return static_cast<int>(em);
  }
}

template<typename xpu>
struct SamplePoissonKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, index_t nParm, index_t nSample, index_t nSeed,
                                  const IType *lambda, OType *out, const unsigned *seed) {
    index_t nBatch(nSample/nParm), nChunk((nSample+nSeed-1)/nSeed),
            start(i*nChunk), end((i+1)*nChunk < nSample ? (i+1)*nChunk : nSample);
    RandGenerator<xpu, float> gen(seed[i]);
    for ( index_t j = start; j < end; ++j ) {
      out[j] = OType(SamplePoisson(lambda[j/nBatch], &gen));
    }
  }
};

template<typename xpu>
struct PoissonSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& lambda,
                                   const Tensor<xpu, 1, OType>& out,
                                   const Tensor<xpu, 1, unsigned>& seed,
                                         Stream<xpu> *s) {
    Kernel<SamplePoissonKernel<xpu>, xpu>
      ::Launch(s, seed.size(0), lambda.size(0), out.size(0), seed.size(0),
               lambda.dptr_, out.dptr_, seed.dptr_);
  }
};

template<typename xpu>
struct SampleNegativeBinomialKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, index_t nParm, index_t nSample, index_t nSeed,
                             const IType *k, const IType *p, OType *out, const unsigned *seed) {
    index_t nBatch(nSample/nParm), nChunk((nSample+nSeed-1)/nSeed),
            start(i*nChunk), end((i+1)*nChunk < nSample ? (i+1)*nChunk : nSample);
    RandGenerator<xpu, float> gen(seed[i]);
    for ( index_t j = start; j < end; ++j ) {
      float alpha = k[j/nBatch];
      float prob = p[j/nBatch];
      float beta = (1.0 - prob) / prob;
      float lambda = SampleGamma(alpha, beta, &gen);
      out[j] = OType(SamplePoisson(lambda, &gen));
    }
  }
};

template<typename xpu>
struct NegativeBinomialSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& k,
                                   const Tensor<xpu, 1, IType>& p,
                                   const Tensor<xpu, 1, OType>& out,
                                   const Tensor<xpu, 1, unsigned>& seed,
                                         Stream<xpu> *s) {
    Kernel<SampleNegativeBinomialKernel<xpu>, xpu>
      ::Launch(s, seed.size(0), k.size(0), out.size(0), seed.size(0),
               k.dptr_, p.dptr_, out.dptr_, seed.dptr_);
  }
};

template<typename xpu>
struct SampleGeneralizedNegativeBinomialKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, index_t nParm, index_t nSample, index_t nSeed,
                        const IType *mu, const IType *alpha, OType *out, const unsigned *seed) {
    index_t nBatch(nSample/nParm), nChunk((nSample+nSeed-1)/nSeed),
            start(i*nChunk), end((i+1)*nChunk < nSample ? (i+1)*nChunk : nSample);
    RandGenerator<xpu, float> gen(seed[i]);
    for ( index_t j = start; j < end; ++j ) {
      float lambda = alpha[j/nBatch] == 0 ? static_cast<float>(mu[j/nBatch])
              : SampleGamma(IType(1) / alpha[j/nBatch], alpha[j/nBatch] * mu[j/nBatch], &gen);
      out[j] = OType(SamplePoisson(lambda, &gen));
    }
  }
};

template<typename xpu>
struct GeneralizedNegativeBinomialSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& mu,
                                   const Tensor<xpu, 1, IType>& alpha,
                                   const Tensor<xpu, 1, OType>& out,
                                   const Tensor<xpu, 1, unsigned>& seed,
                                         Stream<xpu> *s) {
    Kernel<SampleGeneralizedNegativeBinomialKernel<xpu>, xpu>
      ::Launch(s, seed.size(0), mu.size(0), out.size(0), seed.size(0),
               mu.dptr_, alpha.dptr_, out.dptr_, seed.dptr_);
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_RANDOM_SAMPLER_H_
