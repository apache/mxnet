/*!
 *  Copyright (c) 2014 by Contributors
 *  \file random.h
 *  \brief Random inline functions for tensor.
 *  \author Bing Xu, Tianqi Chen
 *   Based on curand|MKL|stdlib
 */
#ifndef MSHADOW_RANDOM_H_
#define MSHADOW_RANDOM_H_

#include <cstdlib>
#include <algorithm>
#include <random>
#include "./base.h"
#include "./tensor.h"
#include "./tensor_container.h"

#if MSHADOW_IN_CXX11
#include <random>  // use cxx11 random by default
#endif

#if _MSC_VER
#define rand_r(x) rand()
#endif


namespace mshadow {
/*!
 * \brief random number generator
 * \tparam Device the device of random number generator
 * \tparam DType the target data type of random number can be float for double
 */
template<typename Device, typename DType MSHADOW_DEFAULT_DTYPE>
class Random {};

/*! \brief CPU random number generator */
template<typename DType>
class Random<cpu, DType> {
 public:
  /*!
   * \brief constructor of random engine
   * \param seed random number seed
   */
  explicit Random(int seed) {
    this->Seed(seed);
    buffer_.Resize(Shape1(kRandBufferSize));
  }
  ~Random(void) {
  }
  /*!
   * \brief seed random number generator using this seed
   * \param seed seed of prng
   */
  inline void Seed(int seed) {
#if MSHADOW_IN_CXX11
    rnd_engine_.seed(seed);
#endif
    this->rseed_ = static_cast<unsigned>(seed);
  }
  /*!
   * \brief get random seed used in random generator
   * \return seed in unsigned
   */
  inline unsigned GetSeed() const {
    return rseed_;
  }
  /*!
   * \brief set the stream of computation
   * \param stream computation stream
   */
  inline void set_stream(Stream<cpu> *stream) {
  }

// These samplers are only avail in C++11.
#if MSHADOW_IN_CXX11

  /*!
   * \brief get some random integer
   * \return integer as unsigned
   */
  inline unsigned GetRandInt() {
    return rnd_engine_();
  }

  /*!
   * \brief get a set of random integers
   */
  inline void GetRandInt(const Tensor<cpu, 1, unsigned>& dst) {
    std::generate_n(dst.dptr_, dst.size(0), [&](){ return rnd_engine_(); });
  }

  /*!
   * \brief generate data from a distribution
   * \param dst destination
   * \tparam dim dimension of tensor
   * \param sampler sampler of the distribution
   */
  template<int dim, class Sampler>
  inline void SampleDistribution(Tensor<cpu, dim, DType> *dst, Sampler sampler) {
    if (dst->CheckContiguous()) {
      std::generate_n(dst->dptr_, dst->shape_.Size(), sampler);
    } else {
      Tensor<cpu, 2, DType> mat = dst->FlatTo2D();
      for (index_t i = 0; i < mat.size(0); ++i) {
        std::generate_n(mat[i].dptr_, mat.size(1), sampler);
      }
    }
  }

  /*!
   * \brief generate data from uniform [a,b)
   * \param dst destination
   * \param a lower bound of uniform
   * \param b upper bound of uniform
   * \tparam dim dimension of tensor
   */
  template<int dim, typename PType>
  inline void SampleUniform(Tensor<cpu, dim, DType> *dst,
                            PType a = 0.0f , PType b = 1.0f ) {
    // Ensure that half_t is handled correctly.
    typedef typename std::conditional<std::is_floating_point<DType>::value,
                                      DType, double>::type FType;
    typedef typename std::conditional<std::is_integral<DType>::value,
                                      std::uniform_int_distribution<DType>,
                                      std::uniform_real_distribution<FType>>::type GType;
    GType dist_uniform(a, b);
    SampleDistribution(dst, [&](){ return dist_uniform(rnd_engine_);});
  }

  /*!
   * \brief generate data from standard gaussian
   * \param dst destination
   * \param mu mean variable
   * \param sigma standard deviation
   * \tparam dim dimension of tensor
   */
  template<int dim, typename PType>
  inline void SampleGaussian(Tensor<cpu, dim, DType> *dst,
                             PType mu = 0.0f, PType sigma = 1.0f ) {
    if (sigma <= 0) {
      *dst = mu; return;
    }
    typedef typename std::conditional<std::is_floating_point<DType>::value,
                                      DType, double>::type GType;
    std::normal_distribution<GType> dist_normal(mu, sigma);
    SampleDistribution(dst, [&](){ return dist_normal(rnd_engine_);});
  }

  /*!
   * \brief generate data from a gamma distribution
   * \param dst destination
   * \param alpha (shape) parameter
   * \param beta (scale) parameter
   * \tparam dim dimension of tensor
   */
  template<int dim, typename PType>
  inline void SampleGamma(Tensor<cpu, dim, DType> *dst,
                          PType alpha, PType beta) {
    typedef typename std::conditional<std::is_floating_point<DType>::value,
                                      DType, double>::type GType;
    std::gamma_distribution<GType> dist_gamma(alpha, beta);
    SampleDistribution(dst, [&](){ return dist_gamma(rnd_engine_);});
  }

  /*!
   * \brief generate data from an exponential distribution
   * \param dst destination
   * \param lambda parameter (rate) of the distribution
   * \tparam dim dimension of tensor
   */
  template<int dim, typename PType>
  inline void SampleExponential(Tensor<cpu, dim, DType> *dst, PType lambda ) {
    typedef typename std::conditional<std::is_floating_point<DType>::value,
                                      DType, double>::type GType;
    std::exponential_distribution<GType> dist_exp(lambda);
    SampleDistribution(dst, [&](){ return dist_exp(rnd_engine_);});
  }

  /*!
   * \brief generate data from a poisson distribution
   * \param dst destination
   * \param lambda parameter (rate) of the distribution
   * \tparam dim dimension of tensor
   */
  template<int dim, typename PType>
  inline void SamplePoisson(Tensor<cpu, dim, DType> *dst, PType lambda) {
    typedef typename std::conditional<std::is_integral<DType>::value, DType, int>::type GType;
    std::poisson_distribution<GType> dist_poisson(lambda);
    SampleDistribution(dst, [&](){ return static_cast<DType>(dist_poisson(rnd_engine_));});
  }

  /*!
   * \brief generate data from a negative binomial distribution
   * \param dst destination
   * \param k limit on number of failures
   * \param p success probability
   * \tparam dim dimension of tensor
   */
  template<int dim, typename PType1, typename PType2>
  inline void SampleNegativeBinomial(Tensor<cpu, dim, DType> *dst, PType1 k, PType2 p) {
    typedef typename std::conditional<std::is_integral<DType>::value, DType, int>::type GType;
    std::negative_binomial_distribution<GType> dist_negbinomial(k, p);
    SampleDistribution(dst, [&](){ return static_cast<DType>(dist_negbinomial(rnd_engine_));});
  }

  /*!
   * \brief generate data from a generalized negative binomial distribution
   * \param dst destination
   * \param mu parameter (mean) of the distribution
   * \param alpha parameter (over dispersion) of the distribution
   *   (for alpha=0 this gives a Poisson)
   * \tparam dim dimension of tensor
   */
  template<int dim, typename PType>
  inline void SampleGeneralizedNegativeBinomial(Tensor<cpu, dim, DType> *dst,
                                                PType mu, PType alpha) {
    if (alpha == PType(0)) {
      SamplePoisson(dst, mu);  // limit of Poisson
    } else {
      PType r(PType(1) / alpha);
      PType beta = mu * alpha;
      std::gamma_distribution<> dist_gamma(r, beta);
      typedef typename std::conditional<std::is_integral<DType>::value, DType, int>::type GType;
      SampleDistribution(dst,
        [&](){ std::poisson_distribution<GType> dist_poisson(dist_gamma(rnd_engine_));
               return static_cast<DType>(dist_poisson(rnd_engine_));});
    }
  }
#endif

  /*!
   * \brief return a temporal expression storing standard gaussian random variables
   *        the temporal tensor is only valid before next call of gaussian or uniform
   *        can be used as part of expression
   *  Caution: this means expression such as A = gaussian(s1) * gaussian(s2) will give invalid result,
   *           since second call of gaussian(s2) makes gaussian(s1) invalid
   *           A = gaussian(s1)*B+C; is correct; use one gaussian/uniform in each expression
   * \param shape shape of the tensor
   * \return a temporal expression storing standard gaussian random variables
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline expr::ReshapeExp<Tensor<cpu, 1, DType>, DType, dim, 1>
  gaussian(Shape<dim> shape) {
    buffer_.Resize(Shape1(shape.Size()));
    this->SampleGaussian(&buffer_, 0.0f, 1.0f);
    return expr::reshape(buffer_, shape);
  }
  /*!
   * \brief return a temporal expression storing standard uniform [0,1)
   *        the temporal tensor is only valid before next call of gaussian or uniform
   *        can be used as part of expression
   *  Caution: this means expression such as A = uniform(s1) * uniform(s2) will give invalid result,
   *           since second call of gaussian(s2) makes gaussian(s1) invalid
   *           A = gaussian(s1)*B+C; is correct; use one gaussian/uniform in each expression
   * \param shape shape of the tensor
   * \return a temporal expression storing standard uniform [0,1)
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline expr::ReshapeExp<Tensor<cpu, 1, DType>, DType, dim, 1>
  uniform(Shape<dim> shape) {
    buffer_.Resize(Shape1(shape.Size()));
    this->SampleUniform(&buffer_, 0.0f, 1.0f);
    return expr::reshape(buffer_, shape);
  }

  std::mt19937 &GetRndEngine() {
    return rnd_engine_;
  }

 private:
#if MSHADOW_IN_CXX11
  /*! \brief use c++11 random engine. */
  std::mt19937 rnd_engine_;
  /*! \brief random number seed used in random engine */
  unsigned rseed_;

#else

  /*! \brief random number seed used by PRNG */
  unsigned rseed_;
  // functions
  template<int dim>
  inline void SampleUniform(Tensor<cpu, dim, DType> *dst,
                            DType a = 0.0f, DType b = 1.0f) {
    if (dst->CheckContiguous()) {
      this->GenUniform(dst->dptr_, dst->shape_.Size(), a, b);
    } else {
      Tensor<cpu, 2, DType> mat = dst->FlatTo2D();
      for (index_t i = 0; i < mat.size(0); ++i) {
        this->GenUniform(mat[i].dptr_, mat.size(1), a, b);
      }
    }
  }
  template<int dim>
  inline void SampleGaussian(Tensor<cpu, dim, DType> *dst,
                             DType mu = 0.0f, DType sigma = 1.0f) {
    if (sigma <= 0.0f) {
      *dst = mu; return;
    }
    if (dst->CheckContiguous()) {
      this->GenGaussian(dst->dptr_, dst->shape_.Size(), mu, sigma);
    } else {
      Tensor<cpu, 2, DType> mat = dst->FlatTo2D();
      for (index_t i = 0; i < mat.size(0); ++i) {
        this->GenGaussian(mat[i].dptr_, mat.size(1), mu, sigma);
      }
    }
  }
  inline void GenUniform(float *dptr, index_t size, float a, float b) {
    for (index_t j = 0; j < size; ++j) {
      dptr[j] = static_cast<float>(RandNext()) * (b - a) + a;
    }
  }
  inline void GenUniform(double *dptr, index_t size, double a, double b) {
    for (index_t j = 0; j < size; ++j) {
      dptr[j] = static_cast<double>(RandNext()) * (b - a) + a;
    }
  }
  inline void GenGaussian(float *dptr, index_t size, float mu, float sigma) {
    this->GenGaussianX(dptr, size, mu, sigma);
  }
  inline void GenGaussian(double *dptr, index_t size, double mu, double sigma) {
    this->GenGaussianX(dptr, size, mu, sigma);
  }
  inline void GenGaussianX(DType *dptr, index_t size, DType mu, DType sigma) {
    DType g1 = 0.0f, g2 = 0.0f;
    for (index_t j = 0; j < size; ++j) {
      if ((j & 1) == 0) {
        this->SampleNormal2D(&g1, &g2);
        dptr[j] = mu + g1 * sigma;
      } else {
        dptr[j] = mu + g2 * sigma;
      }
    }
  }
  /*! \brief get next random number from rand */
  inline DType RandNext(void) {
    return static_cast<DType>(rand_r(&rseed_)) /
        (static_cast<DType>(RAND_MAX) + 1.0f);
  }
  /*! \brief return a real numer uniform in (0,1) */
  inline DType RandNext2(void) {
    return (static_cast<DType>(rand_r(&rseed_)) + 1.0f) /
        (static_cast<DType>(RAND_MAX) + 2.0f);
  }
  /*!
   * \brief sample iid xx,yy ~N(0,1)
   * \param xx first  gaussian output
   * \param yy second gaussian output
   */
  inline void SampleNormal2D(DType *xx_, DType *yy_) {
    DType &xx = *xx_, &yy = *yy_;
    DType x, y, s;
    do {
      x = 2.0f * RandNext2() - 1.0f;
      y = 2.0f * RandNext2() - 1.0f;
      s = x * x + y * y;
    } while (s >= 1.0f || s == 0.0f);
    DType t = std::sqrt(-2.0f * std::log(s) / s);
    xx = x * t; yy = y * t;
  }
#endif
  /*! \brief temporal space used to store random numbers */
  TensorContainer<cpu, 1, DType> buffer_;
};  // class Random<cpu, DType>

// only allow GPU PRNG when cuda is enabled
#if MSHADOW_USE_CUDA
/*! \brief GPU random number generator */
template<typename DType>
class Random<gpu, DType> {
 public:
  /*!
   * \brief constructor of random engine
   * \param seed random number seed
   */
  explicit Random(int seed) : gen_(NULL) {
    this->Seed(seed);
    buffer_.Resize(Shape1(kRandBufferSize));
  }
  ~Random(void) MSHADOW_THROW_EXCEPTION {
    DeleteGenerator();
  }
  /*!
   * \brief set the stream of computation
   * \param stream computation stream
   */
  inline void set_stream(Stream<gpu> *stream) {
    curandStatus_t status;
    status = curandSetStream(gen_, Stream<gpu>::GetStream(stream));

    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << "set_stream CURAND failed";
  }
  /*!
   * \brief seed random number generator using this seed
   * \param seed seed of prng
   */
  inline void Seed(int seed) {
    // Create a new rng, either initially or if the RNG type can't reset its offset.
    if (gen_ == NULL || (curandSetGeneratorOffset(gen_, 0ULL) != CURAND_STATUS_SUCCESS))
      CreateGenerator();
    // Now set the seed.
    curandStatus_t status;
    status = curandSetPseudoRandomGeneratorSeed(gen_, static_cast<uint64_t>(seed));
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << "Set CURAND seed failed.";
  }
  /*!
   * \brief get a set of random integers
   */
  inline void GetRandInt(const Tensor<gpu, 1, unsigned>& dst) {
    curandStatus_t status = curandGenerate(gen_, dst.dptr_, dst.size(0));
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << "CURAND Gen rand ints failed.";
  }
  /*!
   * \brief generate data from uniform [a,b)
   * \param dst destination
   * \param a lower bound of uniform
   * \param b upper bound of uniform
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline void SampleUniform(Tensor<gpu, dim, DType> *dst,
                            DType a = 0.0f, DType b = 1.0f);

  /*!
   * \brief generate data from standard gaussian
   * \param dst destination
   * \param mu mean variable
   * \param sigma standard deviation
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline void SampleGaussian(Tensor<gpu, dim, DType> *dst,
                             DType mu = 0.0f, DType sigma = 1.0f);
  /*!
   * \brief return a temporal expression storing standard gaussian random variables
   *        the temporal tensor is only valid before next call of gaussian or uniform
   *        can be used as part of expression
   *  Caution: this means expression such as A = gaussian(s1) * gaussian(s2) will give invalid result,
   *           since second call of gaussian(s2) makes gaussian(s1) invalid
   *           A = gaussian(s1)*B+C; is correct; use one gaussian/uniform in each expression
   * \param shape shape of the tensor
   * \param mu mean
   * \param sigma variance
   * \return a temporal expression storing standard gaussian random variables
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline expr::ReshapeExp<Tensor<gpu, 1, DType>, DType, dim, 1>
  gaussian(Shape<dim> shape, DType mu = 0.0f, DType sigma = 1.0f);
  /*!
   * \brief return a temporal expression storing standard uniform [0,1)
   *        the temporal tensor is only valid before next call of gaussian or uniform
   *        can be used as part of expression
   *  Caution: this means expression such as A = gaussian(s1) * gaussian(s2) will give invalid result,
   *           since second call of gaussian(s2) makes gaussian(s1) invalid
   *           A = gaussian(s1)*B+C; is correct; use one gaussian/uniform in each expression
   * \param shape shape of the tensor
   * \return a temporal expression storing standard uniform [0,1)
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline expr::ReshapeExp<Tensor<gpu, 1, DType>, DType, dim, 1>
  uniform(Shape<dim> shape);

 private:
  inline void GenGaussian(float *dptr, size_t size, float mu, float sigma) {
    curandStatus_t status;
    status = curandGenerateNormal(gen_, dptr, size, mu, sigma);
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << "CURAND Gen Normal float failed."
                                            << " size = " << size
                                            << ",mu = " << mu
                                            << ",sigma = " << sigma;
  }
  inline void GenGaussian(double *dptr, size_t size, double mu, double sigma) {
    curandStatus_t status;
    status = curandGenerateNormalDouble(gen_, dptr, size, mu, sigma);
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << "CURAND Gen Normal double failed."
                                            << " size = " << size
                                            << ",mu = " << mu
                                            << ",sigma = " << sigma;
  }
  inline void GenUniform(float *dptr, size_t size) {
    curandStatus_t status;
    status = curandGenerateUniform(gen_, dptr, size);
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << "CURAND Gen Uniform float failed."
                                            << " size = " << size;
  }
  inline void GenUniform(double *dptr, size_t size) {
    curandStatus_t status;
    status = curandGenerateUniformDouble(gen_, dptr, size);
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << "CURAND Gen Uniform double failed."
                                            << " size = " << size;
  }
  inline void CreateGenerator() {
    if (gen_ != NULL)
      DeleteGenerator();
    curandStatus_t status;
    status = curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT);
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << "Cannot create CURAND Generator";
  }
  inline void DeleteGenerator() {
    if (gen_ != NULL) {
      curandStatus_t status;
      status = curandDestroyGenerator(gen_);
      CHECK_EQ(status, CURAND_STATUS_SUCCESS) << "Destory CURAND Gen failed";
      gen_ = NULL;
    }
  }
  /*! \brief random number generator */
  curandGenerator_t gen_;
  /*! \brief templ buffer */
  TensorContainer<gpu, 1, DType> buffer_;
};  // class Random<gpu, DType>
#endif  // MSHADOW_USE_CUDA

#ifdef __CUDACC__
// implementations that depends on cuda kernels
template<typename DType>
template<int dim>
inline void Random<gpu, DType>::SampleUniform(
    Tensor<gpu, dim, DType> *dst, DType a, DType b) {
  if (a == 0.0f && b == 1.0f) {
    if (dst->CheckContiguous()) {
      this->GenUniform(dst->dptr_, dst->shape_.Size());
    } else {
      *dst = this->uniform(dst->shape_);
    }
  } else {
    *dst = this->uniform(dst->shape_) * (b - a) + a;
  }
}
template<typename DType>
template<int dim>
inline void Random<gpu, DType>::SampleGaussian(
    Tensor<gpu, dim, DType> *dst, DType mu, DType sigma) {
  // We need to check whether the shape size is even since CuRand supports only normal distribution
  // generation of even number of elements.
  if (dst->CheckContiguous() && (dst->shape_.Size() % 2 == 0)) {
    this->GenGaussian(dst->dptr_, dst->shape_.Size(), mu, sigma);
  } else {
    *dst = this->gaussian(dst->shape_, mu, sigma);
  }
}

template<typename DType>
template<int dim>
inline expr::ReshapeExp<Tensor<gpu, 1, DType>, DType, dim, 1>
Random<gpu, DType>::gaussian(Shape<dim> shape, DType mu, DType sigma) {
  size_t aligned_sz = ((shape.Size() + 1UL) >> 1) << 1;
  // allocate alligned size
  buffer_.Resize(Shape1(aligned_sz));
  buffer_.Resize(Shape1(shape.Size()));
  this->GenGaussian(buffer_.dptr_, aligned_sz, mu, sigma);
  return expr::reshape(buffer_, shape);
}

template<typename DType>
template<int dim>
inline expr::ReshapeExp<Tensor<gpu, 1, DType>, DType, dim, 1>
Random<gpu, DType>::uniform(Shape<dim> shape) {
  buffer_.Resize(Shape1(shape.Size()));
  this->GenUniform(buffer_.dptr_, buffer_.size(0));
  return expr::reshape(buffer_, shape);
}
#endif  // __CUDACC__
}  // namespace mshadow
#endif  // MSHADOW_RANDOM_H_
