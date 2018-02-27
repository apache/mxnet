/*!
 * Copyright (c) 2015 by Contributors
 * \file special_functions-inl.h
 * \brief
 * \author Valentin Flunkert
*/


#ifndef MXNET_OPERATOR_SPECIAL_FUNCTIONS_INL_H_
#define MXNET_OPERATOR_SPECIAL_FUNCTIONS_INL_H_

#include "math_functions-inl.h"

namespace mxnet {
namespace op {

namespace special_functions {

template<typename DType>
struct helper_numeric_limits {
  MSHADOW_XINLINE static DType max();
};

template<>
struct helper_numeric_limits<double> {
  MSHADOW_XINLINE static double max() {
    return DBL_MAX;
  }
};

template<>
struct helper_numeric_limits<float> {
  MSHADOW_XINLINE static double max() {
    return FLT_MAX;
  }
};


// This code is based on the Cephes Library availible at http://www.netlib.org/cephes
// The original author, Stephen Moshier, has kindly given permission to use this code
// in mxnet.  (See email below).
//
//     Date: Tue, 13 Sep 2016 09:28:20 -0400
//     From: Stephen Moshier
//     To: Flunkert, Valentin
//     Subject: Re: cephes code in mxnet
//
//     Hello Valentin,
//
//     Thank you for writing.  You are welcome to use and modify the Cephes code
//     and distribute it under the Apache license.
//
//     Good luck with your project,
//     Steve Moshier
//
// Cephes Math Library Release 2.2:  June, 1992
// Copyright 1984, 1987, 1992 by Stephen L. Moshier
// Direct inquiries to 30 Frost Street, Cambridge, MA 02140
//
struct cephes {
  /*
   * Helper to evaluate a polynomial given an array of coefficients.
   */
  template <typename DType>
  MSHADOW_XINLINE static DType polevl(DType x, const DType coef[], int N) {
    DType ans;
    DType const *p;
    int i;

    p = coef;
    ans = *p++;

    i = N;
    do {
      ans = ans * x  +  *p++;
    } while ( --i );

    return( ans );
  }


  /*
   * Helper function for psi that handles double/float specific differences
   * in the algorithm.
   */
  template<typename DType>
  MSHADOW_XINLINE static DType psi_helper(DType s);

  /*
   *
   *    Psi (digamma) function
   *
   *
   * SYNOPSIS:
   *
   * float x, y, psif();
   *
   * y = psif( x );
   *
   *
   * DESCRIPTION:
   *
   *              d      -
   *   psi(x)  =  -- ln | (x)
   *              dx
   *
   * is the logarithmic derivative of the gamma function.
   * For integer x,
   *                   n-1
   *                    -
   * psi(n) = -EUL  +   >  1/k.
   *                    -
   *                   k=1
   *
   * This formula is used for 0 < n <= 10.  If x is negative, it
   * is transformed to a positive argument by the reflection
   * formula  psi(1-x) = psi(x) + pi cot(pi x).
   * For general positive x, the argument is made greater than 10
   * using the recurrence  psi(x+1) = psi(x) + 1/x.
   * Then the following asymptotic expansion is applied:
   *
   *                           inf.   B
   *                            -      2k
   * psi(x) = log(x) - 1/2x -   >   -------
   *                            -        2k
   *                           k=1   2k x
   *
   * where the B2k are Bernoulli numbers.
   *
   * ACCURACY:
   *    Absolute error,  relative when |psi| > 1 :
   * arithmetic   domain     # trials      peak         rms
   *    IEEE      -33,0        30000      8.2e-7      1.2e-7
   *    IEEE      0,33        100000      7.3e-7      7.7e-8
   *
   * ERROR MESSAGES:
   *     message         condition      value returned
   * psi singularity    x integer <=0      MAXNUMF
   */
  template<typename DType>
  MSHADOW_XINLINE static DType psi(DType x) {
    DType p, q, nz, s, w, y;
    int i, n, negative;

    DType EUL(0.57721566490153286061);
    DType PI(3.14159265358979323846);

    negative = 0;
    nz = 0.0;

    if ( x <= 0.0 ) {
      negative = 1;
      q = x;
      p = std::floor(q);
      if ( p == q ) {
        return helper_numeric_limits<double>::max();
      }
      /* Remove the zeros of tan(PI x)
       * by subtracting the nearest integer from x
       */
      nz = q - p;
      if ( nz != 0.5 ) {
        if ( nz > 0.5 ) {
          p += 1.0;
          nz = q - p;
        }
        nz = PI/std::tan(PI*nz);
      } else {
        nz = 0.0;
      }
      x = 1.0 - x;
    }

    /* check for positive integer up to 10 */
    if ( (x <= 10.0) && (x == std::floor(x)) ) {
      y = 0.0;
      n = x;
      for ( i = 1; i < n; i++ ) {
        w = i;
        y += 1.0/w;
      }
      y -= EUL;
      goto done;
    }

    s = x;
    w = 0.0;
    while ( s < 10.0 ) {
      w += 1.0/s;
      s += 1.0;
    }

    y = psi_helper(s);

    y = logf(s)  -  (0.5/s)  -  y  -  w;

done:

    if ( negative ) {
      y -= nz;
    }

    return(y);
  }
};


template<>
MSHADOW_XINLINE double cephes::psi_helper<double>(double s) {
  double z;
  const double A[] = {
    8.33333333333333333333E-2,
    -2.10927960927960927961E-2,
    7.57575757575757575758E-3,
    -4.16666666666666666667E-3,
    3.96825396825396825397E-3,
    -8.33333333333333333333E-3,
    8.33333333333333333333E-2
  };

  if ( s < 1.0e17 ) {
    z = 1.0/(s * s);
    return z * cephes::polevl<double>(z, A, 6);
  } else {
    return 0.0;
  }
}

template<>
MSHADOW_XINLINE float cephes::psi_helper<float>(float s) {
  float z;
  const float A[] = {
    -4.16666666666666666667E-3f,
    3.96825396825396825397E-3f,
    -8.33333333333333333333E-3f,
    8.33333333333333333333E-2f
  };

  if ( s < 1.0e8 ) {
    z = 1.0/(s * s);
    return z * cephes::polevl<float>(z, A, 3);
  } else {
    return 0.0;
  }
}


// This is code extracted from ApBsInT, available at
//    https://github.com/mseeger/apbsint
// The author of ApBsInT, Matthias Seeger (mseeger@gmail.com) is the same one
// who ported the code to MXNet.
//
// NOTE: Instantiate this with DType in {float, double}, nothing else will work!

// Some constants used in apbsint struct below
template<typename DType>
struct apbsint_const {
  MSHADOW_XINLINE static DType m_ln2pi() {
    return DType(1.83787706640934533908193770913);
  }
  MSHADOW_XINLINE static DType m_ln2() {
    return DType(0.69314718055994530941723212146);
  }
  MSHADOW_XINLINE static DType m_sqrtpi() {
    return DType(1.77245385090551602729816748334);
  }
  MSHADOW_XINLINE static DType m_sqrt2() {
    return DType(1.41421356237309504880168872421);
  }
  MSHADOW_XINLINE static DType erf_cody_limit1() {
    return DType(0.6629);
  }
  MSHADOW_XINLINE static DType erf_cody_limit2() {
    return DType(5.6569);
  }
};

struct apbsint {
  // Internal helpers

  /**
   * For x >= erf_cody_limit1(), define Q(x) by
   *   1 - Phi(x) approx N(x) x^{-1} Q(x).
   * We compute Q(x) according to
   *   Cody
   *   Rational Chebyshev approximation to the error function
   * This is done differently for x >= erf_cody_limit2() and
   * erf_cody_limit1() <= x < erf_cody_limit2().
   * NOTE: Q(x) -> 1 for x->infty.
   */
  template<typename DType>
  MSHADOW_XINLINE static DType erfRationalHelper(DType x) {
    int i;
    DType res, den, y;

    // MYASS(x>0.0);
    if (x >= apbsint_const<DType>::erf_cody_limit2()) {
      // x/sqrt(2) >= 4
      // Q(x)   = 1 + sqrt(pi) y R_1(y),
      // R_1(y) = poly(p_j,y) / poly(q_j,y),   y = 2/x^2
      // Ordering of arrays: 4,3,2,1,0,5 (only for numerator p_j; q_5=1)
      // ATTENTION: The p_j are negative of the entries here
      DType p[] = {3.05326634961232344e-1, 3.60344899949804439e-1,
                   1.25781726111229246e-1, 1.60837851487422766e-2,
                   6.58749161529837803e-4, 1.63153871373020978e-2};
      DType q[] = {2.56852019228982242,    1.87295284992346047,
                   5.27905102951428412e-1, 6.05183413124413191e-2,
                   2.33520497626869185e-3};
      y = 2.0/x/x;
      res = y*p[5];
      den = y;
      for (i = 0; i < 4; i++) {
        res = (res + p[i])*y;
        den = (den + q[i])*y;
      }
      // Minus, because p[j] values have to be negated
      res = 1.0 - apbsint_const<DType>::m_sqrtpi()*y*(res + p[4])/(den + q[4]);
    } else {
      // x/sqrt(2) < 4, x/sqrt(2) >= 0.469
      // Q(x)   = sqrt(pi) y R_2(y),
      // R_2(y) = poly(p_j,y) / poly(q_j,y),   y = x/sqrt(2)
      // Ordering of arrays: 7,6,5,4,3,2,1,0,8 (only p_8; q_8=1)
      DType p[] = {5.64188496988670089e-1, 8.88314979438837594,
                   6.61191906371416295e+1, 2.98635138197400131e+2,
                   8.81952221241769090e+2, 1.71204761263407058e+3,
                   2.05107837782607147e+3, 1.23033935479799725e+3,
                   2.15311535474403846e-8};
      DType q[] = {1.57449261107098347e+1, 1.17693950891312499e+2,
                   5.37181101862009858e+2, 1.62138957456669019e+3,
                   3.29079923573345963e+3, 4.36261909014324716e+3,
                   3.43936767414372164e+3, 1.23033935480374942e+3};
      y = x/apbsint_const<DType>::m_sqrt2();
      res = y*p[8];
      den = y;
      for (i = 0; i < 7; i++) {
        res = (res + p[i])*y;
        den = (den + q[i])*y;
      }
      res = apbsint_const<DType>::m_sqrtpi()*y*(res + p[7])/(den + q[7]);
    }

    return res;
  }

  /**
   * Implements rational function R_3(y),  y = x^2/2,
   * which is used if 0 <= x < erf_cody_limit1(). In this range:
   *   Phi(x) approx (1 + (x/sqrt(2)) R_3(x^2/2))/2
   * See
   *   Cody
   *   Rational Chebyshev approximation to the error function
   */
  template<typename DType>
  MSHADOW_XINLINE static DType erfRationalHelperR3(DType y) {
    int i;
    DType nom, den;

    // MYASS(y>=0.0);
    // R_3(y) = poly(p_j,y) / poly(q_j,y)
    // Ordering of arrays: 3,2,1,0,4 (only for p_5; q_5=1)
    DType p[] = {3.16112374387056560,    1.13864154151050156e+2,
                 3.77485237685302021e+2, 3.20937758913846947e+3,
                 1.85777706184603153e-1};
    DType q[] = {2.36012909523441209e+1, 2.44024637934444173e+2,
                 1.28261652607737228e+3, 2.84423683343917062e+3};
    nom = y*p[4];
    den = y;
    for (i = 0; i < 3; i++) {
      nom = (nom + p[i])*y;
      den = (den + q[i])*y;
    }

    return (nom + p[3])/(den + q[3]);
  }

  // Exported functions

  /**
   * @param z Argument
   * @return  log N(z|0,1)
   */
  template<typename DType>
  MSHADOW_XINLINE static DType logPdfNormal(DType z) {
    return -0.5 * (apbsint_const<DType>::m_ln2pi() + z*z);
  }

  /**
   * If Phi(z) denotes the c.d.f. of N(0,1), this method computes
   * log Phi(z).
   *
   * @param z Argument
   * @return  log Phi(z)
   */
  template<typename DType>
  MSHADOW_XINLINE static DType logCdfNormal(DType z) {
    DType res;

    if (math::fabs(z) < apbsint_const<DType>::erf_cody_limit1()) {
      // Part 3 approximation:
      // Phi(z) approx (1 + y R_3(y^2))/2, y = z/sqrt(2)
      res = math::log1p((z/apbsint_const<DType>::m_sqrt2())*erfRationalHelperR3(0.5*z*z))
        - apbsint_const<DType>::m_ln2();
    } else {
      // Part 1 or 2 approximation:
      // Phi(z) approx N(z) Q(-z)/(-z), z < 0
      // NOTE: The case z >= erf_cody_limit1() is uncritical, we could even use
      // a cheaper approximation then
      if (z < 0.0)
        res = logPdfNormal(z) - math::log(-z) + math::log(erfRationalHelper(-z));
      else
        res = math::log1p(-math::exp(logPdfNormal(z))*erfRationalHelper(z)/z);
    }

    return res;
  }

  /**
   * If Phi(z) denotes the c.d.f. of N(0,1), this method computes
   *   f(z) = (d/dz) log Phi(z) = N(z)/Phi(z).
   * NOTE: The technical report defines the hazard function
   *   h(x) = N(x)/(1 - Phi(x)).
   * This method computes h(-z).
   *
   * @param z Argument
   * @return  (d/dz) log Phi(z)
   */
  template<typename DType>
  MSHADOW_XINLINE static DType derivLogCdfNormal(DType z) {
    DType res;

    if (math::fabs(z) < apbsint_const<DType>::erf_cody_limit1()) {
      // Part 3 approximation:
      // Phi(z) approx (1 + y R_3(y^2))/2, y = z/sqrt(2)
      res = 2.0 * math::exp(logPdfNormal(z)) /
        (1.0 + (z/apbsint_const<DType>::m_sqrt2())*erfRationalHelperR3(0.5*z*z));
    } else {
      // Part 1 or 2:
      // Phi(z) approx N(z) Q(-z)/(-z), z<0
      // NOTE: The case z >= erf_cody_limit1() is uncritical, we could even use
      // a cheaper approximation then
      if (z < 0.0) {
        res = -z/erfRationalHelper(-z);
      } else {
        DType temp = math::exp(logPdfNormal(z));
        res = temp / (1.0 - temp*erfRationalHelper(z)/z);
      }
    }

    return res;
  }
};

}  // namespace special_functions
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SPECIAL_FUNCTIONS_INL_H_
