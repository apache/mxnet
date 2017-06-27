#ifndef CHECK_H_
#define CHECK_H_

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

void checkValue(SEXP x, const std::string varName = "x", const int RTYPE = 14, const int len = -1);
void checkValueNum(SEXP x, const std::string varName = "x", const int len = -1);
void checkValueInt(const double& x, const std::string varName = "x", bool positive = false, bool zero = true);

#endif
