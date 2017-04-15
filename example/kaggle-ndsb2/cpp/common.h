#ifndef COMMON_H_
#define COMMON_H_

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

void RMessage(const std::string& msg);
SEXP asMatrix(SEXP x);
SEXP asVector(arma::mat x);
arma::umat lookup(const arma::vec& edges, const arma::mat& x);
#endif
