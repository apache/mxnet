#ifndef INTERP_H_
#define INTERP_H_

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

arma::mat splineFunc(const arma::vec& x, const arma::mat& y, const arma::vec& xi);
arma::mat interp1Func(const arma::vec& x, const arma::mat& y, const arma::vec& xi, const std::string& method);
arma::mat interp2Func(const arma::vec& x, const arma::vec& y, const arma::mat& v,
                      const arma::vec& xi, const arma::vec& yi, const std::string& method);
#endif
