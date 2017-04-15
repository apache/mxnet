#include "common.h"
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

void RMessage(const std::string& msg) {
  Rcpp::Function messageFunc("message");
  messageFunc(msg);
}

SEXP asMatrix(SEXP x) {
  SEXP xDim = Rf_getAttrib(x, R_DimSymbol);
  if (Rf_isNull(xDim)) {
    SEXP x2 = PROTECT(Rf_duplicate(x));
    UNPROTECT(1);
    arma::Col<int> dim(2);
    dim << LENGTH(x) << 1 << arma::endr;
    Rf_setAttrib(x2, R_DimSymbol, Rcpp::wrap(dim));
    return(x2);
  } else {
    return(x);
  }
}

SEXP asVector(arma::mat x) {
  if (x.n_cols == 1 || x.n_rows == 1) {
    SEXP out = Rcpp::wrap(x);
    Rf_setAttrib(out, R_DimSymbol, R_NilValue);
    return out;
  } else {
    return Rcpp::wrap(x);
  }
}

arma::umat lookup(const arma::vec& edges, const arma::mat& x) {
  if (!edges.is_sorted())
    Rcpp::stop("edges is not strictly monotonic increasing.");

  arma::umat idx(size(x));
  const double* pos;
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    pos = std::upper_bound(edges.begin(), edges.end(), x(i));
    idx(i) = std::distance(edges.begin(), pos);
  }
  return idx;
}
