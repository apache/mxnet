#include <string>
#include <sstream>
#include <Rcpp.h>

template <typename T>
std::string num2str(T Number) {
  std::ostringstream ss;
  ss << Number;
  return ss.str();
}

// [[Rcpp::export]]
void checkValue(SEXP x, const std::string varName = "x", const int RTYPE = 14, const int len = -1) {
  int n = LENGTH(x);
  if (len > 0) {
    if (n != len)
      Rcpp::stop("The length of " + varName + " must be " + num2str(len) + "!\n");
  }
  if (TYPEOF(x) != RTYPE) {
    switch(RTYPE) {
    case LGLSXP:
      Rcpp::stop(varName + " must be logical type!\n");
    case INTSXP:
      Rcpp::stop(varName + " must be integer type!\n");
    case REALSXP:
      Rcpp::stop(varName + " must be double type!\n");
    case STRSXP:
      Rcpp::stop(varName + " must be string type!\n");
    case CPLXSXP:
      Rcpp::stop(varName + " must be complex type!\n");
    default:
      Rcpp::stop("Not supported type!\n");
    }
  }
  for (int i = 0; i < n; i++) {
    switch(TYPEOF(x)) {
    case LGLSXP:
      if (LOGICAL(x)[i] == NA_LOGICAL)
        Rcpp::stop(varName + " must not contain NA!\n");
      break;
    case INTSXP:
      if (INTEGER(x)[i] == NA_INTEGER)
        Rcpp::stop(varName + " must not contain NA!\n");
      break;
    case REALSXP:
      if (ISNA(REAL(x)[i]) || ISNAN(REAL(x)[i]) || !R_FINITE(REAL(x)[i]))
        Rcpp::stop(varName + " must not contain NA, NaN or Inf!\n");
      break;
    case STRSXP:
      if (STRING_ELT(x, i) == NA_STRING)
        Rcpp::stop(varName + " must not contain NA!\n");
      break;
    case CPLXSXP:
      if (ISNA(COMPLEX(x)[i].r) || ISNAN(COMPLEX(x)[i].r) || !R_FINITE(COMPLEX(x)[i].r) ||
          ISNA(COMPLEX(x)[i].i) || ISNAN(COMPLEX(x)[i].i) || !R_FINITE(COMPLEX(x)[i].i))
        Rcpp::stop(varName + " must not contain NA, NaN or Inf!\n");
      break;
    default:
      Rcpp::stop("Not supported type!\n");
    }
  }
}

// [[Rcpp::export]]
void checkValueNum(SEXP x, const std::string varName = "x", const int len = -1) {
  if (TYPEOF(x) == INTSXP) {
    checkValue(x, varName, INTSXP, len);
  } else if (TYPEOF(x) == REALSXP) {
    checkValue(x, varName, REALSXP, len);
  }
}

// [[Rcpp::export]]
void checkValueInt(const double& x, const std::string varName = "x", bool positive = false, bool zero = true) {
  if (ISNA(x) || ISNAN(x) || !R_FINITE(x) || std::abs(x - std::floor(x)) > 1e-6)
    Rcpp::stop(varName + " cannot be NA, NaN or Inf and must be a integer!\n");

  if (positive && zero && x < 0) {
    Rcpp::stop(varName + " must be a positive integer.\n");
  } else if (positive && !zero && x <= 0) {
    Rcpp::stop(varName + " must be a non-negative integer.\n");
  }
}
