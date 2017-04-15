#include "common.h"
#include "interp.h"
#include "checkValue.h"
#include <RcppArmadillo.h>
#include <string>
#include <Rinternals.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' 1-D data interpolation.
//'
//' Returns interpolated values of a 1-D function at specific query points using linear interpolation.
//' The extrapolation is used, please be caution in using the values which \code{xi} is larger than
//' \code{max(x)} and smaller than \code{min(x)}.
//'
//' @param x A vector with n elements, \code{x[i]} is a support, \code{i = 1, ..., n}.
//'   If \code{x} is not sorted, it will be sorted. If \code{x} is not unique, the corresponding \code{y} values
//'   will be averaged.
//' @param y If \code{y} is vector, the length of \code{y} must be equal to the lenght of \code{x}.
//'   If \code{y} is matrix, the number of rows or the number of columns must be equal to the lenght of \code{x}.
//'   If the number of rows is equal to the lenght of \code{x, y[i, j]} is jth values on corresponding
//'   value of \code{x[i], i = 1, ..., n}.
//' @param xi A vector with m elements, \code{xi[k]} is the point which you want to interpolate,
//'   \code{k = 1, ..., m}.
//' @param method A string "linear" or "spline", the method of interpolation.
//' @return A vector or matrix (depends on \code{y}) with the interpolated values corresponding to
//'   \code{xi}.
//' @section Reference:
//' Cleve Moler, Numerical Computing with MATLAB, chapter 3,
//'   \url{http://www.mathworks.com/moler/index_ncm.html}. \cr
//' Nir Krakauer, Paul Kienzle, VZLU Prague, interp1, Octave.
//' @examples
//' library(lattice)
//' plot_res <- function(x, y, xi, yl, ys){
//'   xyplot(y ~ x, data.frame(x, y), pch = 16, col = "black", cex = 1.2,
//'          xlab = "", ylab = "", main = "Results of Interpolation",
//'          panel = function(x, y, ...){
//'            panel.xyplot(x, y, ...)
//'            panel.xyplot(xi, yl, "l", col = "red")
//'            panel.xyplot(xi, ys, "l", col = "blue")
//'          }, key = simpleKey(c("linear", "spline"), points = FALSE,
//'                             lines = TRUE, columns = 2))
//' }
//' x <- c(0.8, 0.3, 0.1, 0.6, 0.9, 0.5, 0.2, 0.0, 0.7, 1.0, 0.4)
//' y <- matrix(c(x**2 - 0.6*x, 0.2*x**3 - 0.6*x**2 + 0.5*x), length(x))
//' xi <- seq(0, 1, len=81)
//' yl <- interp1(x, y, xi, 'linear')
//' ys <- interp1(x, y, xi, 'spline')
//' plot_res(x, y[,1], xi, yl[,1], ys[,1])
//' plot_res(x, y[,2], xi, yl[,2], ys[,2])
//'
//' x <- seq(0, 2*pi, pi/4)
//' y <- sin(x)
//' xi <- seq(0, 2*pi, pi/16)
//' yl <- interp1(x, as.matrix(y), xi, 'linear')
//' ys <- interp1(x, as.matrix(y), xi, 'spline')
//' plot_res(x, y, xi, yl, ys)
//' @export
// [[Rcpp::export]]
SEXP interp1(SEXP xr, SEXP yr, SEXP xir, std::string method = "linear") {
  // check data
  checkValueNum(xr, "x");
  checkValueNum(yr, "y");
  checkValueNum(xir, "xi");

  arma::vec x = Rcpp::as<arma::vec>(xr);
  arma::mat y = Rcpp::as<arma::mat>(asMatrix(yr));
  arma::vec xi = Rcpp::as<arma::vec>(xir);
  return asVector(interp1Func(x, y, xi, method));
}

//' 2-D data interpolation.
//'
//' Returns interpolated values of a 2-D function at specific query points using
//' linear interpolation. The extrapolation is used, please be caution in using the
//' values which \code{xi} is larger than \code{max(x)/max(y)} and smaller than \code{min(x)/min(y)}.
//'
//' @param x A vector with n1 elements, \code{x[i]} is a support, \code{i = 1, ..., n1}.
//'   If \code{x} is not sorted, it will be sorted. If \code{x} is not unique, the corresponding \code{v} values
//'   will be averaged.
//' @param y A vector with n2 elements, \code{y[j]} is a support, \code{j = 1, ..., n2}.
//'   If \code{y} is not sorted, it will be sorted. If \code{y} is not unique, the corresponding \code{v} values
//'   will be averaged.
//' @param v A matrix with size n1 by n2, \code{v[i, j]} is the corresponding value at grid \code{(x[i], y[j])}.
//' @param xi A vector with m elements, \code{xi[k]} is the point which you want to interpolate,
//'   \code{k = 1, ..., m1}.
//' @param yi A vector with m elements, \code{yi[l]} is the point which you want to interpolate,
//'   \code{l = 1, ..., m2}.
//' @param method A string "linear" or "spline", the method of interpolation.
//' @return A matrix with the interpolated values corresponding to \code{xi} and \code{yi}.
//' @section Reference:
//' Cleve Moler, Numerical Computing with MATLAB, chapter 3,
//'   \url{http://www.mathworks.com/moler/index_ncm.html}. \cr
//' Kai Habel, Jaroslav Hajek, interp2, Octave.
//' @examples
//' # example in MatLab
//' library(lattice)
//' # data generation
//' x <- seq(-3, 3, 1)
//' xm <- expand.grid(x, x)
//' z <- 3*(1-xm[,1])^2.*exp(-(xm[,1]^2) - (xm[,2]+1)^2) -
//'   10*(xm[,1]/5 - xm[,1]^3 - xm[,2]^5)*exp(-xm[,1]^2-xm[,2]^2) -
//'   1/3*exp(-(xm[,1]+1)^2 - xm[,2]^2)
//' dat <- data.frame(xm, z)
//' # graph of original data
//' wireframe(z ~ Var1 + Var2, dat, drape = TRUE, colorkey = TRUE)
//'
//' xi <- seq(-3, 3, 0.25)
//' zi_l <- interp2(x, x, matrix(z, length(x)), xi, xi, 'linear')
//' dat_l <- cbind(expand.grid(x = xi, y = xi), z = as.vector(zi_l))
//' # graph of linearly interpolation
//' wireframe(z ~ x + y, dat_l, drape = TRUE, colorkey = TRUE)
//'
//' zi_s <- interp2(x, x, matrix(z, length(x)), xi, xi, 'spline')
//' dat_s <- cbind(expand.grid(x = xi, y = xi), z = as.vector(zi_s))
//' # graph of interpolation with spline
//' wireframe(z ~ x + y, dat_s, drape = TRUE, colorkey = TRUE)
//' @export
// [[Rcpp::export]]
arma::mat interp2(SEXP xr, SEXP yr, SEXP vr, SEXP xir, SEXP yir, std::string method = "linear") {
  // check data
  checkValueNum(xr, "x");
  checkValueNum(yr, "y");
  checkValueNum(vr, "v");
  checkValueNum(xir, "xi");
  checkValueNum(yir, "yi");

  arma::vec x = Rcpp::as<arma::vec>(xr);
  arma::vec y = Rcpp::as<arma::vec>(yr);
  arma::mat v = Rcpp::as<arma::mat>(vr);
  arma::vec xi = Rcpp::as<arma::vec>(xir);
  arma::vec yi = Rcpp::as<arma::vec>(yir);
  return interp2Func(x, y, v, xi, yi, method);
}
