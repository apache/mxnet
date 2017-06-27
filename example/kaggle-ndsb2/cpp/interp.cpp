#include "common.h"
#include "checkValue.h"
#include <RcppArmadillo.h>
#include <string>
// [[Rcpp::depends(RcppArmadillo)]]

arma::mat splineFunc(const arma::vec& x, const arma::mat& y, const arma::vec& xi) {
  arma::uword n = x.n_elem;
  if (n < 2)
    Rcpp::stop("spline: requires at least two non-NA values.");

  arma::mat a = y;
  arma::uvec szy(2);
  szy << y.n_rows << y.n_cols << arma::endr;
  if (szy(1) == n && szy(0) != n+2 && szy(0) != n && szy(0) >= 1)
    arma::inplace_trans(a);
  if (szy(1) == n+2 && szy(0) != n+2 && szy(0) != n && szy(0) >= 1)
    arma::inplace_trans(a);
  if (x.n_elem != a.n_rows && a.n_rows != x.n_elem+2)
    Rcpp::stop("The number of rows of y must be equal to the length of x.\n");

  bool complete = false;
  arma::rowvec dfs, dfe;
  if (a.n_rows == n+2) {
    complete = true;
    dfs = a.row(0);
    dfe = a.row(a.n_rows-1);
    a = a.rows(1, a.n_rows-2);
  }

  if (!x.is_sorted())
    RMessage("x are not strictly monotonic increasing.\nx will be sorted.");
  arma::uvec xu_idx = arma::find_unique(x);
  arma::vec xu = x(xu_idx);
  arma::mat au = a.rows(xu_idx);
  if (xu.n_elem != x.n_elem) {
    RMessage("The grid vectors are not strictly monotonic increasing.");
    RMessage("The values of y for duplicated values of x will be averaged.");
    for (arma::uword k = 0; k < xu.n_elem; ++k)
      au.row(k) = arma::mean(a.rows(arma::find(x == xu(k))));
    n = xu_idx.n_elem;
  }
  if (!xu.is_sorted()) {
    arma::uvec si = arma::sort_index(xu);
    xu = xu(si);
    au = au.rows(si);
  }

  arma::mat ca, cb, cc, cd;
  arma::vec xou = xu, h = arma::diff(xu);
  if (complete) {
    if (n == 2) {
      cd = (dfs + dfe) / std::pow(xu(1) - xu(0), 2.0) +
        2.0 * (au.row(0) - au.row(1)) / std::pow(xu(1) - xu(0), 3.0);
      cc = (-2.0 * dfs - dfe) / (xu(1) - xu(0)) -
        3.0 * (au.row(0) - au.row(1)) / std::pow(xu(1) - xu(0), 2.0);
      cb = dfs;
      ca = au.row(0);
    } else {
      arma::mat g = arma::zeros<arma::mat>(n, au.n_cols);
      g.row(0) = (au.row(1) - au.row(0)) / h(0) - dfs;
      g.rows(1, n-2) = (au.rows(2, n-1) - au.rows(1, n-2)) / repmat(h.subvec(1, n-2), 1, au.n_cols) -
        (au.rows(1, n-2) - a.rows(0, n-3)) / repmat(h.subvec(0, n-3), 1, au.n_cols);
      g.row(n-1) = dfe-(au.row(n-1) - au.row(n-2)) / h(n-2);

      ca = au;
      cc = arma::solve(arma::diagmat(h/6.0, -1) +
        arma::diagmat(arma::join_cols(arma::join_cols(h(0)/3 * arma::ones<arma::vec>(1), (h.head(n-2) + h.tail(n-2))/3),
                                      h(n-2)/3.0*arma::ones<arma::vec>(1))) + arma::diagmat(h/6.0, 1), 0.5 * g);
      cb = arma::diff(au) / arma::repmat(h.head(n-1), 1, au.n_cols) -
        arma::repmat(h.head(n-1), 1, au.n_cols) / 3.0 % (cc.rows(1, n-1) + 2 * cc.rows(0, n-2));
      cd = arma::diff(cc) / (3.0 * arma::repmat(h.head(n-1), 1, au.n_cols));
      ca = ca.head_rows(n-1);
      cb = cb.head_rows(n-1);
      cc = cc.head_rows(n-1);
      cd = cd.head_rows(n-1);
    }
  } else {
    if (n == 2) {
      cd.zeros(1, au.n_cols);
      cc.zeros(1, au.n_cols);
      cb = (au.row(1) - au.row(0)) / (xu(1) - xu(0));
      ca = au.row(0);
    } else if (n == 3) {
      n = 2;
      cd.zeros(1, au.n_cols);
      cc = (au.row(0) - au.row(2)) / ((xu(2) - xu(0)) * (xu(1) - xu(2))) +
        (au.row(1) - au.row(0)) / ((xu(1) - xu(0)) * (xu(1) - xu(2)));
      cb = (au.row(1) - au.row(0)) * (xu(2) - xu(0)) /  ((xu(1) - xu(0)) * (xu(2) - xu(1))) +
        (au.row(0) - au.row(2)) * (xu(1) - xu(0)) /  ((xu(2) - xu(0)) * (xu(2) - xu(1)));
      ca = au.row(0);
      xou << arma::min(x) << arma::max(x) << arma::endr;
    } else {
      arma::mat g = arma::zeros<arma::mat>(n-2, au.n_cols);
      g.row(0) = 3.0 / (h(0) + h(1)) *
        (au.row(2) - au.row(1) - h(1) / h(0) * (au.row(1) - au.row(0)));
      g.row(n-3) = 3.0 / (h(n-2) + h(n-3)) *
        (h(n-3) / h(n-2) * (au.row(n-1) - au.row(n-2)) - (au.row(n-2) - au.row(n-3)));

      if (n > 4) {
        cc.zeros(n, au.n_cols);
        g.rows(1, n-4) = 3.0 * arma::diff(au.rows(2, n-2)) / arma::repmat(h.subvec(2, n-3), 1, au.n_cols) -
          3.0 * diff(au.rows(1, n-3)) / arma::repmat(h.subvec(1, n-4), 1, au.n_cols);

        arma::vec dg = 2.0 * (h.head(n-2) + h.tail(n-2)),
          ldg = h.subvec(1, n-3), udg = h.subvec(1, n-3);
        dg(0) = dg(0) - h(0);
        dg(n-3) = dg(n-3) - h(n-2);
        udg(0) = udg(0) - h(0);
        ldg(n-4) = ldg(n-4) - h(n-2);
        cc.rows(1, n-2) = solve(diagmat(ldg, -1) + diagmat(dg) + diagmat(udg, 1), g);
      } else {
        cc.zeros(n, au.n_cols);
        arma::mat tmp(2, 2);
        tmp << h(0) + 2.0 * h(1) << h(1) - h(0) << arma::endr
            << h(1) - h(2) << 2.0 * h(1) + h(2) << arma::endr;
        cc.rows(1, 2) = solve(tmp, g);
      }

      ca = au;
      cc.row(0) = cc.row(1) + h(0) / h(1) * (cc.row(1) - cc.row(2));
      cc.row(n-1) = cc.row(n-2) + h(n-2) / h(n-3) * (cc.row(n-2) - cc.row(n-3));
      cb = arma::diff(ca);
      cb.each_col() /= h.head(n-1);
      cb -= arma::repmat(h.head(n-1), 1, au.n_cols) / 3.0 % (cc.rows(1, n-1) + 2.0 * cc.rows(0, n-2));
      cd = arma::diff(cc) / 3.0;
      cd.each_col() /= h.head(n-1);
      ca = ca.head_rows(n-1);
      cb = cb.head_rows(n-1);
      cc = cc.head_rows(n-1);
      cd = cd.head_rows(n-1);
    }
  }

  arma::uvec idx = arma::zeros<arma::uvec>(xi.n_elem);
  for (arma::uword i = 1; i < xou.n_elem-1; ++i)
    idx.elem(find(xou(i) <= xi)).fill(i);
  arma::mat s_mat = repmat(xi - xou.elem(idx), 1, au.n_cols);
  arma::mat ret = ca.rows(idx) + s_mat % cb.rows(idx) + square(s_mat) % cc.rows(idx) +
    pow(s_mat, 3) % cd.rows(idx);
  return ret;
}

arma::mat interp1Func(const arma::vec& x, const arma::mat& y, const arma::vec& xi, const std::string& method) {
  arma::uword n = x.n_elem;
  arma::mat a = y;
  arma::uvec szy(2);
  szy << y.n_rows << y.n_cols << arma::endr;
  if (szy(1) == n && szy(0) != n+2 && szy(0) != n && szy(0) != 1)
    inplace_trans(a);
  if (x.n_elem != a.n_rows)
    Rcpp::stop("The number of rows of y must be equal to the length of x.\n");

  if (!x.is_sorted())
    RMessage("x are not strictly monotonic increasing.\nx will be sorted.");
  arma::uvec xu_idx = find_unique(x);
  arma::vec xu = x(xu_idx);
  arma::mat au = a.rows(xu_idx);
  if (xu.n_elem != x.n_elem) {
    RMessage("The grid vectors are not strictly monotonic increasing.");
    RMessage("The values of y for duplicated values of x will be averaged.");
    for (arma::uword k = 0; k < xu.n_elem; ++k)
      au.row(k) = arma::mean(a.rows(arma::find(x == xu(k))));
  }
  if (!xu.is_sorted()) {
    arma::uvec si = arma::sort_index(xu);
    xu = xu(si);
    au = au.rows(si);
  }

  arma::mat yi;
  if (method == "linear") {
    if (x.n_elem <= 1)
      Rcpp::stop("interp1 - linear: requires at least two non-NA values.\n");
    arma::mat cb = arma::diff(au) / arma::repmat(diff(xu), 1, au.n_cols);
    arma::mat ca = au.rows(0, xu.n_elem-2);
    arma::uvec idx = arma::zeros<arma::uvec>(xi.n_elem);
    for (arma::uword i = 1; i < xu.n_elem-1; ++i)
      idx.elem(arma::find(xu(i) <= xi)).fill(i);
    arma::mat s_mat = arma::repmat(xi - xu.elem(idx), 1, au.n_cols);
    yi = ca.rows(idx) + s_mat % cb.rows(idx);
  } else if (method ==  "spline") {
    yi = splineFunc(xu, au, xi);
  } else {
    Rcpp::stop("Method only support linear and spline.\n");
  }
  return yi;
}

arma::mat interp2Func(const arma::vec& x, const arma::vec& y, const arma::mat& v,
                      const arma::vec& xi, const arma::vec& yi, const std::string& method) {
  if (x.n_elem != v.n_cols)
    Rcpp::stop("The number of columns of v must be equal to the length of x.");
  if (y.n_elem != v.n_rows)
    Rcpp::stop("The number of rows of v must be equal to the length of y.");
  if (!x.is_sorted())
    RMessage("x are not strictly monotonic increasing.\nx will be sorted.");
  if (method != "linear" && method != "spline")
    Rcpp::stop("Method only support linear and spline.\n");

  arma::uvec xu_idx = find_unique(x);
  arma::vec xu = x(xu_idx);
  arma::mat v_tmp = v.cols(xu_idx);
  if (!xu.is_sorted()) {
    arma::uvec si = sort_index(xu);
    xu = xu(si);
    v_tmp = v_tmp.cols(si);
  }
  if (xu.n_elem != x.n_elem) {
    RMessage("The grid vectors are not strictly monotonic increasing.");
    RMessage("The values of v for duplicated values of x will be averaged.");
    for (arma::uword k = 0; k < xu.n_elem; ++k)
      v_tmp.col(k) = mean(v.cols(arma::find(x == xu(k))), 1);
  }

  if (!y.is_sorted())
    RMessage("y are not strictly monotonic increasing.\ny will be sorted.");
  arma::uvec yu_idx = find_unique(y);
  arma::vec yu = y(yu_idx);
  arma::mat vu = v_tmp.rows(yu_idx);
  if (!yu.is_sorted()) {
    arma::uvec si = arma::sort_index(yu);
    yu = yu(si);
    vu = vu.rows(si);
  }
  if (yu.n_elem != y.n_elem) {
    RMessage("The grid vectors are not strictly monotonic increasing.");
    RMessage("The values of v for duplicated values of y will be averaged.");
    for (arma::uword k = 0; k < yu.n_elem; ++k)
      vu.row(k) = arma::mean(v_tmp.rows(arma::find(y == yu(k))));
  }

  arma::mat vi(xi.n_elem, yi.n_elem);
  if (method == "linear") {
    arma::uvec xidx_tmp = lookup(xu, xi), yidx_tmp = lookup(yu, yi);
    xidx_tmp(arma::find(xidx_tmp == xu.n_elem)).fill(xu.n_elem - 1);
    yidx_tmp(arma::find(yidx_tmp == yu.n_elem)).fill(yu.n_elem - 1);
    xidx_tmp--;
    yidx_tmp--;

    arma::uvec xidx = arma::vectorise(arma::repmat(xidx_tmp.t(), yi.n_elem, 1)),
      yidx = arma::vectorise(arma::repmat(yidx_tmp, 1, xi.n_elem));

    arma::uword nvr = vu.n_rows, nvc = vu.n_cols;
    arma::mat a = vu.submat(0, 0, nvr-2, nvc-2),
      b = vu.submat(0, 1, nvr-2, nvc-1) - a,
      c = vu.submat(1, 0, nvr-1, nvc-2) - a,
      d = vu.submat(1, 1, nvr-1, nvc-1) - a - b - c;

    arma::vec dx = arma::diff(xu), dy = arma::diff(yu);
    arma::vec xsc = (vectorise(repmat(xi.t(), yi.n_elem, 1)) - xu.elem(xidx)) / dx.elem(xidx),
      ysc = (arma::repmat(yi, xi.n_elem, 1) - yu.elem(yidx)) / dy.elem(yidx);
    arma::uvec idx = yidx + a.n_rows* xidx;
    vi = reshape(a(idx) + b(idx) % xsc + c(idx) % ysc + d(idx) % xsc % ysc, yi.n_elem, xi.n_elem);
  } else if (method == "spline") {
    vi = splineFunc(yu, vu, yi);
    vi = splineFunc(xu, vi.t(), xi).t();
  }
  return vi;
}
