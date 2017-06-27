/*!
 *  Copyright (c) 2014 by Contributors
 *  \file cp_decomp_lapack.h
 *  \brief Linear algebra routines used by CPDecomp
 *  \author Jencir Lee
 */
#ifndef MXNET_OPERATOR_CONTRIB_TENSOR_CP_DECOMP_LAPACK_H_
#define MXNET_OPERATOR_CONTRIB_TENSOR_CP_DECOMP_LAPACK_H_

#ifdef __cplusplus
extern "C" {
#endif

void dposv_(char *uplo, int *n, int *nrhs,
    double *a, int *lda, double *b, int *ldb, int *info);

void sposv_(char *uplo, int *n, int *nrhs,
    float *a, int *lda, float *b, int *ldb, int *info);

void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau,
    double *work, int *lwork, int *info);

void sgeqrf_(int *m, int *n, float *a, int *lda, float *tau,
    float *work, int *lwork, int *info);

void dorgqr_(int *m, int *n, int *k, double *a, int *lda, double *tau,
    double *work, int *lwork, int *info);

void sorgqr_(int *m, int *n, int *k, float *a, int *lda, float *tau,
    float *work, int *lwork, int *info);

void dgesdd_(char *jobz, int *m, int *n, double *a, int *lda, double *s,
    double *u, int *ldu, double *vt, int *ldvt, double *work,
    int *lwork, int *iwork, int *info);

void sgesdd_(char *jobz, int *m, int *n, float *a, int *lda, float *s,
    float *u, int *ldu, float *vt, int *ldvt, float *work,
    int *lwork, int *iwork, int *info);

#ifdef __cplusplus
}
#endif

#endif  // MXNET_OPERATOR_CONTRIB_TENSOR_CP_DECOMP_LAPACK_H_
