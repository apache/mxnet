#ifndef CBLAS_H
#define CBLAS_H

/*
 * This file modified from the OpenBLAS repository, 
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
	/* Assume C declarations for C++ */
#endif  /* __cplusplus */


/*
 * Since all of GotoBlas was written without const,
 * we disable it at build time.
 */
#ifndef OPENBLAS_CONST
# define OPENBLAS_CONST const
#endif

/*
 * Add definitions for BLASLONG and blasint
 */

#if defined(OS_WINDOWS) && defined(__64BIT__)
typedef long long BLASLONG;
typedef unsigned long long BLASULONG;
#else
typedef long BLASLONG;
typedef unsigned long BLASULONG;
#endif

#ifdef INTERFACE64
typedef BLASLONG blasint;
#else
typedef int blasint;
#endif

#ifdef INTERFACE64
#define cblas_sgemm cblas_sgemm64_
#define cblas_sgemv cblas_sgemv64_
#define cblas_sger  cblas_sger64_
#define cblas_dgemm cblas_dgemm64_
#define cblas_dgemv cblas_dgemv64_
#define cblas_dger  cblas_dger64_
#define cblas_sdot  cblas_sdot64_
#define cblas_ddot  cblas_ddot64_
#define cblas_strmm  cblas_strmm64_
#define cblas_dtrmm  cblas_dtrmm64_
#define cblas_strsm  cblas_strsm64_
#define cblas_dtrsm  cblas_dtrsm64_
#endif


typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum CBLAS_DIAG      {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum CBLAS_SIDE      {CblasLeft=141, CblasRight=142} CBLAS_SIDE;

float  cblas_sdot(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST float  *y, OPENBLAS_CONST blasint incy);
double cblas_ddot(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST double *y, OPENBLAS_CONST blasint incy);

void cblas_sgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n,
		 OPENBLAS_CONST float alpha, OPENBLAS_CONST float  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST float beta,  float  *y, OPENBLAS_CONST blasint incy);
void cblas_dgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n,
		 OPENBLAS_CONST double alpha, OPENBLAS_CONST double  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST double  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST double beta,  double  *y, OPENBLAS_CONST blasint incy);
void cblas_sger (OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST float   alpha, OPENBLAS_CONST float  *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST float  *Y, OPENBLAS_CONST blasint incY, float  *A, OPENBLAS_CONST blasint lda);
void cblas_dger (OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST double  alpha, OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST double *Y, OPENBLAS_CONST blasint incY, double *A, OPENBLAS_CONST blasint lda);

void cblas_sgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
void cblas_dgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST double beta, double *C, OPENBLAS_CONST blasint ldc);

void cblas_strmm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, float *B, OPENBLAS_CONST blasint ldb);
void cblas_dtrmm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, double *B, OPENBLAS_CONST blasint ldb);
void cblas_strsm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, float *B, OPENBLAS_CONST blasint ldb);
void cblas_dtrsm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, double *B, OPENBLAS_CONST blasint ldb);

#ifdef __cplusplus
}
#endif  /* __cplusplus */

#endif
