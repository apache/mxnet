/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * Copyright (c) 2018 by Contributors
 * \file binary_inference_convolution-inl.h
 * \brief
 * \ref: https://arxiv.org/abs/1705.09864
 * \author HPI-DeepLearning Haojin Yang, Martin Fritzsche
*/

#include "xnor.h"
#include <chrono>
#include <cstring>
#include <iomanip>

namespace mxnet {
namespace op {
namespace xnor {

#define UNROLLN 6

void xnor_gemm_unrolled(int M, int N, int K,
                        BINARY_WORD *A, int lda,
                        BINARY_WORD *B, int ldb,
                        float *C, int ldc){
  int m,k,n;
  #pragma omp parallel for    
  for (m = 0; m < M; ++m) {
    #pragma omp parallel for
    for (k = 0; k < ((K / UNROLLN) * UNROLLN); k+=UNROLLN) {
      BINARY_WORD A_PART[UNROLLN];
      A_PART[0] = A[m*lda+k];
      A_PART[1] = A[m*lda+k+1];
      A_PART[2] = A[m*lda+k+2];
      A_PART[3] = A[m*lda+k+3];
      A_PART[4] = A[m*lda+k+4];
      A_PART[5] = A[m*lda+k+5];
      #pragma omp parallel for
      for (n = 0; n < N; ++n) {
        int popc[UNROLLN];
        popc[0] = __builtin_popcountl(~(A_PART[0] ^ B[(k+0)*ldb+n]));
        popc[1] = __builtin_popcountl(~(A_PART[1] ^ B[(k+1)*ldb+n]));
        popc[2] = __builtin_popcountl(~(A_PART[2] ^ B[(k+2)*ldb+n]));
        popc[3] = __builtin_popcountl(~(A_PART[3] ^ B[(k+3)*ldb+n]));
        popc[4] = __builtin_popcountl(~(A_PART[4] ^ B[(k+4)*ldb+n]));
        popc[5] = __builtin_popcountl(~(A_PART[5] ^ B[(k+5)*ldb+n]));
        C[m*ldc+n] += popc[0] + popc[1] + popc[2] + popc[3] + popc[4] + popc[5];
      }
    }

    #pragma omp parallel for 
    for (k=(K / UNROLLN) * UNROLLN; k < K; ++k) {
      BINARY_WORD A_PART = A[m*lda+k];
      #pragma omp parallel for
      for (n = 0; n < N; ++n) {
        C[m * ldc + n] += __builtin_popcountl(~(A_PART ^ B[k * ldb + n]));
      }
    }
  }
}

void xnor_gemm_unrolled_no_omp(int M, int N, int K,
                        BINARY_WORD *A, int lda,
                        BINARY_WORD *B, int ldb,
                        float *C, int ldc){
  int m,k,n;
  BINARY_WORD A_PART[UNROLLN];
  int popc[UNROLLN];
  for (m = 0; m < M; ++m) {
    for (k = 0; k < ((K / UNROLLN) * UNROLLN); k+=UNROLLN) {
      A_PART[0] = A[m*lda+k];
      A_PART[1] = A[m*lda+k+1];
      A_PART[2] = A[m*lda+k+2];
      A_PART[3] = A[m*lda+k+3];
      A_PART[4] = A[m*lda+k+4];
      A_PART[5] = A[m*lda+k+5];
      for (n = 0; n < N; ++n) {
        popc[0] = __builtin_popcountl(~(A_PART[0] ^ B[(k+0)*ldb+n]));
        popc[1] = __builtin_popcountl(~(A_PART[1] ^ B[(k+1)*ldb+n]));
        popc[2] = __builtin_popcountl(~(A_PART[2] ^ B[(k+2)*ldb+n]));
        popc[3] = __builtin_popcountl(~(A_PART[3] ^ B[(k+3)*ldb+n]));
        popc[4] = __builtin_popcountl(~(A_PART[4] ^ B[(k+4)*ldb+n]));
        popc[5] = __builtin_popcountl(~(A_PART[5] ^ B[(k+5)*ldb+n]));
        C[m*ldc+n] += popc[0] + popc[1] + popc[2] + popc[3] + popc[4] + popc[5];
      }
    }

    for (k=(K / UNROLLN) * UNROLLN; k < K; ++k) {
      A_PART[0] = A[m*lda+k];
      for (n = 0; n < N; ++n) {
        C[m * ldc + n] += __builtin_popcountl(~(A_PART[0] ^ B[k * ldb + n]));
      }
    }
  }
}

// write popc in int array, in the end convert back
void xnor_gemm_convert_to_int(int M, int N, int K,
                      BINARY_WORD *A, int lda,
                      BINARY_WORD *B, int ldb,
                      float *C, int ldc){
  int m,k,n;
  int popc[M*N];
  #pragma omp parallel for collapse(2)    
  for (m = 0; m < M; ++m) {
    for (k = 0; k < K; k++) {
      BINARY_WORD A_PART = A[m*lda+k];
      #pragma omp parallel for
      for (n = 0; n < N; ++n) {
        popc[m*ldc+n] += __builtin_popcountll(~(A_PART ^ B[k*ldb+n]));
      }
    }
  }

  for (int i=0; i < M*N; i++) {
    C[i] = popc[i];
  }
}

// write popc in int array, in the end convert back
void xnor_gemm_convert_to_int_no_omp(int M, int N, int K,
                      BINARY_WORD *A, int lda,
                      BINARY_WORD *B, int ldb,
                      float *C, int ldc){
  int m,k,n;
  int popc[M*N];

  for (m = 0; m < M; ++m) {
    for (k = 0; k < K; k++) {
      BINARY_WORD A_PART = A[m*lda+k];
      for (n = 0; n < N; ++n) {
        popc[m*ldc+n] += __builtin_popcountll(~(A_PART ^ B[k*ldb+n]));
      }
    }
  }

  for (int i=0; i < M*N; i++) {
    C[i] = popc[i];
  }
}

// our baseline xnor
void xnor_gemm_baseline(int M, int N, int K,
                      BINARY_WORD *A, int lda,
                      BINARY_WORD *B, int ldb,
                      float *C, int ldc){
  int m,k,n;
  #pragma omp parallel for collapse(2)    
  for (m = 0; m < M; ++m) {
    for (k = 0; k < K; k++) {
      BINARY_WORD A_PART = A[m*lda+k];
      #pragma omp parallel for
      for (n = 0; n < N; ++n) {
        C[m*ldc+n] += __builtin_popcountll(~(A_PART ^ B[k*ldb+n]));
      }
    }
  }
}

// our baseline sans omp
void xnor_gemm_baseline_no_omp(int M, int N, int K,
                      BINARY_WORD *A, int lda,
                      BINARY_WORD *B, int ldb,
                      float *C, int ldc){
  int m,k,n;
  for (m = 0; m < M; ++m) {
    for (k = 0; k < K; k++) {
      BINARY_WORD A_PART = A[m*lda+k];
      for (n = 0; n < N; ++n) {
        C[m*ldc+n] += __builtin_popcountll(~(A_PART ^ B[k*ldb+n]));
      }
    }
  }
}

//========================================================================//
//                       Optimized XNOR GEMM                              //
//========================================================================//
/* Create macros so that the matrices are stored in column-major order */
#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Block sizes which are based on L2-cache of cpu*/
#define mc 128
#define kc 128
#define nc 256

void pack_matrixB( int k, BINARY_WORD *b, int ldb, BINARY_WORD *b_to )
{
  int j;    
  for( j=0; j<k; j++){  /* loop over rows of B */
    BINARY_WORD 
    *b_ij_pntr = &B( 0, j );

    *b_to = *b_ij_pntr;
    *(b_to+1) = *(b_ij_pntr+1);
    *(b_to+2) = *(b_ij_pntr+2);
    *(b_to+3) = *(b_ij_pntr+3);
    b_to += 4;
  }
}

void add_dot_4x4( int k, BINARY_WORD *a, int lda,  BINARY_WORD *b, int ldb, float *c, int ldc )
{
  int p;
  register BINARY_WORD 
    /* hold contributions to
       C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ) 
       C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ) 
       C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ) 
       C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 )   */
       c_00_reg,   c_01_reg,   c_02_reg,   c_03_reg,  
       c_10_reg,   c_11_reg,   c_12_reg,   c_13_reg,  
       c_20_reg,   c_21_reg,   c_22_reg,   c_23_reg,  
       c_30_reg,   c_31_reg,   c_32_reg,   c_33_reg;     
  register BINARY_WORD 
        b_0p_reg,
        b_1p_reg,
        b_2p_reg,
        b_3p_reg,
        a_p0_reg,
        a_p1_reg,
        a_p2_reg,
        a_p3_reg;
  BINARY_WORD 
    /* Point to the current elements in the four columns of A */
    *a_p0_pntr, *a_p1_pntr, *a_p2_pntr, *a_p3_pntr; 
    
  a_p0_pntr = &A( 0, 0 );
  a_p1_pntr = &A( 0, 1 );
  a_p2_pntr = &A( 0, 2 );
  a_p3_pntr = &A( 0, 3 );

  c_00_reg = 0;   c_01_reg = 0;   c_02_reg = 0;   c_03_reg = 0;
  c_10_reg = 0;   c_11_reg = 0;   c_12_reg = 0;   c_13_reg = 0;
  c_20_reg = 0;   c_21_reg = 0;   c_22_reg = 0;   c_23_reg = 0;
  c_30_reg = 0;   c_31_reg = 0;   c_32_reg = 0;   c_33_reg = 0;

  for ( p=0; p<k; p++ ){
    b_0p_reg = B( 0, p );
    b_1p_reg = B( 1, p );
    b_2p_reg = B( 2, p );
    b_3p_reg = B( 3, p );

    a_p0_reg = *a_p0_pntr++;
    a_p1_reg = *a_p1_pntr++;
    a_p2_reg = *a_p2_pntr++;
    a_p3_reg = *a_p3_pntr++;
 
    /* First row and second rows */
    c_00_reg += __builtin_popcountll(~(b_0p_reg ^ a_p0_reg));
    c_10_reg += __builtin_popcountll(~(b_1p_reg ^ a_p0_reg));

    c_01_reg += __builtin_popcountll(~(b_0p_reg ^ a_p1_reg));
    c_11_reg += __builtin_popcountll(~(b_1p_reg ^ a_p1_reg));

    c_02_reg += __builtin_popcountll(~(b_0p_reg ^ a_p2_reg));
    c_12_reg += __builtin_popcountll(~(b_1p_reg ^ a_p2_reg));

    c_03_reg += __builtin_popcountll(~(b_0p_reg ^ a_p3_reg));
    c_13_reg += __builtin_popcountll(~(b_1p_reg ^ a_p3_reg));

    /* Third and fourth rows */
    c_20_reg += __builtin_popcountll(~(b_2p_reg ^ a_p0_reg));
    c_30_reg += __builtin_popcountll(~(b_3p_reg ^ a_p0_reg));

    c_21_reg += __builtin_popcountll(~(b_2p_reg ^ a_p1_reg));
    c_31_reg += __builtin_popcountll(~(b_3p_reg ^ a_p1_reg));

    c_22_reg += __builtin_popcountll(~(b_2p_reg ^ a_p2_reg));
    c_32_reg += __builtin_popcountll(~(b_3p_reg ^ a_p2_reg));

    c_23_reg += __builtin_popcountll(~(b_2p_reg ^ a_p3_reg));
    c_33_reg += __builtin_popcountll(~(b_3p_reg ^ a_p3_reg));
  }

  C( 0, 0 ) += c_00_reg;   C( 0, 1 ) += c_01_reg;   C( 0, 2 ) += c_02_reg;   C( 0, 3 ) += c_03_reg;
  C( 1, 0 ) += c_10_reg;   C( 1, 1 ) += c_11_reg;   C( 1, 2 ) += c_12_reg;   C( 1, 3 ) += c_13_reg;
  C( 2, 0 ) += c_20_reg;   C( 2, 1 ) += c_21_reg;   C( 2, 2 ) += c_22_reg;   C( 2, 3 ) += c_23_reg;
  C( 3, 0 ) += c_30_reg;   C( 3, 1 ) += c_31_reg;   C( 3, 2 ) += c_32_reg;   C( 3, 3 ) += c_33_reg;
}

void xnor_gemm_blocking_packing_inner_kernel( int m, int n, int k, BINARY_WORD *a, int lda, 
                                       BINARY_WORD *b, int ldb,
                                       float *c, int ldc, int first_time )
{
  int i, j;
  BINARY_WORD* packedB = new BINARY_WORD[ n * k ];
  #pragma omp parallel for 
  for ( j=0; j<n; j+=4 ){          /* Loop over the columns of C, unrolled by 4 */      
    if(first_time)
      pack_matrixB( k, &B( j, 0 ), ldb, &packedB[ j*k ] ); 
    #pragma omp parallel for
    for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
      one routine (four inner products) */
      //add_dot_4x4( k, &A( 0,i ), lda, &B( j,0 ), ldb, &C( j,i ), ldc );
      add_dot_4x4( k, &A( 0,i ), lda, &packedB[ j*k ], 4, &C( j,i ), ldc );
    }
    
  } 

  delete[] packedB;
}

void xnor_gemm_blocking_packing_inner_kernel_no_omp( int m, int n, int k, BINARY_WORD *a, int lda, 
                                       BINARY_WORD *b, int ldb,
                                       float *c, int ldc, int first_time )
{
  int i, j;
  BINARY_WORD*  packedB = new BINARY_WORD[ n * k ];
  
  for ( j=0; j<n; j+=4 ){          /* Loop over the columns of C, unrolled by 4 */  
      if(first_time)
        pack_matrixB( k, &B( j, 0 ), ldb, &packedB[ j*k ] ); 
    for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
      one routine (four inner products) */
      //add_dot_4x4( k, &A( 0,i ), lda, &B( j,0 ), ldb, &C( j,i ), ldc );
      add_dot_4x4( k, &A( 0,i ), lda, &packedB[ j*k ], 4, &C( j,i ), ldc );
    }
  }

  delete[] packedB;
}

/**
* applys blocking, packing, loop unrolling, register vars to improve the
* xnor_gemm performance. ~100% performance improvement without openmp
* compared with xnor_gemm() method.
*/
void xnor_gemm_blocking_packing( int m, int n, int k, BINARY_WORD *a, int lda, 
                                    BINARY_WORD *b, int ldb,
                                    float *c, int ldc )
{
  int i, p, pb, ib;

  /* This time, we compute a mc x n block of C by a call to the InnerKernel */
  for ( p=0; p<k; p+=kc ){
    pb = std::min( kc, k-p );    
    for ( i=0; i<m; i+=mc ){
      ib = std::min( mc, m-i );
      xnor_gemm_blocking_packing_inner_kernel( ib, n, pb, &A(p, i), lda, &B(0, p), ldb, &C( 0, i ), ldc, i==0 );
    }
  }
}

void xnor_gemm_blocking_packing_no_omp( int m, int n, int k, BINARY_WORD *a, int lda, 
                                    BINARY_WORD *b, int ldb,
                                    float *c, int ldc )
{
  int i, p, pb, ib;

  /* This time, we compute a mc x n block of C by a call to the InnerKernel */
  for ( p=0; p<k; p+=kc ){
    pb = std::min( kc, k-p );    
    for ( i=0; i<m; i+=mc ){
      ib = std::min( mc, m-i );
      xnor_gemm_blocking_packing_inner_kernel_no_omp( ib, n, pb, &A(p, i), lda, &B(0, p), ldb, &C( 0, i ), ldc, i==0 );
    }
  }
}
//========================= END optimized xnor GEMM ===============================//

void xnor_gemm_combined(int M, int N, int K,
               BINARY_WORD *A, int lda,
               BINARY_WORD *B, int ldb,
               float *C, int ldc){
  if(K <= 4 || M < 4)
    xnor_gemm_baseline(M, N, K, A, lda, B, ldb, C, ldc);
  else if (K < 10 && N <= 64)
    xnor_gemm_unrolled_no_omp(M, N, K, A, lda, B, ldb, C, ldc);
  else
    xnor_gemm_unrolled(M, N, K, A, lda, B, ldb, C, ldc);

  //TODO: xnor_gemm_blocking_packing need to be checked!!
  // if(K <= 4 || M < 4)
  //   xnor_gemm_baseline(M, N, K, A, lda, B, ldb, C, ldc);
  // else if (K <= 4 && N <= 64)
  //   xnor_gemm_unrolled_no_omp(M, N, K, A, lda, B, ldb, C, ldc);
  // else if(M <= 64 && K <= 100)
  //   xnor_gemm_unrolled(M, N, K, A, lda, B, ldb, C, ldc);
  // else
  //   xnor_gemm_blocking_packing(M, N, K, A, lda, B, ldb, C, ldc);
}

void xnor_gemm_benchmarking(int M, int N, int K,
               BINARY_WORD *A, int lda,
               BINARY_WORD *B, int ldb,
               float *C, int ldc)
{
  std::vector<std::pair<std::string, std::function<void(int,int,int,BINARY_WORD*,int,BINARY_WORD*,int,float*,int)>>> gemm_methods;  
  gemm_methods.push_back(std::make_pair("baseline", xnor_gemm_baseline_no_omp));
  gemm_methods.push_back(std::make_pair("convert_to_int", xnor_gemm_convert_to_int_no_omp));
  gemm_methods.push_back(std::make_pair("unrolled", xnor_gemm_unrolled_no_omp));
  gemm_methods.push_back(std::make_pair("blocking_packing", xnor_gemm_blocking_packing_no_omp));
  gemm_methods.push_back(std::make_pair("baseline", xnor_gemm_baseline));
  gemm_methods.push_back(std::make_pair("convert_to_int", xnor_gemm_convert_to_int));
  gemm_methods.push_back(std::make_pair("unrolled", xnor_gemm_unrolled));
  gemm_methods.push_back(std::make_pair("blocking_packing", xnor_gemm_blocking_packing));
  gemm_methods.push_back(std::make_pair("combined", xnor_gemm_combined));  

  std::ostringstream line;
  line.setf(std::ios::fixed,std::ios::floatfield);
  line.precision(4);

  for (auto gemm : gemm_methods) {
    line << "-----------";
  }
  line << std::endl;
  line << "xnor_gemm:(M: " << M << " N: " << N << " K: " << K << ")";
  line << std::endl;

  int i = 0;
  for (auto gemm : gemm_methods) {
    line << gemm.first;
    if(i==3){
      line << "\t (Methods NO omp)" << std::endl;
      break;
    }else
      line << std::setw(20);
    i++;
  }


  i = 0;
  for (auto gemm : gemm_methods) {
    if(i<4){
      //reset output array
      std::memset(C, 0, M * N * sizeof(float));

      auto start = std::chrono::high_resolution_clock::now();

      gemm.second(M, N, K, A, lda, B, ldb, C, ldc);

      auto finish = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = finish - start;
      line << round(elapsed.count() * 10000.0) / 10000.0;
      if(i != 3) line << std::setw(20);
    }else{
      line << std::endl;
      break;
    }
    i++;
  }
  
  i = 0;
  for (auto gemm : gemm_methods) {
    if(i >3){
      line << gemm.first << std::setw(20);
      if(i==8){
        line << "\t (Methods USE omp)";
        line << std::endl;
      }
    }
    i++;
  }

  i = 0;
  for (auto gemm : gemm_methods) {
    if(i>3){
      //reset output array
      std::memset(C, 0, M * N * sizeof(float));

      auto start = std::chrono::high_resolution_clock::now();

      gemm.second(M, N, K, A, lda, B, ldb, C, ldc);

      auto finish = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = finish - start;
      line << round(elapsed.count() * 10000.0) / 10000.0;
      if(i != 8) line << std::setw(20);
    }
    i++;
  }
  line << std::endl;

  for (auto gemm : gemm_methods) {
    line << "-----------";
  }
  std::cout << line.str() << std::endl;  
}


 /*
  * XNOR GEMM kernel
  */
  void xnor_gemm(int M, int N, int K,
                 BINARY_WORD *A, int lda,
                 BINARY_WORD *B, int ldb,
                 float *C, int ldc){    
    //benchmarking several xnor_gemm methods
    //xnor_gemm_benchmarking(M, N, K, A, lda, B, ldb, C, ldc);
    xnor_gemm_combined(M, N, K, A, lda, B, ldb, C, ldc);
  }

} //namespace xnor_cpu
} //namespace op
} //namespace mxnet
