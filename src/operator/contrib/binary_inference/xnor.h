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
 * \author HPI-DeepLearning
*/

#ifndef MXNET_OPERATOR_CONTRIB_BINARY_INFERENCE_XNOR_H
#define MXNET_OPERATOR_CONTRIB_BINARY_INFERENCE_XNOR_H

#include <dmlc/logging.h>
#include <mshadow/base.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>
#include <limits.h>
#include <tgmath.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>


namespace mxnet {
namespace op {
namespace xnor {

  // variable, position, value
  #define BIT_SET(var, pos, val) var |= (val << pos)
  
  //uint32_t, uint64_t
  #if BINARY_WORD_32 == 1
    typedef uint32_t BINARY_WORD;
  #endif
  #if BINARY_WORD_64 == 1
    typedef uint64_t BINARY_WORD;
  #endif

  const int BITS_PER_BINARY_WORD (sizeof(BINARY_WORD) * CHAR_BIT);

  /**
  * @brief returns a mshadow dtype with corresponding bitwidth to BINARY_WORD
  *
  */
  inline mshadow::TypeFlag corresponding_dtype() {
    if (BITS_PER_BINARY_WORD == 32) {
      return mshadow::kFloat32;
    } else if (BITS_PER_BINARY_WORD == 64) {
      return mshadow::kFloat64;
    }
    assert(false);
    return mshadow::kFloat32;
  }

  /**
  * @brief a helper method for print out bit wise result
  * of a binary_word
  *
  */
  inline void print_int2Bin ( BINARY_WORD a )
  {
     
    for (int i=0; i <BITS_PER_BINARY_WORD; i++ )
    {
      if( a & (1 << i) ) 
        std::cout << 1;
      else
        std::cout << 0;
    }
    std::cout<<std::endl;
  }

  inline void print_int2Bin64 ( uint64_t a )
  {
     
    for (int i=0; i <64; i++ )
    {
      if( a & (1 << i) ) 
        std::cout << 1;
      else
        std::cout << 0;
    }
    std::cout<<std::endl;
  }

  /**
  * @brief this method scales the _popc(xnor(...)) result
  * into the dot(-1...1) result
  * Example: if scale range is 8, then 
  * the dot product result based -1 and 1:
  * -8  -6  -4  -2  0 2 4 6 8
  * XNOR&POPC result:
  *  0   1   2   3  4 5 6 7 8
  * so the equa should be:
  * dot_ouput = 2 * xnor_output - scale_range
  */
  inline float xnor_to_binary_dot ( float num, int scale_range)
  {
    return 2*num - scale_range;
  }

  /**
   * @brief binarize an array of floats via the sign function into a single BINARY_WORD
   *
   */
  inline BINARY_WORD concatenate(float* array)
  {
    BINARY_WORD rvalue=0;
    BINARY_WORD sign;

    for (int i = 0; i < BITS_PER_BINARY_WORD; i++)
    {
      sign = (array[i]>=0);
      rvalue = rvalue | (sign<< (i));
    }

    return rvalue;
  }

  /**
   * @brief binarize matrix
   *
   */
  inline void get_binary_row(float* row, BINARY_WORD * b_row, int size){

    #pragma omp parallel for
    for (int i = 0; i < size; i+=BITS_PER_BINARY_WORD) {
      BINARY_WORD rvalue=0;
      BINARY_WORD sign;
      for (int j = 0;j < BITS_PER_BINARY_WORD; ++j) {
        sign = (row[i+j]>=0);
        BIT_SET(rvalue, j, sign);
      }
      b_row[i/BITS_PER_BINARY_WORD] = rvalue;
    }
  }

  /**
  * @brief binarize matrix column wise
  *
  */
  inline void get_binary_col(float* col, BINARY_WORD * b_col, int n, int k){        
    
    for(int y=0; y<(n/BITS_PER_BINARY_WORD); y++){
      #pragma omp parallel for
      for(int x=0; x < k; ++x){          
        BINARY_WORD rvalue=0;
        BINARY_WORD sign;    
        for(int b=0; b<BITS_PER_BINARY_WORD; ++b){
          sign = (col[(y*BITS_PER_BINARY_WORD+b)*k + x]>=0);          
          BIT_SET(rvalue, b, sign);
        }
        b_col[y*k + x] = rvalue;
      }
    }    
  }


  /**
  * @brief binarize matrix column wise. 
  * Loop unroll and using register vars.
  * ~30% performance improvement without openmp
  * compared with get_binary_col() method.
  */
  inline void get_binary_col_unrolled(float* col, BINARY_WORD * b_col, int n, int k){        

    for(int y=0; y<(n/BITS_PER_BINARY_WORD); y++){
      BINARY_WORD * y_col_pt = &b_col[y*k];
      #pragma omp parallel for
      for(int x=0; x < k; x+=4){          
        register BINARY_WORD rvalue0=0, rvalue1=0, rvalue2=0, rvalue3=0;
           
        for(int b=0; b<BITS_PER_BINARY_WORD; b+=4){
          register BINARY_WORD sign0, sign1, sign2, sign3, sign4, sign5, sign6, sign7,
          sign8, sign9, sign10, sign11, sign12, sign13, sign14, sign15;

          float* col_0 = &col[(y*BITS_PER_BINARY_WORD+b)*k + x];
          float* col_1 = &col[(y*BITS_PER_BINARY_WORD+b+1)*k + x];
          float* col_2 = &col[(y*BITS_PER_BINARY_WORD+b+2)*k + x];
          float* col_3 = &col[(y*BITS_PER_BINARY_WORD+b+3)*k + x];

          sign0 = (*col_0>=0);          
          sign1 = (*col_1>=0);          
          sign2 = (*col_2>=0);          
          sign3 = (*col_3>=0);          
         
          BIT_SET(rvalue0, b, sign0);
          BIT_SET(rvalue0, (b+1), sign1);
          BIT_SET(rvalue0, (b+2), sign2);
          BIT_SET(rvalue0, (b+3), sign3);

          sign4 = (*(col_0+1)>=0);          
          sign5 = (*(col_1+1)>=0);          
          sign6 = (*(col_2+1)>=0);          
          sign7 = (*(col_3+1)>=0);          
         
          BIT_SET(rvalue1, b, sign4);
          BIT_SET(rvalue1, (b+1), sign5);
          BIT_SET(rvalue1, (b+2), sign6);
          BIT_SET(rvalue1, (b+3), sign7);

          sign8 = (*(col_0+2)>=0);          
          sign9 = (*(col_1+2)>=0);          
          sign10 = (*(col_2+2)>=0);          
          sign11 = (*(col_3+2)>=0);          
         
          BIT_SET(rvalue2, b, sign8);
          BIT_SET(rvalue2, (b+1), sign9);
          BIT_SET(rvalue2, (b+2), sign10);
          BIT_SET(rvalue2, (b+3), sign11);

          sign12 = (*(col_0+3)>=0);          
          sign13 = (*(col_1+3)>=0);          
          sign14 = (*(col_2+3)>=0);          
          sign15 = (*(col_3+3)>=0);          
         
          BIT_SET(rvalue3, b, sign12);
          BIT_SET(rvalue3, (b+1), sign13);
          BIT_SET(rvalue3, (b+2), sign14);
          BIT_SET(rvalue3, (b+3), sign15);
        }
        BINARY_WORD * pnter = &y_col_pt[x];
        *pnter = rvalue0;   
        *(pnter+1) = rvalue1;        
        *(pnter+2) = rvalue2;        
        *(pnter+3) = rvalue3;        
      }
    }     
  }

  /**
   * @brief based-line xnor-gemm implementation without 
   * dot product, but use XNOR and POPCNT
   * __builtin_popcountll suitable for both 32bit and 64bit 
   *
   *
   */
  void xnor_gemm(int M, int N, int K,
                        BINARY_WORD *A, int lda,
                        BINARY_WORD *B, int ldb,
                        float *C, int ldc);

  /**
   * @brief simple naive baseline gemm implementation
   *
   */
  inline void baseline_gemm(int M, int K, int N,
                            float *A, int lda,
                            float *B, int ldb,
                            float *C, int ldc){
    int i,n,k;    
    for(i = 0; i < M; ++i){
      for(n = 0; n < N; ++n){
        float A_PART = A[i*lda+n];
        for(k = 0; k < K; ++k){
          C[i*ldc+k] += A_PART * B[n*ldb+k];
        }
      }
    }
  }

} //namespace xnor
} //namespace op
} //namespace mxnet
#endif //MXNET_OPERATOR_CONTRIB_BINARY_INFERENCE_XNOR_H
