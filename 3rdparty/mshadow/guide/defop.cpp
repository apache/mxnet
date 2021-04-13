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

#include <cmath>
// header file to use mshadow
#include "mshadow/tensor.h"
// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

// user defined unary operator addone
struct addone {
  // map can be template function
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return  a + static_cast<DType>(1);
  }
};
// user defined binary operator max of two
struct maxoftwo {
  // map can also be normal functions,
  // however, this can only be applied to float tensor
  MSHADOW_XINLINE static float Map(float a, float b) {
    if(a > b) return a;
    else return b;
  }
};

int main(void) {
  // intialize tensor engine before using tensor operation, needed for CuBLAS
  InitTensorEngine<cpu>();
  // take first subscript of the tensor
  Stream<cpu> *stream_ = NewStream<cpu>(0);
  Tensor<cpu,2, float> mat = NewTensor<cpu>(Shape2(2,3), 0.0f, stream_);
  Tensor<cpu,2, float> mat2= NewTensor<cpu>(Shape2(2,3), 0.0f, stream_);

  mat[0][0] = -2.0f;
  mat = F<maxoftwo>(F<addone>(mat) + 0.5f, mat2);

  for (index_t i = 0; i < mat.size(0); ++i) {
    for (index_t j = 0; j < mat.size(1); ++j) {
      printf("%.2f ", mat[i][j]);
    }
    printf("\n");
  }
  FreeSpace(&mat); FreeSpace(&mat2);
  DeleteStream(stream_);
  // shutdown tensor enigne after usage
  ShutdownTensorEngine<cpu>();
  return 0;
}
