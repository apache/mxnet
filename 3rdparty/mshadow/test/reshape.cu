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

#include "mshadow/tensor.h"
#include "old/tensor.h"
#include "assert.h"
#include <cstring>

using mshadow::index_t;
template<typename T>
void Print(T const & ist) {
  for (int i = 0; i < ist.size(0); ++i) {
    for (int j = 0; j < ist.size(1); ++j) {
      printf("%.2f ", ist[i][j]);
    }
    printf("\n");
  }
}

bool Check(mshadow::TensorContainer<mshadow::cpu, 2, float> &mct, \
           Xmshadow::TensorContainer<Xmshadow::cpu, 2> &xct) {
  for (index_t i = 0; i < mct.size(0); ++i) {
    for (index_t j = 0; j < mct.size(1); ++j) {
      assert(mct[i][j] == xct[i][j]);
    }
  }
  return true;
}

template<typename xpua, typename xpub>
void RunTask() {
  const int X = 6;
  const int K = 2;
  mshadow::TensorContainer<mshadow::cpu, 2, float> srcm(mshadow::Shape2(X, X));
  Xmshadow::TensorContainer<Xmshadow::cpu, 2> srcx(Xmshadow::Shape2(X, X));
  
  mshadow::TensorContainer<xpua, 2, float> mct(mshadow::Shape2(X, X));
  Xmshadow::TensorContainer<xpub, 2> xct(Xmshadow::Shape2(X, X));
  for (int i = 0; i < X; ++i) {
    for (int j = 0; j < X; ++j) {
      srcm[i][j] = i * 0.1f + j * 0.1f;
      srcx[i][j] = i * 0.1f + j * 0.1f;
    }
  }
  mshadow::Copy(mct, srcm);
  Xmshadow::Copy(xct, srcx);

  mshadow::TensorContainer<xpua, 4, float> mct4d(mshadow::Shape4(1, 1, X / K, X * K));
  Xmshadow::TensorContainer<xpub, 4> xct4d(Xmshadow::Shape4(X / K, X * K, 1, 1));
  
  mct4d = mshadow::expr::reshape(mct, mct4d.shape_);
  xct4d = Xmshadow::expr::reshape(xct, xct4d.shape);
  
  mct = mshadow::expr::reshape(mct4d, mct.shape_);
  xct = Xmshadow::expr::reshape(xct4d, xct.shape);
  
  mshadow::TensorContainer<mshadow::cpu, 2, float> m_ct(mshadow::Shape2(X, X));
  Xmshadow::TensorContainer<Xmshadow::cpu, 2> x_ct(Xmshadow::Shape2(X, X));
  
  mshadow::Copy(m_ct, mct);
  Xmshadow::Copy(x_ct, xct);
  if (Check(m_ct, x_ct)) {
    printf("Pass\n");
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("Usage: dev\n");
    exit(-1);
  }
  if (!strcmp(argv[1], "cpu")) {
    RunTask<mshadow::cpu, Xmshadow::cpu>();
  } else {
    RunTask<mshadow::gpu, Xmshadow::gpu>();
  }
}
