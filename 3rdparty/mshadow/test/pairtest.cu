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
void Print(T const & ist, int I, int J) {
  for (int i = 0; i < I; ++i) {
    for (int j = 0; j < J; ++j) {
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
  const int O = (X - K) / 2 + 1;
  mshadow::TensorContainer<mshadow::cpu, 4, float> srcm(mshadow::Shape4(1,1,X, X));
  Xmshadow::TensorContainer<Xmshadow::cpu, 4> srcx(Xmshadow::Shape4(1,1,X, X));
  for (int i = 0; i < X; ++i) {
    for (int j = 0; j < X; ++j) {
      srcm[0][0][i][j] = i * 0.1f + j * 0.1f;
      srcx[0][0][i][j] = i * 0.1f + j * 0.1f;
    }
  }
  printf("Source:\n");
  Print(srcm[0][0], X, X);
  printf("\n");
  mshadow::TensorContainer<xpua, 4, float> mct(mshadow::Shape4(1,1,X, X));
  Xmshadow::TensorContainer<xpub, 4> xct(Xmshadow::Shape4(1,1,X, X));
  mshadow::Copy(mct, srcm);
  Xmshadow::Copy(xct, srcx);

  
  mshadow::TensorContainer<xpua, 4, float> pool_ct(mshadow::Shape4(1,1, O, O));
  Xmshadow::TensorContainer<xpub, 4> pool_xct(Xmshadow::Shape4(1,1,O,O));

  pool_ct = mshadow::expr::pool<mshadow::red::maximum>(mct, K, K, K);
  pool_xct = Xmshadow::expr::pool<Xmshadow::red::maximum>(xct, K, K);

  printf("New pool:\n");
  Print(pool_ct[0][0], O, O);
  printf("\nOld pool:\n");
  Print(pool_xct[0][0], O, O);
  printf("\n");
  mshadow::TensorContainer<mshadow::cpu, 4, float> gpool_src(mshadow::Shape4(1,1, O, O));
  Xmshadow::TensorContainer<Xmshadow::cpu, 4> gpool_xsrc(Xmshadow::Shape4(1,1,O,O));
  for (int i = 0; i < O; ++i) {
    for (int j = 0; j < O; ++j) {
      gpool_src[0][0][i][j] = 0.1f;
      gpool_xsrc[0][0][i][j] = 0.1f;
    }
  }
  mshadow::TensorContainer<xpua, 4, float> gpool_ct(mshadow::Shape4(1,1, O, O));
  Xmshadow::TensorContainer<xpub, 4> gpool_xct(Xmshadow::Shape4(1,1,O,O));
  mshadow::Copy(gpool_ct, gpool_src);
  Xmshadow::Copy(gpool_xct, gpool_xsrc);

  mshadow::TensorContainer<xpua, 4, float> mout(mshadow::Shape4(1,1,X, X));
  Xmshadow::TensorContainer<xpub, 4> xout(Xmshadow::Shape4(1,1,X, X));

  mout = mshadow::expr::unpool<mshadow::red::maximum>(mct, pool_ct, gpool_ct, K, K, K);
  xout = Xmshadow::expr::unpool<Xmshadow::red::maximum>(xct, pool_xct, gpool_xct, K, K);

  mshadow::Copy(srcm, mout);
  Xmshadow::Copy(srcx, xout);

  mshadow::TensorContainer<mshadow::cpu, 2> l1(mshadow::Shape2(X,X));
  Xmshadow::TensorContainer<Xmshadow::cpu, 2> l2(Xmshadow::Shape2(X, X));
  l1 = mshadow::expr::reshape(srcm, l1.shape_);
  l2 = Xmshadow::expr::reshape(srcx, l2.shape);
  printf("New unpool\n");
  Print(l1, l1.size(0), l1.size(1));
  printf("\nOld unpool\n");
  Print(l2, X, X);
  if (Check(l1, l2)) {
    printf("Pass\n");
  }
}

int main(int argc, char** argv) {
  if (argc < 1) {
    printf("Usage: dev\n");
    exit(-1);
  }
  if (!strcmp(argv[1], "cpu")) {
    RunTask<mshadow::cpu, Xmshadow::cpu>();
  } else {
    RunTask<mshadow::gpu, Xmshadow::gpu>();
  }
}
