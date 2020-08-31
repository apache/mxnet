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

#include "test.h"

using namespace mshadow;


int main() {
  InitTensorEngine<cpu>();
  InitTensorEngine<gpu>();
  Tensor<cpu, 3, float> tc = NewTensor<cpu, float>(Shape3(3, 2, 4), 0.0f);
  Tensor<gpu, 3, float> tg = NewTensor<gpu, float>(tc.shape_, 0.0f);
  // init
  for (index_t i = 0; i < tc.size(0); ++i) {
    for (index_t j = 0; j < tc.size(1); ++j) {
      for (index_t k = 0; k < tc.size(2); ++k) {
        tc[i][j][k] = i * 0.1f + j * 0.2f + k * 0.1f;
      }
    }
  }
  Copy(tg, tc);
  // print
  printf("\n#print batch 0 of cpu tensor:\n");
  Print2DTensor(tc[0]);
  printf("\n");
  Print2DTensor(tc[1]);
  printf("\n");
  Print2DTensor(tc[2]);
  // check
  if (Check2DTensor(tg[1], tc[1])) {
    printf("batch 1 of gpu & cpu tensor are same.\n");
  }
  // sum of row
  Tensor<cpu, 1, float> tmp_tc = NewTensor<cpu, float>(Shape1(tc[0].size(1)), 0.0f);
  Tensor<gpu, 1, float> tmp_tg = NewTensor<gpu, float>(Shape1(tg[0].size(1)), 0.0f);
  printf("\n#sum_rows of batch 0:\n");
  tmp_tc = sum_rows(tc[0]);
  tmp_tg = sum_rows(tg[0]);
  Print1DTensor(tmp_tc);
  if (Check1DTensor(tmp_tg, tmp_tc)) {
    printf("cpu & gpu result consists\n");
  }
  FreeSpace(&tmp_tc);
  FreeSpace(&tmp_tg);
  // sumall_except_dim
  printf("\n#sumall_except_dim<0> of batch 0:\n");
  Tensor<cpu, 1, float> red_tc = NewTensor<cpu, float>(Shape1(tc.size(0)), 0.0f);
  Tensor<gpu, 1, float> red_tg = NewTensor<gpu, float>(Shape1(tg.size(0)), 0.0f);
  red_tc = sumall_except_dim<0>(tc);
  red_tg = sumall_except_dim<0>(tg);
  Print1DTensor(red_tc);
  if (Check1DTensor(red_tg, red_tc)) {
    printf("cpu & gpu result consists\n");
  }
  FreeSpace(&red_tc);
  FreeSpace(&red_tg);
  // softmax
  printf("\n#Softmax\n");
  Tensor<cpu, 2, float> sm_tc = NewTensor<cpu, float>(tc[0].shape_, 0.0f);
  Tensor<gpu, 2, float> sm_tg = NewTensor<gpu, float>(tg[0].shape_, 0.0f);
  Softmax(sm_tc, tc[0]);
  Softmax(sm_tg, tg[0]);
  if (Check2DTensor(sm_tg, sm_tc)) {
    printf("cpu & gpu result consists\n");
  }
  // mirror
  printf("\n#mirror\n");
  sm_tc = mirror(tc[0]);
  sm_tg = mirror(tg[0]);
  if (Check2DTensor(sm_tg, sm_tc)) {
    printf("cpu & gpu result consists\n");
  }
  FreeSpace(&sm_tc);
  FreeSpace(&sm_tg);
  // reshape
  
  FreeSpace(&tc);
  FreeSpace(&tg);
  ShutdownTensorEngine<cpu>();
  ShutdownTensorEngine<gpu>();
}
