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
 * \file gpu_topology_test.cc
 * \brief gpu topology tests
*/

#include <gtest/gtest.h>
#include <mxnet/base.h>
#include <mxnet/kvstore.h>
#include "../src/kvstore/gpu_topology.h"

#define NUM_GPUS 8

// Permutes matrix W using permutation vector P and stores output in matrix A
// Assumption: W is square and symmetric
void PermuteMatrix( const std::vector<int>& W, 
                    const std::vector<int>& P, 
                    std::vector<int>&       A ) {
  int nrows = P.size();
  std::vector<int> temp(nrows*nrows,0);

  int count = 0;
  for (int row=0; row<nrows; ++row) {
    for (int col=0; col<nrows; ++col) {
      int row_start = P[row];
      temp[count] = W[row_start*nrows+col];
      count++;
    }
  }

  count = 0;
  for (int row=0; row<nrows; ++row) {
    for (int col=0; col<nrows; ++col) {
      int col_index = P[col];
      A[count] = temp[row*nrows+col_index];
      count++;
    }
  }
}

TEST(FormTopologyTest, TestFormTopology) {
  std::vector<int> state0  = {3, 2, 1, 5, 0, 0, 4, 6};
  std::vector<size_t> topo0;
  std::vector<size_t> scan0;
  std::vector<int> correct0= {3, 3, 0, 3, 1, 0, 4, 3, 2, 1, 5, 0, 0, 4, 6};
  std::vector<int> correct_scan0 = {0, 1, 3, 7, 15};
  mxnet::kvstore::FormTopology(state0, topo0, scan0, 3);
  ASSERT_EQ(topo0.size(), correct0.size());
  for (unsigned i = 0; i < correct0.size(); ++i)
    ASSERT_EQ(static_cast<int>(topo0[i]), correct0[i]);
  ASSERT_EQ(scan0.size(), correct_scan0.size());
  for (unsigned i = 0; i < correct_scan0.size(); ++i)
    ASSERT_EQ(scan0[i], correct_scan0[i]);

  std::vector<int> state1  = {3, 2, 0, 4, 1, 1, 5, 6};
  std::vector<size_t> topo1;
  std::vector<size_t> scan1;
  std::vector<int> correct1= {3, 3, 1, 3, 0, 1, 5, 3, 2, 0, 4, 1, 1, 5, 6};
  std::vector<int> correct_scan1 = {0, 1, 3, 7, 15};
  mxnet::kvstore::FormTopology(state1, topo1, scan1, 3);
  ASSERT_EQ(topo1.size(), correct1.size());
  for (unsigned i = 0; i < correct1.size(); ++i)
    ASSERT_EQ(topo1[i], correct1[i]);
  ASSERT_EQ(scan1.size(), correct_scan1.size());
  for (unsigned i = 0; i < correct_scan1.size(); ++i)
    ASSERT_EQ(scan1[i], correct_scan1[i]);
}

TEST(ComputeTreeWeightTest, TestComputeTreeWeight) {

  std::vector<int> W = {0, 2, 2, 3, 3, 0, 0, 
                        2, 0, 3, 2, 0, 3, 0,
                        2, 3, 0, 3, 0, 0, 2,
                        3, 2, 3, 0, 0, 0, 0,
                        3, 0, 0, 0, 0, 2, 2,
                        0, 3, 0, 0, 2, 0, 3,
                        0, 0, 2, 0, 2, 3, 0};

  std::vector<int> state0 = {3, 2, 1, 5, 0, 0, 4, 6};
  ASSERT_EQ(mxnet::kvstore::ComputeTreeWeight(W, state0, 7, 3), 16);

  std::vector<int> state1 = {3, 2, 0, 4, 1, 1, 5, 6};
  ASSERT_EQ(mxnet::kvstore::ComputeTreeWeight(W, state1, 7, 3), 17);
}

TEST(PostprocessTest, TestPostprocess) {
  std::vector<int> result0 = {3, 0, 0, 4, 1, 2, 5, 6};
  std::vector<int> correct0= {3, 3, 0, 4, 1, 2, 5, 6};
  mxnet::kvstore::Postprocess( result0, 7, 3 );
  for (unsigned i = 0; i < correct0.size(); ++i)
    ASSERT_EQ(result0[i], correct0[i]);

  std::vector<int> result1 = {2, 0, 0, 4, 1, 3, 5, 1};
  std::vector<int> correct1= {2, 2, 0, 4, 1, 3, 5, 5};
  mxnet::kvstore::Postprocess( result1, 6, 3 );
  for (unsigned i = 0; i < correct1.size(); ++i)
    ASSERT_EQ(result1[i], correct1[i]);

  std::vector<int> result2 = {5, 4, 1, 3, 1, 0, 2, 0};
  std::vector<int> correct2= {5, 4, 1, 3, 1, 0, 2, 2};
  mxnet::kvstore::Postprocess( result2, 6, 3 );
  for (unsigned i = 0; i < correct2.size(); ++i)
    ASSERT_EQ(result2[i], correct2[i]);
}

TEST(ComputeDepthTest, TestDepth) {
  ASSERT_EQ(mxnet::kvstore::ComputeDepth(8), 3);
  ASSERT_EQ(mxnet::kvstore::ComputeDepth(7), 3);
  ASSERT_EQ(mxnet::kvstore::ComputeDepth(5), 3);
  ASSERT_EQ(mxnet::kvstore::ComputeDepth(4), 2);
  ASSERT_EQ(mxnet::kvstore::ComputeDepth(16), 4);
}

TEST(IsValidTest, TestIsValid) {

  std::vector<int> W = {0, 2, 2, 3, 3, 0, 0, 
                        2, 0, 3, 2, 0, 3, 0,
                        2, 3, 0, 3, 0, 0, 2,
                        3, 2, 3, 0, 0, 0, 0,
                        3, 0, 0, 0, 0, 2, 2,
                        0, 3, 0, 0, 2, 0, 3,
                        0, 0, 2, 0, 2, 3, 0};

  std::vector<int> state0 = {3, 2, 1, 5, 0, 0, 4, 6};
  ASSERT_EQ(mxnet::kvstore::IsValid(W, state0, 7, 7, 3), true);

  // 3 connects to 1 first
  std::vector<int> state1 = {3, 2, 0, 4, 1, 1, 5, 6};
  ASSERT_EQ(mxnet::kvstore::IsValid(W, state1, 7, 7, 3), true);

  // 3 does not connect to 5
  std::vector<int> state2 = {3, 2, 5, 1, 0, 4, 2, 5};
  ASSERT_EQ(mxnet::kvstore::IsValid(W, state2, 7, 7, 3), false);

  // 7 exceeds number of GPUs
  std::vector<int> state3 = {3, 7, 2, 6, 0, 1, 4, 5};
  ASSERT_EQ(mxnet::kvstore::IsValid(W, state3, 7, 7, 3), false);

  // Test -1
  std::vector<int> state4 = {3, -1, 2, 6, 0, 1, 4, 5};
  ASSERT_EQ(mxnet::kvstore::IsValid(W, state4, 7, 7, 3), true);

  // Test -1
  std::vector<int> state5 = {3, -1, 2, 6, 0, 1, 4, -1};
  ASSERT_EQ(mxnet::kvstore::IsValid(W, state5, 7, 8, 3), false);

  // Test 1 row
  std::vector<int> state6 = {3, -1, -1, -1, -1, -1, -1, -1};
  ASSERT_EQ(mxnet::kvstore::IsValid(W, state6, 7, 1, 3), true);

}

TEST(PermuteMatrixTest, TestIdentity) {

  std::vector<int> W = {0, 2, 2, 3, 3, 1, 1, 1, 
                        2, 0, 3, 2, 1, 3, 1, 1,
                        2, 3, 0, 3, 1, 1, 2, 1,
                        3, 2, 3, 0, 1, 1, 1, 2,
                        3, 1, 1, 1, 0, 2, 2, 3,
                        1, 3, 1, 1, 2, 0, 3, 2,
                        1, 1, 2, 1, 2, 3, 0, 3,
                        1, 1, 1, 2, 3, 2, 3, 0};

  std::vector<int> P1 = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> A(NUM_GPUS*NUM_GPUS, 0);
  PermuteMatrix( W, P1, A );
  //PrintMatrix("P1", A, NUM_GPUS, NUM_GPUS);
  for (unsigned i=0; i<W.size(); ++i)
    ASSERT_EQ(A[i], W[i]);
}

TEST(KernighanLinTest, Test1) {
  std::vector<float> W = {0, 1, 2, 3, 2, 4,
                          1, 0, 1, 4, 2, 1,
                          2, 1, 0, 3, 2, 1,
                          3, 4, 3, 0, 4, 3,
                          2, 2, 2, 4, 0, 2,
                          4, 1, 1, 3, 2, 0};
  std::vector<int> P(6, 0);
  std::vector<std::pair<int,int>> cluster_pairs;
  int num_partitions = 1;
  std::mt19937 gen(1);
  mxnet::kvstore::KernighanLin( W, P, num_partitions, cluster_pairs, gen );

}
