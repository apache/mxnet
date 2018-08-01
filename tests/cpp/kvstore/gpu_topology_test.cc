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

void GenerateMatrix(std::vector<float>* W, int num_gpus, float k,
                    std::mt19937* gen) {
  std::uniform_real_distribution<> dis(0., 1.);
  for (int row = 0; row < num_gpus; ++row) {
    for (int col = row+1; col < num_gpus; ++col) {
      float sample = dis(*gen);
      if (sample < k)
        continue;
      sample = dis(*gen);
      if (sample < 0.33f) {
        (*W)[row*num_gpus+col] = 1.f;
        (*W)[col*num_gpus+row] = 1.f;
      } else if (sample < 0.66f) {
        (*W)[row*num_gpus+col] = 2.f;
        (*W)[col*num_gpus+row] = 2.f;
      } else {
        (*W)[row*num_gpus+col] = 3.f;
        (*W)[col*num_gpus+row] = 3.f;
      }
    }
  }
}

bool IsSatisfactory(const std::vector<float>& W, int num_gpus, int depth) {
  for (int row = 0; row < num_gpus; ++row) {
    int out_edges = 0;
    for (int col = 0; col < num_gpus; ++col) {
      if (W[row*num_gpus+col] > 0.f)
        out_edges++;
    }
    if (out_edges < depth)
      return false;
  }
  return true;
}

// Generates random link topology matrix using random number generator
void TestComputeTreesRandomized(int num_gpus, float alpha, int backtrack,
                                std::mt19937* gen) {
  std::uniform_real_distribution<> dis(0.f, 1.f);
  bool satisfied = false;
  std::vector<float> W(num_gpus*num_gpus, 0.f);
  int depth = mxnet::kvstore::ComputeDepth(num_gpus);
  while (!satisfied) {
    float k = dis(*gen);
    std::fill(W.begin(), W.end(), 0.f);
    GenerateMatrix(&W, num_gpus, k, gen);
    satisfied = IsSatisfactory(W, num_gpus, depth);
  }

  std::vector<std::vector<size_t>> topo;
  std::vector<std::vector<size_t>> scan;
  mxnet::kvstore::ComputeTrees(W, num_gpus, alpha, backtrack, &topo, &scan);

  unsigned correct_topo_size = (1 << (depth + 1)) - 1;
  unsigned correct_scan_size = depth+2;
  ASSERT_EQ(topo.size(), static_cast<unsigned>(num_gpus));
  for (unsigned i = 0; i < topo.size(); ++i) {
    ASSERT_EQ(correct_topo_size, topo[i].size());
    ASSERT_EQ(correct_scan_size, scan[i].size());
  }
}

// Permutes matrix W using permutation vector P and stores output in matrix A
// Assumption: W is square and symmetric
void PermuteMatrix(const std::vector<int>& W,
                   const std::vector<int>& P,
                   std::vector<int>*       A) {
  int nrows = P.size();
  std::vector<int> temp(nrows*nrows, 0);

  int count = 0;
  for (int row=0; row < nrows; ++row) {
    for (int col=0; col < nrows; ++col) {
      int row_start = P[row];
      temp[count] = W[row_start*nrows+col];
      count++;
    }
  }

  count = 0;
  for (int row=0; row < nrows; ++row) {
    for (int col=0; col < nrows; ++col) {
      int col_index = P[col];
      (*A)[count] = temp[row*nrows+col_index];
      count++;
    }
  }
}

TEST(GpuTopology, TestFormTopology) {
  std::vector<int> state0 = {3, 2, 1, 5, 0, 0, 4, 6};
  std::vector<size_t> topo0;
  std::vector<size_t> scan0;
  std::vector<int> correct0 = {3, 3, 0, 3, 1, 0, 4, 3, 2, 1, 5, 0, 0, 4, 6};
  std::vector<int> correct_scan0 = {0, 1, 3, 7, 15};
  mxnet::kvstore::FormTopology(state0, &topo0, &scan0, 3);
  ASSERT_EQ(topo0.size(), correct0.size());
  for (unsigned i = 0; i < correct0.size(); ++i)
    ASSERT_EQ(static_cast<int>(topo0[i]), correct0[i]);
  ASSERT_EQ(scan0.size(), correct_scan0.size());
  for (unsigned i = 0; i < correct_scan0.size(); ++i)
    ASSERT_EQ(static_cast<int>(scan0[i]), correct_scan0[i]);

  std::vector<int> state1 = {3, 2, 0, 4, 1, 1, 5, 6};
  std::vector<size_t> topo1;
  std::vector<size_t> scan1;
  std::vector<int> correct1 = {3, 3, 1, 3, 0, 1, 5, 3, 2, 0, 4, 1, 1, 5, 6};
  std::vector<int> correct_scan1 = {0, 1, 3, 7, 15};
  mxnet::kvstore::FormTopology(state1, &topo1, &scan1, 3);
  ASSERT_EQ(topo1.size(), correct1.size());
  for (unsigned i = 0; i < correct1.size(); ++i)
    ASSERT_EQ(static_cast<int>(topo1[i]), correct1[i]);
  ASSERT_EQ(scan1.size(), correct_scan1.size());
  for (unsigned i = 0; i < correct_scan1.size(); ++i)
    ASSERT_EQ(static_cast<int>(scan1[i]), correct_scan1[i]);
}

TEST(GpuTopology, TestComputeTreeWeight) {
  std::vector<int> W = {0, 2, 2, 3, 3, 0, 0,
                        2, 0, 3, 2, 0, 3, 0,
                        2, 3, 0, 3, 0, 0, 2,
                        3, 2, 3, 0, 0, 0, 0,
                        3, 0, 0, 0, 0, 2, 2,
                        0, 3, 0, 0, 2, 0, 3,
                        0, 0, 2, 0, 2, 3, 0};

  std::vector<int> state0 = {3, 2, 1, 5, 0, 0, 4, 6};
  ASSERT_EQ(mxnet::kvstore::ComputeTreeWeight(W, state0, 7, 3, false), 16);

  std::vector<int> state1 = {3, 2, 0, 4, 1, 1, 5, 6};
  ASSERT_EQ(mxnet::kvstore::ComputeTreeWeight(W, state1, 7, 3, false), 17);
}

TEST(GpuTopology, TestPostprocess) {
  std::vector<int> result0  = {3, 0, 0, 4, 1, 2, 5, 6};
  std::vector<int> correct0 = {3, 3, 0, 4, 1, 2, 5, 6};
  mxnet::kvstore::Postprocess(&result0, 7, 3);
  for (unsigned i = 0; i < correct0.size(); ++i)
    ASSERT_EQ(result0[i], correct0[i]);

  std::vector<int> result1  = {2, 0, 0, 4, 1, 3, 5, 1};
  std::vector<int> correct1 = {2, 2, 0, 4, 1, 3, 5, 5};
  mxnet::kvstore::Postprocess(&result1, 6, 3);
  for (unsigned i = 0; i < correct1.size(); ++i)
    ASSERT_EQ(result1[i], correct1[i]);

  std::vector<int> result2  = {5, 4, 1, 3, 1, 0, 2, 0};
  std::vector<int> correct2 = {5, 4, 5, 3, 1, 0, 2, 2};
  mxnet::kvstore::Postprocess(&result2, 6, 3);
  for (unsigned i = 0; i < correct2.size(); ++i)
    ASSERT_EQ(result2[i], correct2[i]);

  std::vector<int> result3  = {10, 10,  0,  0, 0, 0, 0, 1, 2, 3, 6, 4, 7, 5, 8, 9};
  std::vector<int> correct3 = {10, 10, 10, 10, 0, 0, 0, 1, 2, 3, 6, 4, 7, 5, 8, 9};
  mxnet::kvstore::Postprocess(&result3, 11, 4);
  for (unsigned i = 0; i < correct3.size(); ++i)
    ASSERT_EQ(result3[i], correct3[i]);
}

TEST(GpuTopology, TestDepth) {
  ASSERT_EQ(mxnet::kvstore::ComputeDepth(2), 1);
  ASSERT_EQ(mxnet::kvstore::ComputeDepth(3), 2);
  ASSERT_EQ(mxnet::kvstore::ComputeDepth(8), 3);
  ASSERT_EQ(mxnet::kvstore::ComputeDepth(7), 3);
  ASSERT_EQ(mxnet::kvstore::ComputeDepth(5), 3);
  ASSERT_EQ(mxnet::kvstore::ComputeDepth(4), 2);
  ASSERT_EQ(mxnet::kvstore::ComputeDepth(16), 4);
}

TEST(GpuTopology, TestIsValid) {
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

// gemvTest
TEST(GpuTopology, TestGemv) {
  std::vector<int> A = {0, 2, 2, 3, 3, 1, 1, 1,   // 13
                        2, 0, 3, 2, 1, 3, 1, 1,   // 13
                        2, 3, 0, 3, 1, 1, 2, 1,   // 13
                        3, 2, 3, 0, 1, 1, 1, 2,   // 13
                        3, 1, 1, 1, 0, 2, 2, 3,   // 13
                        1, 3, 1, 1, 2, 0, 3, 2,   // 13
                        1, 1, 2, 1, 2, 3, 0, 3,   // 13
                        1, 1, 1, 2, 3, 2, 3, 0};  // 13
  std::vector<int> x(8, 1);
  std::vector<int> y(8, 0);
  std::iota(y.begin(), y.end(), 0);
  std::vector<int> correct_y(8, 13);
  mxnet::kvstore::gemv(A, x, &y);

  ASSERT_EQ(y.size(), correct_y.size());
  for (unsigned i = 0; i < y.size(); ++i)
    ASSERT_EQ(y[i], correct_y[i]);
}

// ewisemultTest
TEST(GpuTopology, TestEwisemult) {
  std::vector<int> x(8, 1);
  std::vector<int> y(8, 0);
  std::iota(y.begin(), y.end(), 0);
  int alpha = 5;
  std::vector<int> correct_y = {0, 5, 10, 15, 20, 25, 30, 35};
  mxnet::kvstore::ewisemult(x, alpha, &y);

  ASSERT_EQ(y.size(), correct_y.size());
  for (unsigned i = 0; i < y.size(); ++i)
    ASSERT_EQ(y[i], correct_y[i]);
}

// FindBestMoveTest
TEST(GpuTopology, TestFindBestMove) {
  std::vector<int> W = {0, 2, 2, 3, 3, 1, 1, 1,
                        2, 0, 3, 2, 1, 3, 1, 1,
                        2, 3, 0, 3, 1, 1, 2, 1,
                        3, 2, 3, 0, 1, 1, 1, 2,
                        3, 1, 1, 1, 0, 2, 2, 3,
                        1, 3, 1, 1, 2, 0, 3, 2,
                        1, 1, 2, 1, 2, 3, 0, 3,
                        1, 1, 1, 2, 3, 2, 3, 0};
  std::vector<int> P(8, 0);
  std::iota(P.begin(), P.end(), 1);
  std::unordered_set<int> used;

  std::vector<int> D1 = {20, 0, 0, 0, 0, 0, 0, 20};
  int a1, b1, g1;
  int correct_a1 = 0;
  int correct_b1 = 7;
  int correct_g1 = 38;
  mxnet::kvstore::FindBestMove(W, P, D1, used, &a1, &b1, &g1);
  ASSERT_EQ(a1, correct_a1);
  ASSERT_EQ(b1, correct_b1);
  ASSERT_EQ(g1, correct_g1);

  // -1, -1, 0 indicates no best edge found
  std::vector<int> D2 = {0, 0, 0, 0, 0, 0, 0, 0};
  int a2, b2, g2;
  int correct_a2 = -1;
  int correct_b2 = -1;
  int correct_g2 = 0;
  mxnet::kvstore::FindBestMove(W, P, D2, used, &a2, &b2, &g2);
  ASSERT_EQ(a2, correct_a2);
  ASSERT_EQ(b2, correct_b2);
  ASSERT_EQ(g2, correct_g2);
}

// GetRootTest
TEST(GpuTopology, TestGetRoot) {
  std::vector<int> P = {0, 0, 1, 1, 2, 2, 3, 3};

  // Test when roots are non-empty, and matches color
  std::unordered_set<int> roots1 = {0, 2, 4, 6};
  std::vector<int> color1 = {0, 1, 2, 3};
  for (unsigned i = 0; i < color1.size(); ++i) {
    int root1 = mxnet::kvstore::GetRoot(P, color1[i], roots1);
    int correct_root1 = 2*i;
    ASSERT_EQ(root1, correct_root1);
  }

  // Test when roots is empty
  std::unordered_set<int> roots2;
  int color2 = 0;
  int correct_root2 = -1;
  int root2  = mxnet::kvstore::GetRoot(P, color2, roots2);
  ASSERT_EQ(root2, correct_root2);

  // Test when roots is non-empty, but no root matches color
  std::unordered_set<int> roots3 = {0};
  int color3 = 1;
  int correct_root3 = -1;
  int root3  = mxnet::kvstore::GetRoot(P, color3, roots3);
  ASSERT_EQ(root3, correct_root3);

  std::vector<int> P2 = {0, 1, 1, 0, 2, 3, 3, 2};
  std::unordered_set<int> roots4 = roots1;
  int color4 = 0;
  int correct_root4 = 0;
  int root4 = mxnet::kvstore::GetRoot(P, color4, roots4);
  ASSERT_EQ(root4, correct_root4);
}

// GetChildTest
TEST(GpuTopology, TestGetChild) {
  std::vector<int> P = {0, 0, 1, 2, 2, 2, 3, 3};

  // Test when color is not found
  int color1  = 4;
  int parent1 = 4;
  int correct_child1 = -1;
  int child1  = mxnet::kvstore::GetChild(P, color1, parent1);
  ASSERT_EQ(child1, correct_child1);

  // Test when color is found, but is equal to parent
  int color2  = 1;
  int parent2 = 2;
  int correct_child2 = -1;
  int child2  = mxnet::kvstore::GetChild(P, color2, parent2);
  ASSERT_EQ(child2, correct_child2);

  // Test when color is found and not equal to parent
  int color3  = 3;
  int parent3 = 6;
  int correct_child3 = 7;
  int child3  = mxnet::kvstore::GetChild(P, color3, parent3);
  ASSERT_EQ(child3, correct_child3);
}

// FindBestEdgeTest
TEST(GpuTopology, TestFindBestEdge) {
  std::vector<int> W = {0, 2, 2, 3, 3, 1, 1, 1,
                        2, 0, 3, 2, 1, 3, 1, 1,
                        2, 3, 0, 3, 1, 1, 2, 1,
                        3, 2, 3, 0, 1, 1, 1, 2,
                        3, 1, 1, 1, 0, 2, 2, 3,
                        1, 3, 1, 1, 2, 0, 3, 2,
                        1, 1, 2, 1, 2, 3, 0, 3,
                        1, 1, 1, 2, 3, 2, 3, 0};
  std::vector<int> P(8, 0);
  std::unordered_set<int> used;

  int parent1 = 3;
  int dest1   = 0;
  std::vector<int> b1;
  int g1;
  std::vector<int> correct_b1 = {0, 2};
  int correct_g1 = 3;
  mxnet::kvstore::FindBestEdge(W, P, parent1, dest1, &b1, &g1);
  ASSERT_EQ(b1.size(), correct_b1.size());
  for (unsigned i = 0; i < b1.size(); ++i)
    ASSERT_EQ(b1[i], correct_b1[i]);
  ASSERT_EQ(g1, correct_g1);

  // {-1}, 0 indicates no best edge found
  int parent2 = 4;
  int dest2   = 1;
  std::vector<int> b2;
  int g2;
  std::vector<int> correct_b2 = {-1};
  int correct_g2 = 0;
  mxnet::kvstore::FindBestEdge(W, P, parent2, dest2, &b2, &g2);
  ASSERT_EQ(b2.size(), correct_b2.size());
  for (unsigned i = 0; i < b2.size(); ++i)
    ASSERT_EQ(b2[i], correct_b2[i]);
  ASSERT_EQ(g2, correct_g2);
}

// KLGenerateBinaryTreeTest
TEST(GpuTopology, TestKLGenerateBinaryTree1) {
  std::vector<int> W = {0, 2, 3, 3, 3, 1, 1, 1,
                        2, 0, 3, 2, 1, 3, 1, 1,
                        2, 3, 0, 3, 1, 1, 2, 1,
                        3, 2, 3, 0, 1, 1, 1, 2,
                        3, 1, 1, 1, 0, 2, 3, 3,
                        1, 3, 1, 1, 2, 0, 3, 2,
                        1, 1, 2, 1, 2, 3, 0, 3,
                        1, 1, 1, 2, 3, 2, 3, 0};
  std::vector<int> P = {0, 1, 1, 0, 2, 3, 3, 2};
  std::vector<std::pair<int, int>> cluster_pairs;
  cluster_pairs.push_back(std::pair<int, int>(0, -2));
  cluster_pairs.push_back(std::pair<int, int>(1, -2));
  cluster_pairs.push_back(std::pair<int, int>(2, -2));
  cluster_pairs.push_back(std::pair<int, int>(3, -2));
  std::unordered_set<int> roots = {0, 2, 4, 6};
  std::vector<size_t> topo = {0, 2, 4, 6};
  std::vector<size_t> scan(2, 0);
  std::mt19937 gen(1);
  mxnet::kvstore::KLGenerateBinaryTree(W, P, &cluster_pairs, &roots, &topo,
                                       &scan, &gen);
  std::vector<size_t> correct_topo = {0, 2, 4, 6, 0, 3, 2, 1, 4, 7, 6, 5};
  std::vector<size_t> correct_scan = {0, 0, 4};
  ASSERT_EQ(topo.size(), correct_topo.size());
  for (unsigned i = 0; i < topo.size(); ++i)
    ASSERT_EQ(topo[i], correct_topo[i]);
  ASSERT_EQ(scan.size(), correct_scan.size());
  for (unsigned i = 0; i < scan.size(); ++i)
    ASSERT_EQ(scan[i], correct_scan[i]);
}

TEST(GpuTopology, TestKLGenerateBinaryTree2) {
  std::vector<int> W = {0, 2, 3, 3, 3, 1, 1, 1,
                        2, 0, 3, 2, 1, 3, 1, 1,
                        2, 3, 0, 3, 1, 1, 2, 1,
                        3, 2, 3, 0, 1, 1, 1, 2,
                        3, 1, 1, 1, 0, 2, 3, 3,
                        1, 3, 1, 1, 2, 0, 3, 2,
                        1, 1, 2, 1, 2, 3, 0, 3,
                        1, 1, 1, 2, 3, 2, 3, 0};
  std::vector<int> P = {0, 1, 1, 0, 2, 3, 3, 2};
  std::vector<std::pair<int, int>> cluster_pairs;
  cluster_pairs.push_back(std::pair<int, int>(0, -2));
  cluster_pairs.push_back(std::pair<int, int>(1, -2));
  cluster_pairs.push_back(std::pair<int, int>(2, -2));
  cluster_pairs.push_back(std::pair<int, int>(3, -2));
  std::unordered_set<int> roots = {0, 2, 4, 6};
  std::vector<size_t> topo = {0, 6, 4, 2};
  std::vector<size_t> scan(2, 0);
  std::mt19937 gen(1);
  mxnet::kvstore::KLGenerateBinaryTree(W, P, &cluster_pairs, &roots, &topo,
                                       &scan, &gen);
  std::vector<size_t> correct_topo = {0, 6, 4, 2, 0, 3, 6, 5, 4, 7, 2, 1};
  std::vector<size_t> correct_scan = {0, 0, 4};
  ASSERT_EQ(topo.size(), correct_topo.size());
  for (unsigned i = 0; i < topo.size(); ++i)
    ASSERT_EQ(topo[i], correct_topo[i]);
  ASSERT_EQ(scan.size(), correct_scan.size());
  for (unsigned i = 0; i < scan.size(); ++i)
    ASSERT_EQ(scan[i], correct_scan[i]);
}

// UpdateWeightTest
TEST(GpuTopology, TestUpdateWeight) {
  std::vector<float> W = {0.f, 1.f,
                          1.f, 0.f};
  std::vector<size_t> topo = {1, 1, 0};
  int num_gpus = 2;
  float alpha  = 0.7;
  std::vector<float> correct_W = {0.f, 0.7f,
                                  0.7f, 0.f};
  mxnet::kvstore::UpdateWeight(&W, topo, num_gpus, alpha);
  ASSERT_EQ(W.size(), correct_W.size());
  for (unsigned i = 0; i < W.size(); ++i) {
    ASSERT_EQ(W[i], correct_W[i]);
  }
}

// ComputeTreesFromRoot
TEST(GpuTopology, TestComputeTreesFromRoot1) {
  std::vector<float> W = {0, 2, 2, 3, 3, 1, 1, 1,
                          2, 0, 3, 2, 1, 3, 1, 1,
                          2, 3, 0, 3, 1, 1, 2, 1,
                          3, 2, 3, 0, 1, 1, 1, 2,
                          3, 1, 1, 1, 0, 2, 2, 3,
                          1, 3, 1, 1, 2, 0, 3, 2,
                          1, 1, 2, 1, 2, 3, 0, 3,
                          1, 1, 1, 2, 3, 2, 3, 0};
  int num_gpus = 8;
  int root     = 0;
  float alpha  = 0.7;
  bool backtrack = true;
  unsigned correct_topo_size = 15;
  unsigned correct_scan_size = 5;
  std::vector<size_t> topo;
  std::vector<size_t> scan;

  mxnet::kvstore::ComputeTreesFromRoot(&W, num_gpus, root, alpha, backtrack,
                                       &topo, &scan);

  ASSERT_EQ(topo.size(), correct_topo_size);
  ASSERT_EQ(scan.size(), correct_scan_size);
}

// IsConnected
// Test on graph that is "disconnected" by NVLink
TEST(GpuTopology, TestIsConnected1) {
  std::vector<float> W = {0, 0, 2, 0,
                          0, 0, 0, 2,
                          2, 0, 0, 0,
                          0, 2, 0, 0};
  int num_gpus = 4;

  bool connected = mxnet::kvstore::IsConnected(W, num_gpus);

  bool correct_connected = false;
  ASSERT_EQ(connected, correct_connected);
}

// IsConnected
// Test on graph that is "disconnected" by NVLink
TEST(GpuTopology, TestIsConnected2) {
  std::vector<float> W = {1, 1, 2, 1,
                          1, 1, 1, 2,
                          2, 1, 1, 1,
                          1, 2, 1, 1};
  int num_gpus = 4;

  bool connected = mxnet::kvstore::IsConnected(W, num_gpus);

  bool correct_connected = false;
  ASSERT_EQ(connected, correct_connected);
}

// IsConnected
// Test on graph that is "disconnected" by NVLink
TEST(GpuTopology, TestIsConnected3) {
  std::vector<float> W = {1, 1, 2, 2,
                          1, 1, 1, 2,
                          2, 1, 1, 1,
                          2, 2, 1, 1};
  int num_gpus = 4;

  bool connected = mxnet::kvstore::IsConnected(W, num_gpus);

  bool correct_connected = true;
  ASSERT_EQ(connected, correct_connected);
}

// ComputeTreesTest with backtracking
TEST(GpuTopology, TestComputeTrees1) {
  std::mt19937 gen(1);
  float alpha = 0.7;
  bool backtrack = true;
  // Do 5 randomized tests per GPU count from 2 to 16
  for (int num_gpus = 2; num_gpus <= 16; ++num_gpus) {
    LOG(INFO) << "Testing " << num_gpus << " x " << num_gpus;
    for (int i = 0; i < 5; ++i) {
      TestComputeTreesRandomized(num_gpus, alpha, backtrack, &gen);
    }
  }
}

// ComputeTreesTest with Kernighan-Lin
TEST(GpuTopology, TestComputeTrees2) {
  std::mt19937 gen(1);
  float alpha = 0.7;
  bool backtrack = false;
  // Do 5 randomized tests per GPU count from 2 to 16
  for (int num_gpus = 2; num_gpus <= 16; ++num_gpus) {
    LOG(INFO) << "Testing " << num_gpus << " x " << num_gpus;
    for (int i = 0; i < 5; ++i) {
      TestComputeTreesRandomized(num_gpus, alpha, backtrack, &gen);
    }
  }
}

TEST(GpuTopology, TestPermuteMatrix) {
  std::vector<int> W = {0, 2, 2, 3, 3, 1, 1, 1,
                        2, 0, 3, 2, 1, 3, 1, 1,
                        2, 3, 0, 3, 1, 1, 2, 1,
                        3, 2, 3, 0, 1, 1, 1, 2,
                        3, 1, 1, 1, 0, 2, 2, 3,
                        1, 3, 1, 1, 2, 0, 3, 2,
                        1, 1, 2, 1, 2, 3, 0, 3,
                        1, 1, 1, 2, 3, 2, 3, 0};

  std::vector<int> P1 = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> A(8*8, 0);
  PermuteMatrix(W, P1, &A);
  for (unsigned i=0; i < W.size(); ++i)
    ASSERT_EQ(A[i], W[i]);
}

TEST(GpuTopology, TestKernighanLin1) {
  std::vector<float> W = {0, 1, 2, 3, 2, 4,
                          1, 0, 1, 4, 2, 1,
                          2, 1, 0, 3, 2, 1,
                          3, 4, 3, 0, 4, 3,
                          2, 2, 2, 4, 0, 2,
                          4, 1, 1, 3, 2, 0};
  std::vector<int> P(6, 0);
  std::vector<std::pair<int, int>> cluster_pairs;
  int num_partitions = 1;
  std::mt19937 gen(1);
  bool stop = mxnet::kvstore::KernighanLin(W, &P, &num_partitions,
                                           &cluster_pairs, &gen);

  std::vector<std::pair<int, int>> correct_pairs;
  correct_pairs.push_back(std::pair<int, int>(0, 1));
  std::vector<int> correct_P = {0, 1, 0, 1, 1, 0};
  ASSERT_EQ(stop, false);
  ASSERT_EQ(num_partitions, 2);
  ASSERT_EQ(cluster_pairs.size(), correct_pairs.size());
  for (unsigned i = 0; i < cluster_pairs.size(); ++i) {
    ASSERT_EQ(cluster_pairs[i].first, correct_pairs[i].first);
    ASSERT_EQ(cluster_pairs[i].second, correct_pairs[i].second);
  }
  ASSERT_EQ(P.size(), correct_P.size());
  unsigned error = 0;
  for (unsigned i = 0; i < P.size(); ++i) {
    if (P[i] != correct_P[i])
      error++;
  }
  EXPECT_TRUE(error == 0 || error == P.size())
           << "Where real value: "   << error
           << " not equal neither: " << 0
           << " nor: "               << P.size() << ".";
}

TEST(GpuTopology, TestKernighanLin2) {
  std::vector<float> W = {0, 1, 0, 0, 1, 1, 0, 0,
                           1, 0, 0, 0, 1, 1, 0, 0,
                           0, 0, 0, 1, 0, 1, 1, 1,
                           0, 0, 1, 0, 0, 0, 1, 1,
                           1, 1, 0, 0, 0, 1, 0, 0,
                           1, 1, 1, 0, 1, 0, 0, 0,
                           0, 0, 1, 1, 0, 0, 0, 1,
                           0, 0, 1, 1, 0, 0, 1, 0};
  std::vector<int> P(8, 0);
  std::vector<std::pair<int, int>> cluster_pairs;
  int num_partitions = 1;
  std::mt19937 gen(1);
  bool stop = mxnet::kvstore::KernighanLin(W, &P, &num_partitions,
                                           &cluster_pairs, &gen);

  std::vector<std::pair<int, int>> correct_pairs;
  correct_pairs.push_back(std::pair<int, int>(0, 1));
  std::vector<int> correct_P = {0, 0, 1, 1, 0, 0, 1, 1};
  ASSERT_EQ(stop, false);
  ASSERT_EQ(num_partitions, 2);
  ASSERT_EQ(cluster_pairs.size(), correct_pairs.size());
  for (unsigned i = 0; i < cluster_pairs.size(); ++i) {
    ASSERT_EQ(cluster_pairs[i].first, correct_pairs[i].first);
    ASSERT_EQ(cluster_pairs[i].second, correct_pairs[i].second);
  }
  ASSERT_EQ(P.size(), correct_P.size());
  unsigned error = 0;
  for (unsigned i = 0; i < P.size(); ++i) {
    if (P[i] != correct_P[i])
      error++;
  }
  EXPECT_TRUE(error == 0 || error == P.size())
           << "Where real value: "   << error
           << " not equal neither: " << 0
           << " nor: "               << P.size() << ".";
}
