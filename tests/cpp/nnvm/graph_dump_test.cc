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

#include <gtest/gtest.h>
#include "nnvm/graph_dump.h"

using namespace nnvm;
using namespace std;

/*
 * Test that enum and string values are in sync
 */
TEST(Graph_dump, basic) {
  auto x = NodeEntry(Node::Create(nullptr, "x"));
  auto w = NodeEntry(Node::Create(nullptr, "w"));
  auto x_mul_w = MakeNode("dot", "x_mul_w", {x, w});
  vector<NodeEntry> outputs = {x_mul_w};
  string graph_dump = GraphDump(outputs);
  string expected_graph_dump = R"x(digraph G {
  "x" -> "dot x_mul_w"
  "w" -> "dot x_mul_w"
})x";
  EXPECT_EQ(graph_dump, expected_graph_dump);
}
