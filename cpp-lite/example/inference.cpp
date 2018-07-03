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
 */
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <cstdlib>

#include <mxnet-lite/symbol_block.h>


int main() {
  using namespace mxnet::lite;

  Symbol out = Symbol::Load("resnet-symbol.json");
  auto params = NDArray::LoadDict("resnet-0000.params");
  SymbolBlock block(Context::CPU(0), {}, out, {"data"}, params);

  NDArray x({1, 3, 32, 32}, Context::CPU(0), false);
  auto y = block.Forward({x});

  std::vector<float> y0(y[0].GetSize(), 0);
  y[0].CopyTo(y0.data(), y0.size());

  std::cout << "[";
  for (const auto& i : y0) std::cout << i << ", ";
  std::cout << "]" << std::endl;
  return 0;
}
