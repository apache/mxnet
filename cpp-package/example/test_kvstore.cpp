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
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

static bool test_single_key() {
  std::string key = "singlekeytest";

  NDArray result(Shape(4), Context::cpu());

  // initialize data
  NDArray data({0.f, 233.f, -0.12f, 9.f}, Shape(4), Context::cpu());
  KVStore::Init(key, data);
  NDArray::WaitAll();

  // retrieve result
  KVStore::Pull(key, &result);
  NDArray::WaitAll();

  // compare
  for (size_t j=0; j < result.Size(); j++) {
    if (result.GetData()[j] != data.GetData()[j]) {
      LG << "Error: wrong initialized data in singlekeytest, expect "
          << data.GetData()[j] << " got " << result.GetData()[j];
      return false;
    }
  }

  // push gradient
  NDArray grad({0.1f, -2.f, -4.4f, 0.f}, Shape(4), Context::cpu());
  KVStore::Push(key, grad);
  NDArray::WaitAll();

  // retrieve result
  KVStore::Pull(key, &result);
  NDArray::WaitAll();

  // compare
  for (size_t j=0; j < result.Size(); j++) {
    if (result.GetData()[j] != grad.GetData()[j]) {
      LG << "Error: wrong gradient data in singlekeytest, expect "
          << grad.GetData()[j] << " got " << result.GetData()[j];
      return false;
    }
  }

  return true;
}

static bool test_multiple_key() {
  std::vector<std::string> keys(2);
  keys[0] = "multikeytest-0";
  keys[1] = "multikeytest-1";

  std::vector<NDArray> results(2);
  results[0] = NDArray(Shape(4), Context::cpu());
  results[1] = NDArray(Shape(4), Context::cpu());

  // initialize data
  std::vector<NDArray> data(2);
  data[0] = NDArray({0.f, 2.f, -3.12f, 4.f}, Shape(4), Context::cpu());
  data[1] = NDArray({0.8f, -2.f, 6.6f, 77.f}, Shape(4), Context::cpu());
  KVStore::Init(keys, data);
  NDArray::WaitAll();

  // retrieve result
  KVStore::Pull(keys, &results);
  NDArray::WaitAll();

  // compare
  for (size_t i=0; i < results.size(); i++) {
    for (size_t j=0; j < results[i].Size(); j++) {
      if (results[i].GetData()[j] != data[i].GetData()[j]) {
        LG << "Error: wrong initialized data in multikeytest, expect "
            << data[i].GetData()[j] << " got " << results[i].GetData()[j];
        return false;
      }
    }
  }

  // push gradient, reduce for the second
  std::vector<std::string> push_keys(3);
  push_keys[0] = "multikeytest-0";
  push_keys[1] = "multikeytest-1";
  push_keys[2] = "multikeytest-1";

  std::vector<NDArray> grads(3);
  grads[0] = NDArray({0.2f, -0.3f, -1.1f, 0.0f}, Shape(4), Context::cpu());
  grads[1] = NDArray({2.f, 4.f, -4.f, -5.f}, Shape(4), Context::cpu());
  grads[2] = NDArray({-3.f, -0.2f, 12.f, -9.f}, Shape(4), Context::cpu());
  KVStore::Push(push_keys, grads);
  NDArray::WaitAll();

  // retrieve result
  KVStore::Pull(keys, &results);
  NDArray::WaitAll();

  // compare the first
  for (size_t j=0; j < results[0].Size(); j++) {
    if (results[0].GetData()[j] != grads[0].GetData()[j]) {
      LG << "Error: wrong gradient data in multikeytest, expect " << grads[0].GetData()[j]
          << " got " << results[0].GetData()[j];
      return false;
    }
  }

  // compare the second
  for (size_t j=0; j < results[1].Size(); j++) {
    if (results[1].GetData()[j] != (grads[1].GetData()[j] + grads[2].GetData()[j])) {
      LG << "Error: wrong reduced gradient data in multikeytest, expect "
          << (grads[1].GetData()[j] + grads[2].GetData()[j])
          << " got " << results[1].GetData()[j];
      return false;
    }
  }

  return true;
}

int main(int argc, char** argv) {
  KVStore::SetType("local");

  bool success1 = test_single_key();
  bool success2 = test_multiple_key();

  int ret = (success1 && success2) ? 1 : 0;

  MXNotifyShutdown();
  return ret;
}
