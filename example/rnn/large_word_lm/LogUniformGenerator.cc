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
 * \file LogUniformGenerator.cc
 * \brief log uniform distribution generator
*/

#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <stddef.h>
#include <iostream>

#include "LogUniformGenerator.h"

LogUniformGenerator::LogUniformGenerator(const int range_max)
  : range_max_(range_max), log_range_max_(log(range_max)),
    generator_(), distribution_(0.0, 1.0) {}

std::unordered_set<long> LogUniformGenerator::draw(const size_t size, int* num_tries) {
  std::unordered_set<long> result;
  int tries = 0;
  while (result.size() != size) {
    tries += 1;
    double x = distribution_(generator_);
    long value = lround(exp(x * log_range_max_)) - 1;
    // sampling without replacement
    if (result.find(value) == result.end()) {
      result.emplace(value);
    }
  }
  *num_tries = tries;
  return result;
}
