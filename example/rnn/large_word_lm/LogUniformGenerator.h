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
 * \file LogUniformGenerator.h
 * \brief log uniform distribution generator
*/

#ifndef _LOG_UNIFORM_GENERATOR_H
#define _LOG_UNIFORM_GENERATOR_H

#include <unordered_set>
#include <utility>
#include <random>

class LogUniformGenerator {
private:
  const int range_max_;
  const double log_range_max_;
  std::default_random_engine generator_;
  std::uniform_real_distribution<double> distribution_;
public:
  LogUniformGenerator(const int);
  std::unordered_set<long> draw(const size_t, int*);
};

#endif // _LOG_UNIFORM_GENERATOR_H

