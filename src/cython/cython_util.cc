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
#include <iostream>
#include <cstdarg>
#include <sys/time.h>
#include <chrono>
#include "./cython_util.h"

extern "C" int CythonPrintFromCPP(const char *foo) {
  if(foo) {
    std::cout << foo << std::endl << std::flush;
  }
  return 0;
}

extern "C" int Printf(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  const int res = vprintf(fmt, args);
  va_end(args);
  return res;
}

extern "C" int TrivialCPPCall(int var) {
  static int static_var = 0;
  static_var = var;
  return static_var;
}

using Tick = std::chrono::high_resolution_clock::time_point;
static inline Tick Now() { return std::chrono::high_resolution_clock::now(); }

static const Tick _app_start_time = Now();

static inline uint64_t GetDurationInNanoseconds(const Tick &t1, const Tick &t2) {
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
}

static inline uint64_t GetDurationInNanoseconds(const Tick &since) {
  return GetDurationInNanoseconds(since, Now());
}

constexpr size_t SLEEP_DURATION = 500;
constexpr size_t TIMER_PERIOD = 10;  // Ideal is 50 periods occur
constexpr size_t MIN_COUNT_WHILE_SLEEPING = 10;
constexpr size_t MAX_COUNT_WHILE_SLEEPING = 150;

static inline size_t GetDurationInMilliseconds(const Tick& start_time) {
  return static_cast<size_t>(GetDurationInNanoseconds(start_time)/1000/1000);
}


extern "C" uint64_t TimeInMilliseconds() {
  return GetDurationInMilliseconds(_app_start_time);
}

namespace shapes {

Rectangle::Rectangle() { }

Rectangle::Rectangle(int X0, int Y0, int X1, int Y1) {
  x0 = X0;
  y0 = Y0;
  x1 = X1;
  y1 = Y1;
}

Rectangle::~Rectangle() { }

int Rectangle::getArea() {
  return (x1 - x0) * (y1 - y0);
}

void Rectangle::getSize(int *width, int *height) {
  (*width) = x1 - x0;
  (*height) = y1 - y0;
}

void Rectangle::move(int dx, int dy) {
  x0 += dx;
  y0 += dy;
  x1 += dx;
  y1 += dy;
}

}  // namespace shapes
