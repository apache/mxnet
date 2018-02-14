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
 *  \file memory_test.cc
 *  \brief Perf/profile run of ActivationOp
 *  \author Chris Olivier
 */

#include <gtest/gtest.h>
#include <dmlc/omp.h>
#include <mxnet/tensor_blob.h>
#include "../include/test_util.h"
#include "../include/test_perf.h"

using namespace mxnet;

#ifdef _OPENMP
template<typename Container>
static typename Container::value_type average(const Container& cont) {
  typename Container::value_type avg = 0;
  const size_t sz = cont.size();
  for (auto iter = cont.begin(), e_iter = cont.end(); iter != e_iter; ++iter) {
    avg += *iter / sz;  // Attempt to not overflow by dividing up incrementally
  }
  return avg;
}

static int GetOMPThreadCount() {
  return omp_get_max_threads() >> 1;
}

/*!
 * \brief Generic bidirectional sanity test
 */
TEST(MEMORY_TEST, MemsetAndMemcopyPerformance) {
  const size_t GB = 1000000000;  // memset never slower
  uint64_t base = 100000;
  std::list<uint64_t> memset_times, omp_set_times, memcpy_times, omp_copy_times;
  size_t pass = 0;
  do {
    memset_times.resize(0);
    omp_set_times.resize(0);
    memcpy_times.resize(0);
    omp_copy_times.resize(0);;

    const size_t test_size = 2 * base;
    std::cout << "====================================" << std::endl
              << "Data size: " << test::pretty_num(test_size) << std::endl << std::flush;

    std::unique_ptr<float[]> buffer_1(new float[test_size]), buffer_2(new float[test_size]);
    float *src = buffer_1.get(), *dest = buffer_2.get();

    for (size_t x = 0; x < 5; ++x) {
      // Init memory with different values
      memset(src, 3, test_size * sizeof(float));
      memset(dest, 255, test_size * sizeof(float));  // wipe out some/all of src cache

      // memset
      uint64_t start = mxnet::test::perf::getNannoTickCount();
      memset(src, 0, test_size * sizeof(float));
      const uint64_t memset_time = mxnet::test::perf::getNannoTickCount() - start;

      start = mxnet::test::perf::getNannoTickCount();
      #pragma omp parallel for num_threads(GetOMPThreadCount())
      for (int i = 0; i < static_cast<int>(test_size); ++i) {
        src[i] = 42.0f;
      }
      const uint64_t omp_set_time = mxnet::test::perf::getNannoTickCount() - start;

      start = mxnet::test::perf::getNannoTickCount();
      memcpy(dest, src, test_size * sizeof(float));
      const uint64_t memcpy_time = mxnet::test::perf::getNannoTickCount() - start;

      // bounce the cache and dirty logic
      memset(src, 6, test_size * sizeof(float));
      memset(dest, 200, test_size * sizeof(float));

      start = mxnet::test::perf::getNannoTickCount();
      #pragma omp parallel for num_threads(GetOMPThreadCount())
      for (int i = 0; i < static_cast<int>(test_size); ++i) {
        dest[i] = src[i];
      }
      const uint64_t omp_copy_time = mxnet::test::perf::getNannoTickCount() - start;

      memset_times.push_back(memset_time);
      omp_set_times.push_back(omp_set_time);
      memcpy_times.push_back(memcpy_time);
      omp_copy_times.push_back(omp_copy_time);

      std::cout << "memset time:   " << test::pretty_num(memcpy_time) << " ns" << std::endl
                << "omp set time:  " << test::pretty_num(omp_set_time) << " ns" << std::endl
                << std::endl;
      std::cout << "memcpy time:   " << test::pretty_num(memcpy_time) << " ns" << std::endl
                << "omp copy time: " << test::pretty_num(omp_copy_time) << " ns" << std::endl
                << std::endl;
    }
    std::cout << "------------------------------------" << std::endl;
    if (average(memset_times) > average(omp_set_times)) {
      std::cout << "<< MEMSET SLOWER FOR " << test::pretty_num(test_size)
                << " items >>" << std::endl;
    }
    if (average(memcpy_times) > average(omp_copy_times)) {
      std::cout << "<< MEMCPY SLOWER FOR " << test::pretty_num(test_size)
                << " items >>" << std::endl;
    }
    if (!pass) {
      // Skipping assertions due to flaky timing.
      // Tracked in Issue: https://github.com/apache/incubator-mxnet/issues/9649
    if (average(memset_times) < average(omp_set_times)
        || average(memcpy_times) < average(omp_copy_times)) {
        std::cout << "Warning: Skipping assertion failures, see issue 9649" <<std::endl;
      }
    }
    base *= 10;
    ++pass;
  } while (test::performance_run
           && base <= GB
           && (average(memset_times) < average(omp_set_times)
               || average(memcpy_times), average(omp_copy_times)));
}
#endif  // _OPENMP
