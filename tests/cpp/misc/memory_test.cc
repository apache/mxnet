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
#include <dmlc/logging.h>
#include <dmlc/omp.h>
#include <mxnet/tensor_blob.h>
#include "../include/test_perf.h"

using namespace mxnet;
#ifdef _OPENMP

template<typename Container>
inline typename Container::value_type average(const Container& cont) {
  typename Container::value_type avg = 0;
  const size_t sz = cont.size();
  for(auto iter = cont.begin(), e_iter = cont.end(); iter != e_iter; ++iter) {
    avg += *iter / sz;  // Attempt to not overflow by dividing up incrementally
  }
  return avg;
}

/*!
 * \brief Generic bidirectional sanity test
 */
TEST(MEMORY_TEST, MemsetAndMemcopyPerformance) {
  //const size_t GB = 1000000000;  // memset sometimes slower
  const size_t GB = 100000000;  // memset never slower
  const size_t test_size = 2 * GB;
  std::cout << "Data size: " << test_size << std::endl << std::flush;

  std::list<uint64_t> memset_times, omp_set_times, memcpy_times, omp_copy_times;
  std::unique_ptr<uint8_t> buffer_1(new uint8_t[test_size]), buffer_2(new uint8_t[test_size]);
  uint8_t *src = buffer_1.get(), *dest = buffer_2.get();

  for(size_t x = 0; x < 10; ++x) {
    uint64_t start = test::perf::getNannoTickCount();
    memset(src, 123, test_size);
    const uint64_t memset_time = test::perf::getNannoTickCount() - start;

    start = test::perf::getNannoTickCount();
    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (int i = 0; i < test_size; ++i) {
      src[i] = 123;
    }
    const uint64_t omp_set_time = test::perf::getNannoTickCount() - start;

    start = test::perf::getNannoTickCount();
    memcpy(dest, src, test_size);
    const uint64_t memcpy_time = test::perf::getNannoTickCount() - start;

    start = test::perf::getNannoTickCount();
    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (int i = 0; i < test_size; ++i) {
      dest[i] = src[i];
    }
    const uint64_t omp_copy_time = test::perf::getNannoTickCount() - start;

    memset_times.push_back(memset_time);
    omp_set_times.push_back(omp_set_time);
    memcpy_times.push_back(memcpy_time);
    omp_copy_times.push_back(omp_copy_time);

    std::cout << "memset: " << memcpy_time
              << " ns, omp set time:  " << omp_set_time << " ns" << std::endl;
    std::cout << "memcpy: " << memcpy_time
              << " ns, omp copy time: " << omp_copy_time << " ns" << std::endl;
  }

  ASSERT_LE(average(memset_times), average(omp_set_times));
  ASSERT_LE(average(memcpy_times), average(omp_copy_times));

}
#endif  // _OPENMP