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
#include <dmlc/omp.h>
#include <dmlc/base.h>
#include <dmlc/parameter.h>
#include <climits>
#include "./openmp.h"

namespace mxnet {
namespace engine {

#if defined(__i386__) || defined(_M_X86) || defined(_M_X64) || defined(__x86_64__)
#define ARCH_IS_INTEL_X86
#endif

static inline bool is_env_set(const char *var) {
  return dmlc::GetEnv(var, INT_MIN) != INT_MIN;
}

OpenMP *OpenMP::Get() {
  static OpenMP openMP;
  return &openMP;
}

OpenMP::OpenMP()
  : omp_num_threads_set_in_environment(is_env_set("OMP_NUM_THREADS")) {
#ifdef _OPENMP
  const int max = dmlc::GetEnv("MXNET_OMP_MAX_THREADS", INT_MIN);
  if (max != INT_MIN) {
    omp_thread_max_ = max;
  } else {
    if (!omp_num_threads_set_in_environment) {
      omp_thread_max_ = omp_get_num_procs();
#ifdef ARCH_IS_INTEL_X86
      omp_thread_max_ >>= 1;
#endif
      omp_set_num_threads(omp_thread_max_);
    } else {
      omp_thread_max_ = omp_get_max_threads();
    }
  }
#else
  enabled_ = false;
  omp_thread_max_ = 1;
#endif
}

void OpenMP::set_reserve_cores(int cores) {
  CHECK_GE(cores, 0);
  reserve_cores_ = cores;
#ifdef _OPENMP
  if (reserve_cores_ >= omp_thread_max_) {
    omp_set_num_threads(1);
  } else {
    omp_set_num_threads(omp_thread_max_ - reserve_cores_);
  }
#endif
}

int OpenMP::GetRecommendedOMPThreadCount(bool exclude_reserved) const {
#ifdef _OPENMP
  if (omp_num_threads_set_in_environment) {
    return omp_get_max_threads();
  }
  if (enabled_) {
    int thread_count = omp_get_max_threads();
    if (exclude_reserved) {
      if (reserve_cores_ >= thread_count) {
        thread_count = 1;
      } else {
        thread_count -= reserve_cores_;
      }
    }
    // Check that OMP doesn't suggest more than our 'omp_thread_max_' value
    if (!omp_thread_max_ || thread_count < omp_thread_max_) {
      return thread_count;
    }
    return omp_thread_max_;
  }
  return 1;
#else
  return 1;
#endif
}

OpenMP *__init_omp__ = OpenMP::Get();

}  // namespace engine
}  // namespace mxnet

