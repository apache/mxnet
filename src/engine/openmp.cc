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

OpenMP *OpenMP::Get() {
  static OpenMP openMP;
  return &openMP;
}

OpenMP::OpenMP()
  : omp_num_threads_set_in_environment(dmlc::GetEnv("OMP_NUM_THREADS", INT_MIN) == INT_MIN) {
#ifdef _OPENMP
  if (!omp_num_threads_set_in_environment) {
    omp_set_nested(true);
    omp_set_dynamic(false);
  }
  const int max = dmlc::GetEnv("MXNET_OMP_MAX_THREADS", INT_MIN);
  if (max != INT_MIN) {
    omp_thread_max_ = max;
  } else {
#ifdef ARCH_IS_INTEL_X86
    omp_thread_max_ = omp_get_num_procs() >> 1;
#endif
  }
#else
  enabled_ = false;
  omp_thread_max_ = 1;
#endif
}

int OpenMP::GetRecommendedOMPThreadCount() const {
#ifdef _OPENMP
  if (omp_num_threads_set_in_environment) {
    return omp_get_max_threads();
  }
  if (enabled_) {
#ifdef ARCH_IS_INTEL_X86
    // x86 does hyperthreading, but do to cache issues, it's faster to only use # true CPUs
    const int thread_count = omp_get_max_threads() >> 1;
#else
    const int thread_count = omp_get_max_threads();
#endif
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

}  // namespace engine
}  // namespace mxnet

