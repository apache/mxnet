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
#ifndef MXNET_ENGINE_OPENMP_H_
#define MXNET_ENGINE_OPENMP_H_

namespace mxnet {
namespace engine {

/*! \brief OpenMP wrapper and management class
 *         This class manages a layer on top of the OMP implementation and does not
 *         interact bidirectionally with the OMP implementation for all behaviors
 *         (i.e. it's meant to be use explicitly for explicit arguments to omp pragmas
 *         without affecting the behavior when no arguments are given)
 */
class OpenMP {
 public:
  OpenMP();

  /*!
   * \brief Get the recommended number of OMP threads to use given the current context
   * \return Recommended number of OMP threads to use in a parallel operation
   */
  int GetRecommendedOMPThreadCount(bool exclude_reserved = true) const;

  /*!
   * \brief Set whether clients of this class receive pro-OMP behavior guidance
   * \param enabled Set to 'true' if this class should provide OMP behavior
   */
  void set_enabled(bool enabled) { enabled_ = enabled; }
  bool enabled() const { return enabled_; }

  /*!
   * \brief Set maximum number of threads to be used in an OMP region
   * \param thread_max Maximum number of threads to be used in an OMP region
   */
  void set_thread_max(int thread_max) { omp_thread_max_ = thread_max; }
  /*!
   * \brief Maximum number of threads to be used in an OMP region
   * \return Maximum number of threads
   */
  int thread_max() const { return omp_thread_max_; }

  /*!
   * \brief Reserve cores to be excluded from OMP regions
   * \param cores Number of cores to be excluded from OMP region usage
   */
  void set_reserve_cores(int cores);
  /*!
   * \brief Get number of cores to be excluded from OMP regions
   * \return Number of cores to be excluded from OMP regions
   */
  int reserve_cores() const { return reserve_cores_; }

  /*!
   * \brief Call at the beginning of a worker thread's life.  This will set the omp_num_threads
   *        for omp regions created by this thread
   * \param use_omp true if this thread plans to utilize parallel omp regions
   */
  void on_start_worker_thread(bool use_omp);

  /*!
   * \brief Initialize a new process to use omp (after a fork,
   *        in case you're starting threads in the atfork() that may interfere
   *        with the initialization. Can serialize the init with this first.
   */
  void initialize_process();

  /*!
   * \brief Get the OpenMP object's singleton pointer
   * \return Singleton OpenMP object pointer
   */
  static OpenMP *Get();

 private:
  /*!
   * \brief Whether OpenMP layer is enabled (use more then one thread).  Independent of OMP library
   *        behavior
   */
  volatile bool enabled_ = true;
  /*!
   * \brief Maximum number of threads for any OMP region
   */
  volatile int omp_thread_max_ = 0;
  /*!
   * \brief Number of cores to reserve for non-OMP regions
   */
  volatile int reserve_cores_ = 0;
  /*!
   * \brief Whether OMP_NUM_THREADS was set in the environment.  If it is, we fall back to
   *        the OMP's implementation's handling of that environment variable
   */
  const bool omp_num_threads_set_in_environment_;
};

}  // namespace engine
}  // namespace mxnet

#endif  // MXNET_ENGINE_OPENMP_H_
