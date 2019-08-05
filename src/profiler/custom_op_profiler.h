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
#ifndef MXNET_PROFILER_CUSTOM_OP_PROFILER_H_
#define MXNET_PROFILER_CUSTOM_OP_PROFILER_H_

#include <string>
#include <unordered_set>
#include <unordered_map>
#include <thread>
#include "./profiler.h"

namespace mxnet {
namespace profiler {

using Tid = std::thread::id;
using TaskPtr = std::unique_ptr<ProfileTask>;

  /*!
   * \brief Singleton class to assist profiling python callback of custom operators
   * and to assist linking sub-operators to custom operators
   */
class CustomOpProfiler {
 public:
  static CustomOpProfiler* Get() {
    static std::mutex mtx;
    static std::unique_ptr<CustomOpProfiler> prof = nullptr;
    if (!prof) {
      std::unique_lock<std::mutex> lk(mtx);
      if (!prof)
        prof = std::make_unique<CustomOpProfiler>();
    }
    return prof.get();
  }
  /*!
   * \brief Called before the callback of custom operators to start a profile task for python 
   * code execution time
   * \param op_type The registed name of the custom operator
   */
  void OnCustomBegin(const std::string& op_type) {
    const Tid tid = std::this_thread::get_id();
    const std::string task_name = MakePythonCodeName(op_type);
    std::lock_guard<std::mutex> lock(mutex_);
    tid_to_op_type_[tid] = op_type;
    tasks_[tid] = std::make_unique<ProfileTask>(task_name.c_str(), &custom_op_domain);
    tasks_[tid]->start();
  }

  /*!
   * \brief Called after the callback of custom operators to stop the profile task for python 
   * code execution time
   */
  void OnCustomEnd() {
    const Tid tid = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(mutex_);
    tid_to_op_type_.erase(tid);
    // this should never fail
    CHECK(tasks_.find(tid) != tasks_.end()) << "thread_id not found. " <<
        "Please use OnCustomBegin() and OnCustomEnd() in pairs.";
    tasks_[tid]->stop();
    tasks_.erase(tid);
  }

  /*!
   * \brief Generate a display name for sub-operators, which is the name used for OprBlock
   * and later by profiler, and store it in a unordered_set so that it can be referenced 
   * in the future.
   * Notice if the operator is not a sub-operator, just return the char pointer back.
   * \param op_type The registed name of the operator
   * \return Returns a pointer to the display name generated
   */
  const char* GenerateDisplayName(const char* op_type) {
    if (!op_type) {
      return nullptr;
    }
    Tid tid = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(mutex_);
    if (tid_to_op_type_.find(tid) == tid_to_op_type_.end()) {
      return op_type;
    }
    std::string name = MakeSubOperatorName(tid, op_type);
    return display_names_.insert(name).first->c_str();
  }

 private:
  /* !\brief make the display name for sub-operators */
  inline std::string MakeSubOperatorName(const Tid& tid, const char* op_type) {
    return tid_to_op_type_[tid] + "::" + std::string(op_type);
  }
  /* !\brief make the display name for the pure python call back function i.e.
   * forward() or backward() in the custom operator definition
   */
  inline std::string MakePythonCodeName(const std::string& op_type) {
    return op_type + "::pure_python";
  }
  /*! \brief class mutex */
  std::mutex mutex_;
  /* !\brief display names for sub-operators in custom ops */
  std::unordered_set<std::string> display_names_;
  /* !\brief profiling tasks for pure python code in custom operators */
  std::unordered_map<Tid, TaskPtr> tasks_;
  /* !\brief the maping from thread id to the registered name op the custom operator
   * that is runnin on that thread
   */
  std::unordered_map<Tid, std::string> tid_to_op_type_;
};
}  // namespace profiler
}  // namespace mxnet

#endif  // MXNET_PROFILER_CUSTOM_OP_PROFILER_H_
