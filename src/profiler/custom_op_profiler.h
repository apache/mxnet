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
#ifndef MXNET_CUSTOM_OP_PROFILER_H_
#define MXNET_CUSTOM_OP_PROFILER_H_

#include <string>
#include <unordered_set>
#include <unordered_map>
#include <thread>
#include "./profiler.h"

namespace mxnet {
namespace profiler {

using Tid = std::thread::id;
using TaskPtr = std::unique_ptr<ProfileTask>;

class CustomOpProfiler {
public:
    static CustomOpProfiler* Get() {
        static CustomOpProfiler inst;
        return &inst;
    }

    void OnCustomBegin(const std::string& op_type) {
        const Tid tid = std::this_thread::get_id();
        const std::string task_name = op_type + "::pure_python" ;
        std::lock_guard<std::mutex> lock(mutex_);
        tid_to_op_type_[tid] = op_type;
        tasks_[tid] = std::make_unique<ProfileTask>(task_name.c_str(), &custom_op_domain);
        tasks_[tid]->start();
    }

    void OnCustomEnd() {
        const Tid tid = std::this_thread::get_id();
        std::lock_guard<std::mutex> lock(mutex_);
        CHECK(tasks_.find(tid) != tasks_.end());
        tasks_[tid]->stop();
        tasks_.erase(tid);
        tid_to_op_type_.erase(tid);
    }


    const char* GenerateDisplayName(const char* op_type_ptr) {
        if (!op_type_ptr) {
            return NULL;
        }
        Tid tid = std::this_thread::get_id(); 
        std::lock_guard<std::mutex> lock(mutex_);
        if (tid_to_op_type_.find(tid) == tid_to_op_type_.end()) {
            return NULL;
        }
        std::string op_type = std::string(op_type_ptr);
        std::string name = tid_to_op_type_[tid] + "::" + op_type;
        display_names_.insert(name);
        return display_names_.find(name)->c_str();
    }

protected:
    CustomOpProfiler(){};

private:
    /*! \brief */
    std::mutex mutex_;
    /* !\brief task names for sub-operators in custom ops */
    std::unordered_set<std::string> display_names_;
    /* */
    std::unordered_map<Tid, TaskPtr> tasks_;
    /* */
    std::unordered_map<Tid, std::string> tid_to_op_type_;
};	
}  // namespace profiler
}  // namespace mxnet

#endif  // MXNET_CUSTOM_OP_PROFILER_H_ 