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

#include <gtest/gtest.h>

#include "../include/test_util.h"
#include "../../src/engine/openmp.h"

#if defined(unix) || defined(__unix__) || defined(__unix)
#include <unistd.h>
#include <sys/types.h>
#include <dmlc/logging.h>


TEST(OMPBehaviour, after_fork) {
    /* 
     * Check that after fork, OMP is disabled, and the recommended thread count is 1 to prevent 
     * process fanout.
     */
    using namespace mxnet::engine;
    auto openmp = OpenMP::Get();
    pid_t pid = fork();
    if (pid == 0) {
        EXPECT_FALSE(openmp->enabled());
        EXPECT_EQ(openmp->GetRecommendedOMPThreadCount(), 1);
    } else if (pid > 0) {
        int status;
        int ret = waitpid(pid, &status, 0);
        CHECK_EQ(ret, pid) << "waitpid failed";
    } else {
        CHECK(false) << "fork failed";
    }
}
#endif
