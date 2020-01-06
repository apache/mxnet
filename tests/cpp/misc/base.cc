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
#include <mxnet/base.h>

using namespace mxnet;
using namespace std;

/*
 * Test that different Context have different hash values
 */
TEST(ContextHashTest, ContextHashUnique) {
    set<size_t> hashes;
    size_t collision_count = 0;
    size_t total = 0;
    for (size_t dev_type = 0; dev_type < 32; ++dev_type) {
        for (size_t dev_id = 0; dev_id < 64; ++dev_id) {
            auto ctx = Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id);
            size_t res = std::hash<Context>()(ctx);
            auto insert_res = hashes.insert(res);
            if (!insert_res.second)
                ++collision_count;
            ++total;
        }
    }
    double collision = collision_count / static_cast<double>(total);
    cout << "mxnet::Context std::hash collision ratio: " << collision << endl;
    EXPECT_LE(collision, 0.04);
}
