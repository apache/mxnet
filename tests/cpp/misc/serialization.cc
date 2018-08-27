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
#include <../../../src/common/serialization.h>

using namespace mxnet;
using namespace std;

/*
 * Test that used datastruct are properly serialized and deserialized
 */

TEST(SerializerTest, InputMapCorrect) {
    std::map<std::string, int32_t> input_map;
    input_map.emplace("input_0", 2);
    input_map.emplace("another_input", 0);
    input_map.emplace("last_input", 1);
    std::string serialized_data;
    common::Serialize(input_map, &serialized_data);
    std::map<std::string, int32_t> deserialized_input_map;
    common::Deserialize(&deserialized_input_map, serialized_data);
    ASSERT_EQ(input_map.size(), deserialized_input_map.size());
    for (auto& p : input_map) {
        auto it = deserialized_input_map.find(p.first);
        ASSERT_NE(it, deserialized_input_map.end());
        ASSERT_EQ(it->second, p.second);
    }
}

TEST(SerializerTest, OutputMapCorrect) {
    std::map<std::string, std::tuple<uint32_t, TShape, int, int> > output_map;
    output_map.emplace("output_0", std::make_tuple(1, TShape({23, 12, 63, 432}), 0, 1));
    output_map.emplace("another_output", std::make_tuple(2, TShape({23, 123}), 14, -23));
    output_map.emplace("last_output", std::make_tuple(0, TShape({0}), -1, 0));
    std::string serialized_data;
    common::Serialize(output_map, &serialized_data);
    std::map<std::string, std::tuple<uint32_t, TShape, int, int> > deserialized_output_map;
    common::Deserialize(&deserialized_output_map, serialized_data);
    ASSERT_EQ(output_map.size(), deserialized_output_map.size());
    for (auto& p : output_map) {
        auto it = deserialized_output_map.find(p.first);
        ASSERT_NE(it, deserialized_output_map.end());
        auto lhs = it->second;
        auto rhs = p.second;
        ASSERT_EQ(std::get<0>(lhs), std::get<0>(rhs));
        ASSERT_EQ(std::get<1>(lhs), std::get<1>(rhs));
        ASSERT_EQ(std::get<2>(lhs), std::get<2>(rhs));
        ASSERT_EQ(std::get<3>(lhs), std::get<3>(rhs));
    }
}

