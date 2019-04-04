
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
#include "mshadow/base.h"

using namespace std;
using namespace mshadow;

/*
 * Test that enum and string values are in sync
 */
TEST(Overflow, OverflowTest) {
  EXPECT_TRUE(mult_not_overflow_binary<int>(200,400));
  EXPECT_FALSE(mult_not_overflow_binary<int>(1<<31,4));
  EXPECT_FALSE(mult_not_overflow_binary<int>(1<<30,1<<5));
  EXPECT_TRUE(mult_not_overflow<int>(2, 200,400));
  EXPECT_FALSE(mult_not_overflow<int>(2, 1<<31,1<<31));
  EXPECT_TRUE(mult_not_overflow<int>(2, 1<<31,1));
  EXPECT_FALSE(mult_not_overflow<int>(3, 1<<31,1,2));
  EXPECT_TRUE(mult_not_overflow<int>(2, 0, 0));
  EXPECT_TRUE(mult_not_overflow<int>(1, 0));
}

