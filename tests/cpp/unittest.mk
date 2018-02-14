# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

TEST_SRC = $(shell find tests/cpp/ -name "*.cc")
TEST_OBJ = $(patsubst %.cc, build/%.o, $(TEST_SRC))
TEST = build/tests/cpp/mxnet_unit_tests

GTEST_DIR=3rdparty/googletest/googletest/
GTEST_INC=3rdparty/googletest/googletest/include/
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

TEST_CFLAGS = -Itests/cpp/include -Isrc $(CFLAGS)
TEST_LDFLAGS = $(LDFLAGS) -Llib -lmxnet

ifeq ($(USE_BREAKPAD), 1)
TEST_CFLAGS  += -I/usr/local/include/breakpad
TEST_LDFLAGS += -lbreakpad_client -lbreakpad
endif

.PHONY: runtest testclean

gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_INC) -I$(GTEST_DIR) $(CXXFLAGS) -c $(GTEST_DIR)/src/gtest-all.cc

gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

build/tests/cpp/%.o : tests/cpp/%.cc
	@mkdir -p $(@D)
	$(CXX) -std=c++11 $(TEST_CFLAGS) -I$(GTEST_INC) -MM -MT tests/cpp/$* $< > build/tests/cpp/$*.d
	$(CXX) -c -std=c++11 $(TEST_CFLAGS) -I$(GTEST_INC) -o build/tests/cpp/$*.o $(filter %.cc %.a, $^)

build/tests/cpp/operator/%.o : tests/cpp/operator/%.cc
	@mkdir -p $(@D)
	$(CXX) -std=c++11 $(TEST_CFLAGS) -I$(GTEST_INC) -MM -MT tests/cpp/operator/$* $< > build/tests/cpp/operator/$*.d
	$(CXX) -c -std=c++11 $(TEST_CFLAGS) -I$(GTEST_INC) -o build/tests/cpp/operator/$*.o $(filter %.cc %.a, $^)

build/tests/cpp/storage/%.o : tests/cpp/storage/%.cc
	@mkdir -p $(@D)
	$(CXX) -std=c++11 $(TEST_CFLAGS) -I$(GTEST_INC) -MM -MT tests/cpp/storage/$* $< > build/tests/cpp/storage/$*.d
	$(CXX) -c -std=c++11 $(TEST_CFLAGS) -I$(GTEST_INC) -o build/tests/cpp/storage/$*.o $(filter %.cc %.a, $^)

build/tests/cpp/engine/%.o : tests/cpp/engine/%.cc
	@mkdir -p $(@D)
	$(CXX) -std=c++11 $(TEST_CFLAGS) -I$(GTEST_INC) -MM -MT tests/cpp/engine/$* $< > build/tests/cpp/engine/$*.d
	$(CXX) -c -std=c++11 $(TEST_CFLAGS) -I$(GTEST_INC) -o build/tests/cpp/engine/$*.o $(filter %.cc %.a, $^)

$(TEST): $(TEST_OBJ) lib/libmxnet.so gtest.a
	$(CXX) -std=c++11 $(TEST_CFLAGS) -I$(GTEST_INC) -o $@ $^ $(TEST_LDFLAGS)

runtest: $(TEST)
	LD_LIBRARY_PATH=$(shell pwd)/lib:$(LD_LIBRARY_PATH) $(TEST)

testclean:
	rm -f $(TEST) $(TEST_OBJ)

-include build/tests/cpp/*.d
-include build/tests/cpp/operator/*.d
-include build/tests/cpp/storage/*.d
-include build/tests/cpp/engine/*.d
