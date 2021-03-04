/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef DNNL_TEST_MACROS_HPP
#define DNNL_TEST_MACROS_HPP

#include <iostream>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#define TEST_CONCAT_(a, b) a##b
#define TEST_CONCAT(a, b) TEST_CONCAT_(a, b)

#define SKIP_IF(cond, msg) \
    do { \
        if (cond) { \
            std::cout << "[  SKIPPED ] " << (msg) << std::endl; \
            return; \
        } \
    } while (0)

#define TEST_F_(test_fixture, test_name) TEST_F(test_fixture, test_name)

#define CPU_TEST_F(test_fixture, test_name) \
    TEST_F_(test_fixture, TEST_CONCAT(test_name, _CPU))

#define GPU_TEST_F(test_fixture, test_name) \
    TEST_F_(test_fixture, TEST_CONCAT(test_name, _GPU))

#define TEST_P_(test_fixture, test_name) TEST_P(test_fixture, test_name)

#define CPU_TEST_P(test_fixture, test_name) \
    TEST_P_(test_fixture, TEST_CONCAT(test_name, _CPU))

#define GPU_TEST_P(test_fixture, test_name) \
    TEST_P_(test_fixture, TEST_CONCAT(test_name, _GPU))

#define INSTANTIATE_TEST_SUITE_P_(prefix, test_case_name, generator) \
    INSTANTIATE_TEST_SUITE_P(prefix, test_case_name, generator)

#define CPU_INSTANTIATE_TEST_SUITE_P(prefix, test_case_name, generator) \
    INSTANTIATE_TEST_SUITE_P_( \
            TEST_CONCAT(prefix, _CPU), test_case_name, generator)

#define GPU_INSTANTIATE_TEST_SUITE_P(prefix, test_case_name, generator) \
    INSTANTIATE_TEST_SUITE_P_( \
            TEST_CONCAT(prefix, _GPU), test_case_name, generator)

#define GPU_INSTANTIATE_TEST_SUITE_P_(prefix, test_case_name, generator) \
    GPU_INSTANTIATE_TEST_SUITE_P(prefix, test_case_name, generator)

#ifdef DNNL_ENABLE_MEM_DEBUG
#define DERIVED_TEST_CLASS(test_fixture, test_name) \
    test_fixture##_##test_name##_Derived_Test

#define HANDLE_EXCEPTIONS_FOR_TEST_SETUP(...) \
    void SetUp() override { \
        catch_expected_failures([=]() { Testing(); }, false, dnnl_success); \
    } \
    void Testing()

// Wrapper around TEST from gtest, intended to catch exceptions thrown by a unit
// test.
#define HANDLE_EXCEPTIONS_FOR_TEST(test_fixture, test_name) \
    class DERIVED_TEST_CLASS(test_fixture, test_name) : public test_fixture { \
        void TestBody() override {} \
\
    public: \
        void Test_failures(); \
    }; \
    TEST(test_fixture, test_name) { \
        catch_expected_failures( \
                [=]() { \
                    DERIVED_TEST_CLASS(test_fixture, test_name) \
                    ().Test_failures(); \
                }, \
                false, dnnl_success, false); \
    } \
    void DERIVED_TEST_CLASS(test_fixture, test_name)::Test_failures()

// Wrapper around TEST_F from gtest, intended to catch exceptions thrown by a
// test fixture.
#define HANDLE_EXCEPTIONS_FOR_TEST_F(test_fixture, test_name) \
    class DERIVED_TEST_CLASS(test_fixture, test_name) : public test_fixture { \
        void TestBody() override {} \
\
    public: \
        DERIVED_TEST_CLASS(test_fixture, test_name)() { SetUp(); } \
        void Test_failures(); \
    }; \
    TEST_F(test_fixture, test_name) { \
        catch_expected_failures( \
                [=]() { \
                    DERIVED_TEST_CLASS(test_fixture, test_name) \
                    ().Test_failures(); \
                }, \
                false, dnnl_success, false); \
    } \
    void DERIVED_TEST_CLASS(test_fixture, test_name)::Test_failures()

// Wrapper around TEST_P from gtest, intended to catch exceptions thrown by
// a parametrized test.
#define HANDLE_EXCEPTIONS_FOR_TEST_P(test_fixture, test_name) \
    class DERIVED_TEST_CLASS(test_fixture, test_name) : public test_fixture { \
        void TestBody() override {} \
\
    public: \
        DERIVED_TEST_CLASS(test_fixture, test_name)() { SetUp(); } \
        void Test_failures(); \
    }; \
    TEST_P(test_fixture, test_name) { \
        catch_expected_failures( \
                [=]() { \
                    DERIVED_TEST_CLASS(test_fixture, test_name) \
                    ().Test_failures(); \
                }, \
                false, dnnl_success); \
    } \
    void DERIVED_TEST_CLASS(test_fixture, test_name)::Test_failures()

#else
#define HANDLE_EXCEPTIONS_FOR_TEST_SETUP(...) void SetUp() override
#define HANDLE_EXCEPTIONS_FOR_TEST(test_fixture, test_name) \
    TEST(test_fixture, test_name)
#define HANDLE_EXCEPTIONS_FOR_TEST_F(test_fixture, test_name) \
    TEST_F(test_fixture, test_name)
#define HANDLE_EXCEPTIONS_FOR_TEST_P(test_fixture, test_name) \
    TEST_P(test_fixture, test_name)
#endif

#endif
