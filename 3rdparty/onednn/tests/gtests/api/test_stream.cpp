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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.h"

#include <tuple>

namespace dnnl {

static bool are_valid_flags(
        dnnl_engine_kind_t engine_kind, dnnl_stream_flags_t stream_flags) {
    bool ok = true;
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (engine_kind == dnnl_gpu && (stream_flags & dnnl_stream_out_of_order))
        ok = false;
#endif
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    if (engine_kind == dnnl_cpu && (stream_flags & dnnl_stream_out_of_order))
        ok = false;
#endif
    return ok;
}

class stream_test_c
    : public ::testing::TestWithParam<
              std::tuple<dnnl_engine_kind_t, dnnl_stream_flags_t>> {
protected:
    void SetUp() override {
        std::tie(eng_kind, stream_flags) = GetParam();

        if (dnnl_engine_get_count(eng_kind) == 0) return;

        DNNL_CHECK(dnnl_engine_create(&engine, eng_kind, 0));

        // Check that the flags are compatible with the engine
        if (!are_valid_flags(eng_kind, stream_flags)) {
            DNNL_CHECK(dnnl_engine_destroy(engine));
            engine = nullptr;
            return;
        }

        DNNL_CHECK(dnnl_stream_create(&stream, engine, stream_flags));
    }

    void TearDown() override {
        if (stream) { DNNL_CHECK(dnnl_stream_destroy(stream)); }
        if (engine) { DNNL_CHECK(dnnl_engine_destroy(engine)); }
    }

    dnnl_engine_kind_t eng_kind;
    dnnl_stream_flags_t stream_flags;

    dnnl_engine_t engine = nullptr;
    dnnl_stream_t stream = nullptr;
};

class stream_test_cpp
    : public ::testing::TestWithParam<
              std::tuple<dnnl_engine_kind_t, dnnl_stream_flags_t>> {};

TEST_P(stream_test_c, Create) {
    SKIP_IF(!engine, "Engines not found or stream flags are incompatible.");

    DNNL_CHECK(dnnl_stream_wait(stream));
}

TEST(stream_test_c, WaitNullStream) {
    dnnl_stream_t stream = nullptr;
    dnnl_status_t status = dnnl_stream_wait(stream);
    ASSERT_EQ(status, dnnl_invalid_arguments);
}

TEST(stream_test_c, Wait) {
    dnnl_engine_t engine;
    DNNL_CHECK(dnnl_engine_create(&engine, dnnl_cpu, 0));

    dnnl_stream_t stream;
    DNNL_CHECK(dnnl_stream_create(&stream, engine, dnnl_stream_default_flags));

    DNNL_CHECK(dnnl_stream_wait(stream));

    DNNL_CHECK(dnnl_stream_destroy(stream));
    DNNL_CHECK(dnnl_engine_destroy(engine));
}

TEST_P(stream_test_cpp, Wait) {
    dnnl_engine_kind_t eng_kind_c = dnnl_cpu;
    dnnl_stream_flags_t stream_flags_c = dnnl_stream_in_order;
    std::tie(eng_kind_c, stream_flags_c) = GetParam();

    engine::kind eng_kind = static_cast<engine::kind>(eng_kind_c);
    stream::flags stream_flags = static_cast<stream::flags>(stream_flags_c);
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engines not found.");

    engine eng(eng_kind, 0);
    SKIP_IF(!are_valid_flags(static_cast<dnnl_engine_kind_t>(eng.get_kind()),
                    stream_flags_c),
            "Incompatible stream flags.");

    stream s(eng, stream_flags);
    engine s_eng = s.get_engine();
    s.wait();
}

TEST(stream_test_c, GetStream) {
    dnnl_engine_t engine;
    DNNL_CHECK(dnnl_engine_create(&engine, dnnl_cpu, 0));

    dnnl_stream_t stream;
    DNNL_CHECK(dnnl_stream_create(&stream, engine, dnnl_stream_default_flags));

    dnnl_engine_t stream_engine;
    DNNL_CHECK(dnnl_stream_get_engine(stream, &stream_engine));
    ASSERT_EQ(engine, stream_engine);

    DNNL_CHECK(dnnl_stream_destroy(stream));
    DNNL_CHECK(dnnl_engine_destroy(engine));
}

namespace {
struct PrintToStringParamName {
    template <class ParamType>
    std::string operator()(
            const ::testing::TestParamInfo<ParamType> &info) const {
        return to_string(std::get<0>(info.param)) + "_"
                + to_string(std::get<1>(info.param));
    }
};

auto all_params = ::testing::Combine(::testing::Values(dnnl_cpu, dnnl_gpu),
        ::testing::Values(dnnl_stream_in_order, dnnl_stream_out_of_order));

} // namespace

INSTANTIATE_TEST_SUITE_P(
        AllEngineKinds, stream_test_c, all_params, PrintToStringParamName());
INSTANTIATE_TEST_SUITE_P(
        AllEngineKinds, stream_test_cpp, all_params, PrintToStringParamName());

} // namespace dnnl
