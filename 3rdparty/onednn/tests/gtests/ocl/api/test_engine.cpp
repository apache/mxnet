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

#include "oneapi/dnnl/dnnl_ocl.h"
#include "oneapi/dnnl/dnnl_ocl.hpp"

#include <string>
#include <CL/cl.h>

namespace dnnl {
namespace {

enum class dev_kind { null, cpu, gpu };
enum class ctx_kind { null, cpu, gpu };

} // namespace

struct ocl_engine_test_params {
    dev_kind adev_kind;
    ctx_kind actx_kind;
    dnnl_status_t expected_status;
};

class ocl_engine_test
    : public ::testing::TestWithParam<ocl_engine_test_params> {
protected:
    void SetUp() override {
        gpu_ocl_dev = find_ocl_device(CL_DEVICE_TYPE_GPU);
        cpu_ocl_dev = find_ocl_device(CL_DEVICE_TYPE_CPU);

        cl_int err;
        if (gpu_ocl_dev) {
            gpu_ocl_ctx = clCreateContext(
                    nullptr, 1, &gpu_ocl_dev, nullptr, nullptr, &err);
            TEST_OCL_CHECK(err);
        }

        if (cpu_ocl_dev) {
            cpu_ocl_ctx = clCreateContext(
                    nullptr, 1, &cpu_ocl_dev, nullptr, nullptr, &err);
            TEST_OCL_CHECK(err);
        }
    }

    void TearDown() override {
        if (gpu_ocl_ctx) { clReleaseContext(gpu_ocl_ctx); }
        if (cpu_ocl_ctx) { clReleaseContext(cpu_ocl_ctx); }
    }

    cl_context gpu_ocl_ctx = nullptr;
    cl_device_id gpu_ocl_dev = nullptr;

    cl_context cpu_ocl_ctx = nullptr;
    cl_device_id cpu_ocl_dev = nullptr;
};

TEST_P(ocl_engine_test, BasicInteropC) {
    auto p = GetParam();
    cl_device_id ocl_dev = (p.adev_kind == dev_kind::gpu)
            ? gpu_ocl_dev
            : (p.adev_kind == dev_kind::cpu) ? cpu_ocl_dev : nullptr;

    cl_context ocl_ctx = (p.actx_kind == ctx_kind::gpu)
            ? gpu_ocl_ctx
            : (p.actx_kind == ctx_kind::cpu) ? cpu_ocl_ctx : nullptr;

    SKIP_IF(p.adev_kind != dev_kind::null && !ocl_dev,
            "Required OpenCL device not found.");
    SKIP_IF(p.actx_kind != ctx_kind::null && !ocl_ctx,
            "Required OpenCL context not found.");
    SKIP_IF(cpu_ocl_dev == gpu_ocl_dev
                    && (p.adev_kind == dev_kind::cpu
                            || p.actx_kind == ctx_kind::cpu),
            "OpenCL CPU-only device not found.");

    dnnl_engine_t eng = nullptr;
    dnnl_status_t s = dnnl_ocl_interop_engine_create(&eng, ocl_dev, ocl_ctx);

    ASSERT_EQ(s, p.expected_status);

    if (s == dnnl_success) {
        cl_device_id dev = nullptr;
        cl_context ctx = nullptr;

        DNNL_CHECK(dnnl_ocl_interop_get_device(eng, &dev));
        DNNL_CHECK(dnnl_ocl_interop_engine_get_context(eng, &ctx));

        ASSERT_EQ(dev, ocl_dev);
        ASSERT_EQ(ctx, ocl_ctx);

        cl_uint ref_count;
        TEST_OCL_CHECK(clGetContextInfo(ocl_ctx, CL_CONTEXT_REFERENCE_COUNT,
                sizeof(ref_count), &ref_count, nullptr));
        int i_ref_count = int(ref_count);
        ASSERT_EQ(i_ref_count, 2);

        DNNL_CHECK(dnnl_engine_destroy(eng));

        TEST_OCL_CHECK(clGetContextInfo(ocl_ctx, CL_CONTEXT_REFERENCE_COUNT,
                sizeof(ref_count), &ref_count, nullptr));
        i_ref_count = int(ref_count);
        ASSERT_EQ(i_ref_count, 1);
    }
}

TEST_P(ocl_engine_test, BasicInteropCpp) {
    auto p = GetParam();
    cl_device_id ocl_dev = (p.adev_kind == dev_kind::gpu)
            ? gpu_ocl_dev
            : (p.adev_kind == dev_kind::cpu) ? cpu_ocl_dev : nullptr;

    cl_context ocl_ctx = (p.actx_kind == ctx_kind::gpu)
            ? gpu_ocl_ctx
            : (p.actx_kind == ctx_kind::cpu) ? cpu_ocl_ctx : nullptr;

    SKIP_IF(p.adev_kind != dev_kind::null && !ocl_dev,
            "Required OpenCL device not found.");
    SKIP_IF(p.actx_kind != ctx_kind::null && !ocl_ctx,
            "Required OpenCL context not found.");
    SKIP_IF(cpu_ocl_dev == gpu_ocl_dev
                    && (p.adev_kind == dev_kind::cpu
                            || p.actx_kind == ctx_kind::cpu),
            "OpenCL CPU-only device not found.");

    catch_expected_failures(
            [&]() {
                {
                    auto eng = ocl_interop::make_engine(ocl_dev, ocl_ctx);
                    if (p.expected_status != dnnl_success) {
                        FAIL() << "Success not expected";
                    }

                    cl_device_id dev = ocl_interop::get_device(eng);
                    cl_context ctx = ocl_interop::get_context(eng);
                    ASSERT_EQ(dev, ocl_dev);
                    ASSERT_EQ(ctx, ocl_ctx);

                    cl_uint ref_count;
                    TEST_OCL_CHECK(clGetContextInfo(ocl_ctx,
                            CL_CONTEXT_REFERENCE_COUNT, sizeof(ref_count),
                            &ref_count, nullptr));
                    int i_ref_count = int(ref_count);
                    ASSERT_EQ(i_ref_count, 2);
                }

                cl_uint ref_count;
                TEST_OCL_CHECK(
                        clGetContextInfo(ocl_ctx, CL_CONTEXT_REFERENCE_COUNT,
                                sizeof(ref_count), &ref_count, nullptr));
                int i_ref_count = int(ref_count);
                ASSERT_EQ(i_ref_count, 1);
            },
            p.expected_status != dnnl_success, p.expected_status);
}

INSTANTIATE_TEST_SUITE_P(Simple, ocl_engine_test,
        ::testing::Values(ocl_engine_test_params {
                dev_kind::gpu, ctx_kind::gpu, dnnl_success}));

INSTANTIATE_TEST_SUITE_P(InvalidArgs, ocl_engine_test,
        ::testing::Values(ocl_engine_test_params {dev_kind::cpu, ctx_kind::cpu,
                                  dnnl_invalid_arguments},
                ocl_engine_test_params {
                        dev_kind::gpu, ctx_kind::cpu, dnnl_invalid_arguments},
                ocl_engine_test_params {
                        dev_kind::null, ctx_kind::gpu, dnnl_invalid_arguments},
                ocl_engine_test_params {dev_kind::gpu, ctx_kind::null,
                        dnnl_invalid_arguments}));

} // namespace dnnl
