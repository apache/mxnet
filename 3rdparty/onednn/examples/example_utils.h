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

#ifndef EXAMPLE_UTILS_H
#define EXAMPLE_UTILS_H

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dnnl.h"
#include "dnnl_debug.h"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "dnnl_ocl.h"
#endif

#define COMPLAIN_DNNL_ERROR_AND_EXIT(what, status) \
    do { \
        printf("[%s:%d] `%s` returns oneDNN error: %s.\n", __FILE__, __LINE__, \
                what, dnnl_status2str(status)); \
        printf("Example failed.\n"); \
        exit(1); \
    } while (0)

#define COMPLAIN_EXAMPLE_ERROR_AND_EXIT(complain_fmt, ...) \
    do { \
        printf("[%s:%d] Error in the example: " complain_fmt ".\n", __FILE__, \
                __LINE__, __VA_ARGS__); \
        printf("Example failed.\n"); \
        exit(2); \
    } while (0)

static dnnl_engine_kind_t validate_engine_kind(dnnl_engine_kind_t akind) {
    // Checking if a GPU exists on the machine
    if (akind == dnnl_gpu) {
        if (!dnnl_engine_get_count(dnnl_gpu)) {
            printf("Application couldn't find GPU, please run with CPU "
                   "instead.\n");
            exit(0);
        }
    }
    return akind;
}

#define CHECK(f) \
    do { \
        dnnl_status_t s_ = f; \
        if (s_ != dnnl_success) COMPLAIN_DNNL_ERROR_AND_EXIT(#f, s_); \
    } while (0)

static inline dnnl_engine_kind_t parse_engine_kind(int argc, char **argv) {
    // Returns default engine kind, i.e. CPU, if none given
    if (argc == 1) {
        return validate_engine_kind(dnnl_cpu);
    } else if (argc == 2) {
        // Checking the engine type, i.e. CPU or GPU
        char *engine_kind_str = argv[1];
        if (!strcmp(engine_kind_str, "cpu")) {
            return validate_engine_kind(dnnl_cpu);
        } else if (!strcmp(engine_kind_str, "gpu")) {
            return validate_engine_kind(dnnl_gpu);
        }
    }

    // If all above fails, the example should be run properly
    COMPLAIN_EXAMPLE_ERROR_AND_EXIT(
            "inappropriate engine kind.\n"
            "Please run the example like this: %s [cpu|gpu].",
            argv[0]);
}

static inline const char *engine_kind2str_upper(dnnl_engine_kind_t kind) {
    if (kind == dnnl_cpu) return "CPU";
    if (kind == dnnl_gpu) return "GPU";
    return "<Unknown engine>";
}

// Read from memory, write to handle
static inline void read_from_dnnl_memory(void *handle, dnnl_memory_t mem) {
    dnnl_engine_t eng;
    dnnl_engine_kind_t eng_kind;
    const dnnl_memory_desc_t *md;

    CHECK(dnnl_memory_get_engine(mem, &eng));
    CHECK(dnnl_engine_get_kind(eng, &eng_kind));
    CHECK(dnnl_memory_get_memory_desc(mem, &md));
    size_t bytes = dnnl_memory_desc_get_size(md);

#if DNNL_WITH_SYCL
    bool is_cpu_sycl
            = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL && eng_kind == dnnl_cpu);
    bool is_gpu_sycl
            = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL && eng_kind == dnnl_gpu);
    if (is_cpu_sycl || is_gpu_sycl) {
        void *mapped_ptr = NULL;
        CHECK(dnnl_memory_map_data(mem, &mapped_ptr));
        if (mapped_ptr) {
            for (size_t i = 0; i < bytes; ++i) {
                ((char *)handle)[i] = ((char *)mapped_ptr)[i];
            }
        }
        CHECK(dnnl_memory_unmap_data(mem, mapped_ptr));
        return;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (eng_kind == dnnl_gpu) {
        dnnl_stream_t s;
        cl_command_queue q;
        cl_mem m;

        CHECK(dnnl_ocl_interop_memory_get_mem_object(mem, &m));
        CHECK(dnnl_stream_create(&s, eng, dnnl_stream_default_flags));
        CHECK(dnnl_ocl_interop_stream_get_command_queue(s, &q));

        cl_int ret = clEnqueueReadBuffer(
                q, m, CL_TRUE, 0, bytes, handle, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            COMPLAIN_EXAMPLE_ERROR_AND_EXIT(
                    "clEnqueueReadBuffer failed (status code: %d)", ret);

        dnnl_stream_destroy(s);
    }
#endif

    if (eng_kind == dnnl_cpu) {
        void *ptr = NULL;
        CHECK(dnnl_memory_get_data_handle(mem, &ptr));
        if (ptr) {
            for (size_t i = 0; i < bytes; ++i) {
                ((char *)handle)[i] = ((char *)ptr)[i];
            }
        }
        return;
    }

    assert(!"not expected");
}

// Read from handle, write to memory
static inline void write_to_dnnl_memory(void *handle, dnnl_memory_t mem) {
    dnnl_engine_t eng;
    dnnl_engine_kind_t eng_kind;
    const dnnl_memory_desc_t *md;

    CHECK(dnnl_memory_get_engine(mem, &eng));
    CHECK(dnnl_engine_get_kind(eng, &eng_kind));
    CHECK(dnnl_memory_get_memory_desc(mem, &md));
    size_t bytes = dnnl_memory_desc_get_size(md);

#if DNNL_WITH_SYCL
    bool is_cpu_sycl
            = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL && eng_kind == dnnl_cpu);
    bool is_gpu_sycl
            = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL && eng_kind == dnnl_gpu);
    if (is_cpu_sycl || is_gpu_sycl) {
        void *mapped_ptr = NULL;
        CHECK(dnnl_memory_map_data(mem, &mapped_ptr));
        if (mapped_ptr) {
            for (size_t i = 0; i < bytes; ++i) {
                ((char *)mapped_ptr)[i] = ((char *)handle)[i];
            }
        }
        CHECK(dnnl_memory_unmap_data(mem, mapped_ptr));
        return;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (eng_kind == dnnl_gpu) {
        dnnl_stream_t s;
        cl_command_queue q;
        cl_mem m;

        CHECK(dnnl_ocl_interop_memory_get_mem_object(mem, &m));
        CHECK(dnnl_stream_create(&s, eng, dnnl_stream_default_flags));
        CHECK(dnnl_ocl_interop_stream_get_command_queue(s, &q));

        cl_int ret = clEnqueueWriteBuffer(
                q, m, CL_TRUE, 0, bytes, handle, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            COMPLAIN_EXAMPLE_ERROR_AND_EXIT(
                    "clEnqueueWriteBuffer failed (status code: %d)", ret);

        dnnl_stream_destroy(s);
        return;
    }
#endif

    if (eng_kind == dnnl_cpu) {
        void *ptr = NULL;
        CHECK(dnnl_memory_get_data_handle(mem, &ptr));
        if (ptr) {
            for (size_t i = 0; i < bytes; ++i) {
                ((char *)ptr)[i] = ((char *)handle)[i];
            }
        }
        return;
    }

    assert(!"not expected");
}

#endif
