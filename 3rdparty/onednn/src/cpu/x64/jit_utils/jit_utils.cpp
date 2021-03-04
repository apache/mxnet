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

#include <mutex>

#include "common/utils.hpp"

#ifndef DNNL_ENABLE_JIT_PROFILING
#define DNNL_ENABLE_JIT_PROFILING 1
#endif

#ifndef DNNL_ENABLE_JIT_DUMP
#define DNNL_ENABLE_JIT_DUMP 1
#endif

#if DNNL_ENABLE_JIT_PROFILING
#include "cpu/x64/jit_utils/jitprofiling/jitprofiling.h"
#ifdef __linux__
#include "cpu/x64/jit_utils/linux_perf/linux_perf.hpp"
#endif
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace jit_utils {

// WARNING: These functions are not thread safe and must be protected by a
// mutex

// TODO (rsdubtso): support prefix for code dumps

void dump_jit_code(const void *code, size_t code_size, const char *code_name) {
#if DNNL_ENABLE_JIT_DUMP
    if (code && get_jit_dump()) {
        static int counter = 0;
#define MAX_FNAME_LEN 256
        char fname[MAX_FNAME_LEN + 1];
        // TODO (Roma): support prefix for code / linux perf dumps
        snprintf(fname, MAX_FNAME_LEN, "dnnl_dump_%s.%d.bin", code_name,
                counter);
        counter++;

        FILE *fp = fopen(fname, "w+");
        // Failure to dump code is not fatal
        if (fp) {
            size_t unused = fwrite(code, code_size, 1, fp);
            UNUSED(unused);
            fclose(fp);
        }
    }
#undef MAX_FNAME_LEN
#else
    UNUSED(code);
    UNUSED(code_size);
    UNUSED(code_name);
#endif
}

void register_jit_code_vtune(const void *code, size_t code_size,
        const char *code_name, const char *source_file_name) {
#if DNNL_ENABLE_JIT_PROFILING
    unsigned flags = get_jit_profiling_flags();
    if ((flags & DNNL_JIT_PROFILE_VTUNE)
            && iJIT_IsProfilingActive() == iJIT_SAMPLING_ON) {
        auto jmethod = iJIT_Method_Load();
        jmethod.method_id = iJIT_GetNewMethodID(); // XXX: not thread-safe
        jmethod.method_name = (char *)code_name; // XXX: dropping const
        jmethod.class_file_name = nullptr;
        jmethod.source_file_name
                = (char *)source_file_name; // XXX: dropping const
        jmethod.method_load_address = (void *)code;
        jmethod.method_size = (unsigned int)code_size;

        iJIT_NotifyEvent(
                iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED, (void *)&jmethod);
    }
#else
    UNUSED(code);
    UNUSED(code_size);
    UNUSED(code_name);
    UNUSED(source_file_name);
#endif
}

void register_jit_code_linux_perf(const void *code, size_t code_size,
        const char *code_name, const char *source_file_name) {
#if DNNL_ENABLE_JIT_PROFILING && defined(__linux__)
    unsigned flags = get_jit_profiling_flags();
    if (flags & DNNL_JIT_PROFILE_LINUX_JITDUMP)
        linux_perf_jitdump_record_code_load(code, code_size, code_name);
    if (flags & DNNL_JIT_PROFILE_LINUX_PERFMAP)
        linux_perf_perfmap_record_code_load(code, code_size, code_name);
#else
    UNUSED(code);
    UNUSED(code_size);
    UNUSED(code_name);
#endif
    UNUSED(source_file_name);
}

void register_jit_code(const void *code, size_t code_size,
        const char *code_name, const char *source_file_name) {
    // The #ifdef guards are required to avoid generating a function that only
    // consists of lock and unlock code
#if DNNL_ENABLE_JIT_PROFILING || DNNL_ENABLE_JIT_DUMP
    static std::mutex m;
    std::lock_guard<std::mutex> guard(m);

    dump_jit_code(code, code_size, code_name);
    register_jit_code_vtune(code, code_size, code_name, source_file_name);
    register_jit_code_linux_perf(code, code_size, code_name, source_file_name);
#else
    UNUSED(code);
    UNUSED(code_size);
    UNUSED(code_name);
    UNUSED(source_file_name);
#endif
}

} // namespace jit_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
