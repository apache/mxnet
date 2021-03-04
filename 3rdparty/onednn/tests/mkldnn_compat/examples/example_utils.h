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

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mkldnn.h"

#define CHECK(f) \
    do { \
        mkldnn_status_t s_ = f; \
        if (s_ != mkldnn_success) { \
            printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f, \
                    s_); \
            exit(2); \
        } \
    } while (0)

#define CHECK_TRUE(expr) \
    do { \
        int e_ = expr; \
        if (!e_) { \
            printf("[%s:%d] %s failed\n", __FILE__, __LINE__, #expr); \
            exit(2); \
        } \
    } while (0)

static mkldnn_engine_kind_t parse_engine_kind(int argc, char **argv) {
    // Returns default engine kind, i.e. CPU, if none given
    if (argc == 1) {
        return mkldnn_cpu;
    } else if (argc == 2) {
        // Checking the engine type, i.e. CPU or GPU
        char *engine_kind_str = argv[1];
        if (!strcmp(engine_kind_str, "cpu")) {
            return mkldnn_cpu;
        } else if (!strcmp(engine_kind_str, "gpu")) {
            // Checking if a GPU exists on the machine
            if (!mkldnn_engine_get_count(mkldnn_gpu)) {
                fprintf(stderr,
                        "Application couldn't find GPU, please run with CPU "
                        "instead. Thanks!\n");
                exit(0);
            }
            return mkldnn_gpu;
        }
    }

    // If all above fails, the example should be ran properly
    fprintf(stderr, "Please run example like this: %s cpu|gpu\n", argv[0]);
    exit(1);
}

// Read from memory, write to handle
static inline void read_from_mkldnn_memory(void *handle, mkldnn_memory_t mem) {
    mkldnn_engine_t eng;
    mkldnn_engine_kind_t eng_kind;
    const mkldnn_memory_desc_t *md;

    CHECK(mkldnn_memory_get_engine(mem, &eng));
    CHECK(mkldnn_engine_get_kind(eng, &eng_kind));
    CHECK(mkldnn_memory_get_memory_desc(mem, &md));
    size_t bytes = mkldnn_memory_desc_get_size(md);

    if (eng_kind == mkldnn_cpu) {
        void *ptr = NULL;
        CHECK(mkldnn_memory_get_data_handle(mem, &ptr));
        if (ptr) {
            for (size_t i = 0; i < bytes; ++i) {
                ((char *)handle)[i] = ((char *)ptr)[i];
            }
        }
    }
}

// Read from handle, write to memory
static inline void write_to_mkldnn_memory(void *handle, mkldnn_memory_t mem) {
    mkldnn_engine_t eng;
    mkldnn_engine_kind_t eng_kind;
    const mkldnn_memory_desc_t *md;

    CHECK(mkldnn_memory_get_engine(mem, &eng));
    CHECK(mkldnn_engine_get_kind(eng, &eng_kind));
    CHECK(mkldnn_memory_get_memory_desc(mem, &md));
    size_t bytes = mkldnn_memory_desc_get_size(md);

    if (eng_kind == mkldnn_cpu) {
        void *ptr = NULL;
        CHECK(mkldnn_memory_get_data_handle(mem, &ptr));
        if (ptr) {
            for (size_t i = 0; i < bytes; ++i) {
                ((char *)handle)[i] = ((char *)ptr)[i];
            }
        }
    }
}

#endif
