/*******************************************************************************
 * Copyright 2020 Intel Corporation
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

#include "memory_debug.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#if defined __linux__ || defined __APPLE__
#include <unistd.h>
#include <sys/mman.h>
#endif

#include <assert.h>

#include "dnnl_thread.hpp"
#include "nstl.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {
namespace memory_debug {

template <typename T>
static inline T get_page_start(const void *ptr) {
    size_t page_mask = ~(getpagesize() - 1);
    size_t ptr_cast = reinterpret_cast<size_t>(ptr);
    return reinterpret_cast<T>(ptr_cast & page_mask);
}

template <typename T>
static inline T get_page_end(const void *ptr) {
    size_t page_mask = ~(getpagesize() - 1);
    size_t ptr_cast = reinterpret_cast<size_t>(ptr);
    return reinterpret_cast<T>((ptr_cast + getpagesize() - 1) & page_mask);
}

static inline int num_protect_pages() {
    if (is_mem_debug())
        return DNNL_MEM_DEBUG_PROTECT_SIZE;
    else
        return 0;
}

size_t protect_size() {
    return (size_t)num_protect_pages() * getpagesize();
}

#ifdef _WIN32
#define PROT_NONE 0
#define PROT_READ 1
#define PROT_WRITE 2
static inline int mprotect(void *addr, size_t len, int prot) {
    // TODO: Create a mprotect emulation layer to improve debug scratchpad
    // support on windows. This should require the windows.h and memoryapi.h
    // headers
    return 0;
}
#endif

struct memory_tag_t {
    void *memory_start;
    size_t buffer_size;
};

static inline memory_tag_t *get_memory_tags(void *ptr) {
    return get_page_start<memory_tag_t *>(ptr) - 1;
}

void *malloc(size_t size, int alignment) {
    void *ptr;

    size_t buffer_size = utils::rnd_up(size, alignment);
    int buffer_alignment = alignment;
    if (buffer_alignment < getpagesize()) alignment = getpagesize();
    size = utils::rnd_up(
            size + alignment + 2 * protect_size(), (size_t)alignment);

#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
    int rc = ptr ? 0 : -1;
#else
    int rc = ::posix_memalign(&ptr, alignment, size);
#endif

    if (rc == 0) {
        void *mem_start = ptr;
        ptr = utils::align_ptr(
                reinterpret_cast<char *>(ptr) + protect_size(), alignment);
        if (is_mem_debug_overflow()) {
            size_t offset = (alignment - (buffer_size % alignment)) % alignment;
            ptr = reinterpret_cast<char *>(ptr) + offset;
        }
        assert(protect_size() >= 16);
        memory_tag_t *tag = get_memory_tags(ptr);
        tag->memory_start = mem_start;
        tag->buffer_size = buffer_size;
        protect_buffer(ptr, buffer_size, engine_kind_t::dnnl_cpu);
    }

    return (rc == 0) ? ptr : nullptr;
}

void free(void *p) {
    if (p != nullptr) {
        memory_tag_t *tag = get_memory_tags(p);
        int status;
        MAYBE_UNUSED(status);

        status = mprotect(get_page_start<void *>(tag), getpagesize(),
                PROT_WRITE | PROT_READ);
        assert(status == 0);
        unprotect_buffer(p, tag->buffer_size, engine_kind_t::dnnl_cpu);

        p = tag->memory_start;
    }

#ifdef _WIN32
    _aligned_free(p);
#else
    ::free(p);
#endif
}

// Assumes the input buffer is allocated such that there is num_protect_pages()
// pages surrounding the buffer
void protect_buffer(void *addr, size_t size, engine_kind_t engine_kind) {
    if (engine_kind != engine_kind_t::dnnl_cpu)
        return; // Only CPU is supported currently

    char *page_start = get_page_start<char *>(addr);
    char *page_end
            = get_page_end<char *>(reinterpret_cast<const char *>(addr) + size);
    int status;
    MAYBE_UNUSED(status);

    status = mprotect(page_start - protect_size(), protect_size(), PROT_NONE);
    assert(status == 0);
    status = mprotect(page_end, protect_size(), PROT_NONE);
    assert(status == 0);

    // The canary is set so that it will generate NaN for floating point
    // data types. This causes uninitialized memory usage on floating point
    // data to be poisoned, increasing the chance the error is caught.
    uint16_t canary = 0x7ff1;
    size_t work_amount = (size_t)((page_end - page_start) / getpagesize());
    if (work_amount <= 1) {
        // Avoid large memory initializations for small buffers
        uint16_t *ptr_start = reinterpret_cast<uint16_t *>(
                reinterpret_cast<size_t>(addr) & ~1);
        uint16_t *ptr_end = reinterpret_cast<uint16_t *>(
                reinterpret_cast<char *>(addr) + size);
        for (uint16_t *curr = ptr_start; curr < ptr_end; curr++) {
            *curr = canary;
        }
    } else {
        parallel(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0;
            balance211(work_amount, nthr, ithr, start, end);
            uint16_t *ptr_start = reinterpret_cast<uint16_t *>(
                    page_start + getpagesize() * start);
            uint16_t *ptr_end = reinterpret_cast<uint16_t *>(
                    page_start + getpagesize() * end);

            for (uint16_t *curr = ptr_start; curr < ptr_end; curr++) {
                *curr = canary;
            }
        });
    }
}

// Assumes the input buffer is allocated such that there is num_protect_pages()
// pages surrounding the buffer
void unprotect_buffer(
        const void *addr, size_t size, engine_kind_t engine_kind) {
    if (engine_kind != engine_kind_t::dnnl_cpu)
        return; // Only CPU is supported currently

    char *page_start = get_page_start<char *>(addr);
    char *page_end
            = get_page_end<char *>(reinterpret_cast<const char *>(addr) + size);
    int status;
    MAYBE_UNUSED(status);

    status = mprotect(page_start - protect_size(), protect_size(),
            PROT_WRITE | PROT_READ);
    assert(status == 0);
    status = mprotect(page_end, protect_size(), PROT_WRITE | PROT_READ);
    assert(status == 0);
}

} // namespace memory_debug
} // namespace impl
} // namespace dnnl
