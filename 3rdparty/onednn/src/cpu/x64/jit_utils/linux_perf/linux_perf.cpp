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

// A quick-and-dirty implementation of
// ----------------------------------
// tools/perf/Documentation/jitdump-specification.txt
// tools/perf/Documentation/jit-interface.txt

// WARNING: this implementation is inherently non-thread-safe. Any calls to
// linux_perf_record_code_load() MUST be protected by a mutex.

#ifdef __linux__

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <syscall.h>
#include <unistd.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

#include <string>

#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/x64/jit_utils/linux_perf/linux_perf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace jit_utils {

class linux_perf_jitdump_t {
public:
    linux_perf_jitdump_t()
        : marker_addr_ {nullptr}
        , marker_size_ {0}
        , fd_ {-1}
        , failed_ {false}
        , use_tsc_ {false} {
        // The initialization is lazy and nothing happens if no JIT-ed code
        // need to be recorded.
    }

    ~linux_perf_jitdump_t() {
        write_code_close();
        finalize();
    }

    void record_code_load(
            const void *code, size_t code_size, const char *code_name) {
        if (is_active()) write_code_load(code, code_size, code_name);
    }

private:
    bool is_active() {
        if (fd_ >= 0) return true;
        if (failed_) return false;
        return initialize();
    }

    bool initialize() {
        if (!open_file()) return fail();
        if (!create_marker()) return fail();
        if (!write_header()) return fail();
        return true;
    }

    void finalize() {
        close_file();
        delete_marker();
    }

    bool fail() {
        finalize();
        failed_ = true;
        return false;
    }

    bool open_file() {
        auto path_len_ok = [&](const std::string &path) {
            if (path.length() >= PATH_MAX) {
                if (get_verbose())
                    printf("dnnl_verbose,jit_perf,error,"
                           "dump directory path '%s' is too long\n",
                            path.c_str());
                return false;
            }
            return true;
        };

        auto complain = [](const std::string &path) {
            if (get_verbose())
                printf("dnnl_verbose,jit_perf,error,"
                       "cannot create dump directory '%s' (%m)\n",
                        path.c_str());
            return false;
        };

        auto make_dir = [&](const std::string &path) {
            if (!path_len_ok(path)) return false;
            if (mkdir(path.c_str(), 0755) == -1 && errno != EEXIST)
                return complain(path);
            return true;
        };

        auto make_temp_dir = [&](std::string &path) {
            if (!path_len_ok(path)) return false;
            if (mkdtemp(&path[0]) == nullptr) return complain(path);
            return true;
        };

        std::string path(get_jit_profiling_jitdumpdir());
        path.reserve(PATH_MAX);

        if (!make_dir(path)) return false;

        path += "/.debug";
        if (!make_dir(path)) return false;

        path += "/jit";
        if (!make_dir(path)) return false;

        path += "/dnnl.XXXXXX";
        if (!make_temp_dir(path)) return false;

        path += "/jit-" + std::to_string(getpid()) + ".dump";
        if (!path_len_ok(path)) return false;

        fd_ = open(path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0666);
        if (fd_ == -1) {
            if (get_verbose())
                printf("dnnl_verbose,jit_perf,error,"
                       "cannot open jitdump file '%s' (%m)\n",
                        path.c_str());
            return false;
        }

        return true;
    }

    void close_file() {
        if (fd_ == -1) return;
        close(fd_);
        fd_ = -1;
    }

    bool create_marker() {
        // Perf will record an mmap() call and then will find the file we
        // write the JIT-ed code to. PROT_EXEC ensures that the record is not
        // ignored.
        long page_size = sysconf(_SC_PAGESIZE);
        if (page_size == -1) return false;
        marker_size_ = (size_t)page_size;
        marker_addr_ = mmap(nullptr, marker_size_, PROT_READ | PROT_EXEC,
                MAP_PRIVATE, fd_, 0);
        return marker_addr_ != MAP_FAILED;
    }

    void delete_marker() {
        if (marker_addr_) munmap(marker_addr_, marker_size_);
    }

    static uint64_t get_timestamp(bool use_tsc) {
        if (use_tsc) {
            uint32_t hi, lo;
            asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
            return (((uint64_t)hi) << 32) | lo;
        } else {
            struct timespec ts;
            int rc = clock_gettime(CLOCK_MONOTONIC, &ts);
            if (rc) return 0;
            return (ts.tv_sec * 1000000000UL) + ts.tv_nsec;
        }
    }

    static pid_t gettid() {
        // https://sourceware.org/bugzilla/show_bug.cgi?id=6399
        return (pid_t)syscall(__NR_gettid);
    }

    bool write_or_fail(const void *buf, size_t size) {
        // Write data to the output file or do nothing if the object is in the
        // failed state. Enter failed state on errors.
        if (failed_) return false;
        ssize_t ret = write(fd_, buf, size);
        if (ret == -1) return fail();
        return true;
    }

    bool write_header() {
        struct {
            uint32_t magic;
            uint32_t version;
            uint32_t total_size;
            uint32_t elf_mach;
            uint32_t pad1;
            uint32_t pid;
            uint64_t timestamp;
            uint64_t flags;
        } h;
        h.magic = 0x4A695444; // JITHEADER_MAGIC ('DTiJ')
        h.version = 1;
        h.total_size = sizeof(h);
        h.elf_mach = EM_X86_64;
        h.pad1 = 0;
        h.pid = getpid();

        use_tsc_ = get_jit_profiling_flags()
                & DNNL_JIT_PROFILE_LINUX_JITDUMP_USE_TSC;
        h.timestamp = get_timestamp(use_tsc_);
        h.flags = use_tsc_ ? 1 : 0;

        return write_or_fail(&h, sizeof(h));
    }

    bool write_code_close() {
        struct {
            uint32_t id;
            uint32_t total_size;
            uint64_t timestamp;
        } c;
        c.id = 3; // JIT_CODE_CLOSE
        c.total_size = sizeof(c);
        c.timestamp = get_timestamp(use_tsc_);
        return write_or_fail(&c, sizeof(c));
    }

    bool write_code_load(
            const void *code, size_t code_size, const char *code_name) {
        // XXX (rsdubtso): There is no limit on code_size or code_name. This
        // may lead to huge output files. Do we care?
        static uint64_t code_index = 0;
        struct {
            uint32_t id;
            uint32_t total_size;
            uint64_t timestamp;
            uint32_t pid;
            uint32_t tid;
            uint64_t vma;
            uint64_t code_addr;
            uint64_t code_size;
            uint64_t code_index;
        } c;
        c.id = 0; // JIT_CODE_LOAD
        c.total_size = sizeof(c) + strlen(code_name) + 1 + code_size;
        c.timestamp = get_timestamp(use_tsc_);
        c.pid = getpid();
        c.tid = gettid();
        c.vma = c.code_addr = (uint64_t)code;
        c.code_size = code_size;
        c.code_index = code_index++;
        write_or_fail(&c, sizeof(c));
        write_or_fail(code_name, strlen(code_name) + 1);
        return write_or_fail(code, code_size);
    }

    void *marker_addr_;
    size_t marker_size_;
    int fd_;
    bool failed_;
    bool use_tsc_;
};

void linux_perf_jitdump_record_code_load(
        const void *code, size_t code_size, const char *code_name) {
    static linux_perf_jitdump_t jitdump;
    jitdump.record_code_load(code, code_size, code_name);
}

class linux_perf_jitmap_t {
public:
    linux_perf_jitmap_t() : fp_ {nullptr}, failed_ {false} {}
    ~linux_perf_jitmap_t() = default;
    void record_symbol(
            const void *code, size_t code_size, const char *code_name) {
        if (is_initialized()) write_symbol_info(code, code_size, code_name);
    }

private:
    bool is_initialized() {
        if (fp_) return true;
        if (failed_) return false;
        return initialize();
    }

    bool open_map_file() {
        char fname[PATH_MAX];
        int ret = snprintf(fname, PATH_MAX, "/tmp/perf-%d.map", getpid());
        if (ret >= PATH_MAX) return fail();

        fp_ = fopen(fname, "w+");
        if (!fp_) return fail();
        setvbuf(fp_, nullptr, _IOLBF, 0); // disable line buffering

        return true;
    }

    void close_map_file() {
        if (fp_) fclose(fp_);
    }

    bool initialize() { return open_map_file(); }

    bool fail() {
        close_map_file();
        failed_ = true;
        return false;
    }

    void write_symbol_info(
            const void *code, size_t code_size, const char *code_name) {
        if (failed_) return;

        int ret = fprintf(fp_, "%llx %llx %s\n", (unsigned long long)code,
                (unsigned long long)code_size, code_name);

        if (ret == EOF || ret < 0) fail();
    }

    FILE *fp_;
    bool failed_;
};

void linux_perf_perfmap_record_code_load(
        const void *code, size_t code_size, const char *code_name) {
    static linux_perf_jitmap_t jitmap;
    jitmap.record_symbol(code, code_size, code_name);
}

} // namespace jit_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
