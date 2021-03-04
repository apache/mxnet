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

#ifndef GPU_COMPUTE_DEVICE_INFO_HPP
#define GPU_COMPUTE_DEVICE_INFO_HPP

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "common/z_magic.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

enum class gpu_arch_t {
    unknown,
    gen9,
    gen12lp,
};

enum class device_ext_t : uint64_t {
    // clang-format off
    // OpenCL data types
    khr_fp16 = 1ull << 0,
    khr_fp64 = 1ull << 1,
    // OpenCL atomics
    khr_global_int32_base_atomics     = 1ull << 2,
    khr_global_int32_extended_atomics = 1ull << 3,
    khr_int64_base_atomics            = 1ull << 4,
    khr_int64_extended_atomics        = 1ull << 5,
    khr_local_int32_base_atomics      = 1ull << 6,
    khr_local_int32_extended_atomics  = 1ull << 7,
    // Intel specific Gen9+
    intel_subgroups              = 1ull << 16,
    intel_required_subgroup_size = 1ull << 17,
    intel_subgroups_char         = 1ull << 18,
    intel_subgroups_short        = 1ull << 19,
    intel_subgroups_long         = 1ull << 20,
    // Intel specific Gen12LP+
    intel_subgroup_local_block_io = 1ull << 21,
    intel_dot_accumulate          = 1ull << 22,
    last
    // clang-format on
};

inline gpu_arch_t str2gpu_arch(const char *str) {
#define CASE(_case) \
    if (!strcmp(STRINGIFY(_case), str)) return gpu_arch_t::_case

    CASE(gen9);
    CASE(gen12lp);
    return gpu_arch_t::unknown;
#undef CASE
}

inline const char *gpu_arch2str(gpu_arch_t arch) {
#define CASE(_case) \
    case gpu_arch_t::_case: return STRINGIFY(_case)

    switch (arch) {
        CASE(gen9);
        CASE(gen12lp);
        CASE(unknown);
    }
    return "unknown";
#undef CASE
}

static inline const char *ext2cl_str(device_ext_t ext) {
#define CASE(x) \
    case device_ext_t::x: return STRINGIFY(CONCAT2(cl_, x));
    switch (ext) {
        CASE(khr_fp16)
        CASE(khr_fp64)

        CASE(khr_global_int32_base_atomics)
        CASE(khr_global_int32_extended_atomics)
        CASE(khr_int64_base_atomics)
        CASE(khr_int64_extended_atomics)
        CASE(khr_local_int32_base_atomics)
        CASE(khr_local_int32_extended_atomics)

        CASE(intel_subgroups)
        CASE(intel_required_subgroup_size)
        CASE(intel_subgroups_char)
        CASE(intel_subgroups_short)
        CASE(intel_subgroups_long)

        CASE(intel_subgroup_local_block_io)
        CASE(intel_dot_accumulate)

        default: return nullptr;
    }
#undef CASE
}

struct runtime_version_t {
    int major;
    int minor;
    int build;

    runtime_version_t(int major = 0, int minor = 0, int build = 0)
        : major {major}, minor {minor}, build {build} {}

    bool operator==(const runtime_version_t &other) const {
        return (major == other.major) && (minor == other.minor)
                && (build == other.build);
    }

    bool operator!=(const runtime_version_t &other) const {
        return !(*this == other);
    }

    bool operator<(const runtime_version_t &other) const {
        if (major < other.major) return true;
        if (major > other.major) return false;
        if (minor < other.minor) return true;
        if (minor > other.minor) return false;
        return (build < other.build);
    }

    bool operator>(const runtime_version_t &other) const {
        return (other < *this);
    }

    bool operator<=(const runtime_version_t &other) const {
        return !(*this > other);
    }

    bool operator>=(const runtime_version_t &other) const {
        return !(*this < other);
    }

    status_t set_from_string(const char *s) {
        int i_major = 0, i = 0;

        for (; s[i] != '.'; i++)
            if (!s[i]) return status::invalid_arguments;

        auto i_minor = ++i;

        for (; s[i] != '.'; i++)
            if (!s[i]) return status::invalid_arguments;

        auto i_build = ++i;

        major = atoi(&s[i_major]);
        minor = atoi(&s[i_minor]);
        build = atoi(&s[i_build]);

        return status::success;
    }

    std::string str() const {
        return utils::format("%d.%d.%d", major, minor, build);
    }
};

struct device_info_t {
public:
    virtual ~device_info_t() = default;

    status_t init() {
        CHECK(init_device_name());
        CHECK(init_arch());
        CHECK(init_runtime_version());
        CHECK(init_extensions());
        CHECK(init_attributes());

        return status::success;
    }

    virtual bool has(device_ext_t ext) const = 0;

    virtual gpu_arch_t gpu_arch() const = 0;
    virtual int eu_count() const = 0;
    virtual int hw_threads() const = 0;
    virtual size_t llc_cache_size() const = 0;

    const runtime_version_t &runtime_version() const {
        return runtime_version_;
    }
    const std::string &name() const { return name_; }

protected:
    void set_runtime_version(const runtime_version_t &runtime_version) {
        runtime_version_ = runtime_version;
    }

    void set_name(const std::string &name) { name_ = name; }

private:
    virtual status_t init_arch() = 0;
    virtual status_t init_device_name() = 0;
    virtual status_t init_runtime_version() = 0;
    virtual status_t init_extensions() = 0;
    virtual status_t init_attributes() = 0;

    runtime_version_t runtime_version_;
    std::string name_;
};

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
