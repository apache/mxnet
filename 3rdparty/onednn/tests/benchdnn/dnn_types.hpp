/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#ifndef DNN_TYPES_HPP
#define DNN_TYPES_HPP

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "common.hpp"
#include "oneapi/dnnl/dnnl_types.h"

namespace tag {
extern const char *x;
extern const char *abx;
extern const char *axb;
extern const char *any;
extern const char *undef;
} // namespace tag

struct dims_t : public std::vector<int64_t> {
    //  using vector<int64_t>::vector;
    //  There is a bug in Intel compiler 19.0 on MacOS which prevents
    //  using-declaration from being used here. The workaround is to introduce
    //  constructors explicitly.
    dims_t() = default;
    dims_t(size_t size) : vector(size) {}
    dims_t(size_t size, int64_t value) : vector(size, value) {}
};

enum dir_t {
    DIR_UNDEF = 0,
    FLAG_DAT = 1,
    FLAG_WEI = 2,
    FLAG_BIA = 4,
    FLAG_FWD = 32,
    FLAG_BWD = 64,
    FLAG_INF = 128,
    FWD_D = FLAG_FWD + FLAG_DAT,
    FWD_I = FLAG_FWD + FLAG_DAT + FLAG_INF,
    FWD_B = FLAG_FWD + FLAG_DAT + FLAG_BIA,
    BWD_D = FLAG_BWD + FLAG_DAT,
    BWD_DW = FLAG_BWD + FLAG_DAT + FLAG_WEI,
    BWD_W = FLAG_BWD + FLAG_WEI,
    BWD_WB = FLAG_BWD + FLAG_WEI + FLAG_BIA,
};
dir_t str2dir(const char *str);

/* TODO: merge prop and dir_t (in favor of prop) */
const char *prop2str(dnnl_prop_kind_t prop);
dnnl_prop_kind_t prop2prop_kind(dir_t dir);

dims_t off2dims_idx(const dims_t &dims, int64_t off);
std::ostream &operator<<(std::ostream &s, const dims_t &dims);
std::ostream &operator<<(std::ostream &s, dir_t dir);
std::ostream &operator<<(std::ostream &s, dnnl_data_type_t dt);
std::ostream &operator<<(std::ostream &s, dnnl_engine_kind_t ek);
template <typename T>
std::ostream &operator<<(std::ostream &s, const std::vector<T> &v) {
    s << v[0];
    for (size_t d = 1; d < v.size(); ++d)
        s << ":" << v[d];
    return s;
}

typedef int data_kind_t;
enum { SRC = 0, WEI, BIA, DST, ACC, DATA, MEAN, VAR, SS, GWEI, DAT_TOTAL };
const char *data_kind2str(data_kind_t kind);

struct attr_t {
    // policy_t defines the way entity values will be applied to a tensor
    enum policy_t {
        COMMON = 0, // single value for each point in a tensor
        PER_OC, // apply a single value per each channel (dims[1]) point
        PER_DIM_0, // apply a single value er dims[0] point
        PER_DIM_1, // ... per dims[1] point
        PER_DIM_01, // ... per unique combination of dims[0] and dims[1] points
        POLICY_TOTAL // guard
    };

    static policy_t str2policy(const std::string &str);
    static const char *policy2str(policy_t policy);
    static int get_default_mask(policy_t policy);

    struct scale_t {
        scale_t(policy_t apolicy = COMMON, float ascale = 1.,
                bool aruntime = false)
            : policy(apolicy), scale(ascale), runtime(aruntime) {}

        int from_str(const std::string &s);

        bool is_def() const {
            return policy == COMMON && scale == 1. && runtime == false;
        }

        policy_t policy = COMMON;
        float scale = 1.;
        bool runtime = false;
    };

    struct zero_points_t {
        struct entry_t {
            entry_t(policy_t apolicy = COMMON, int avalue = 0,
                    bool aruntime = false)
                : policy(apolicy), value(avalue), runtime(aruntime) {}

            entry_t(const entry_t &other)
                : policy(other.policy)
                , value(other.value)
                , runtime(other.runtime) {}

            bool is_def() const {
                return policy == COMMON && value == 0 && runtime == false;
            }

            policy_t policy = COMMON;
            int value = 0;
            bool runtime = false;
        };

        int from_str(const std::string &s);

        int operator[](int arg) const { return get(arg).value; }
        bool runtime(int arg) const { return get(arg).runtime; }

        bool is_def(int arg) const { return get(arg).is_def(); }
        bool is_def() const { return points.empty(); }

        void set(int arg, policy_t policy, int value, bool runtime) {
            set(arg, entry_t(policy, value, runtime));
        }
        void set(int arg, const entry_t &entry) {
            if (!entry.is_def()) points[arg] = entry;
        }
        entry_t get(int arg) const {
            const auto it = points.find(arg);
            return it == points.end() ? entry_t() : it->second;
        }

        std::map<int, entry_t>::const_iterator begin() const {
            return points.begin();
        }
        std::map<int, entry_t>::const_iterator end() const {
            return points.end();
        }

        zero_points_t() : points() {} // needed for debug icc190 build;

        std::map<int, entry_t> points;
    };

    struct arg_scales_t {
        void set(int arg, scale_t scale) {
            scales.insert(std::make_pair(arg, scale));
        }

        scale_t get(int arg) const {
            const auto &s = scales.find(arg);
            return s == scales.end() ? scale_t() : s->second;
        }

        bool is_def(int arg) const { return get(arg).is_def(); }
        bool is_def() const { return scales.empty(); }
        int from_str(const std::string &s);

        arg_scales_t() : scales() {} // needed for debug icc190 build;

        std::map<int, scale_t> scales;
    };

    struct post_ops_t {
        enum kind_t {
            // sum
            SUM,
            // depthwise convolution
            DW_K3S1P1,
            DW_K3S2P1,
            // eltwise
            ELTWISE_START, // a guard to check kind is eltwise
            ABS,
            BRELU,
            CLIP,
            ELU,
            ELU_DST,
            EXP,
            EXP_DST,
            GELU_ERF,
            GELU_TANH,
            LINEAR,
            LOG,
            LOGISTIC,
            LOGISTIC_DST,
            POW,
            RELU,
            RELU_DST,
            ROUND,
            SQRT,
            SQRT_DST,
            SQUARE,
            SRELU,
            SWISH,
            TANH,
            TANH_DST,
            ELTWISE_END, // a guard to check kind is eltwise
            // binary
            BINARY_START, // a guard to check kind is binary
            ADD,
            DIV,
            MAX,
            MIN,
            MUL,
            BINARY_END, // a guard to check kind is binary
            // guard entry
            KIND_TOTAL
        };
        static kind_t str2kind(const std::string &str);
        static const char *kind2str(kind_t kind);
        static dnnl_alg_kind_t kind2dnnl_kind(kind_t kind);

        struct entry_t {
            entry_t(kind_t akind) : kind(akind) {
                if (is_sum_kind()) {
                    sum.scale = 1.f;
                    sum.dt = dnnl_data_type_undef;
                } else if (is_eltwise_kind()) {
                    eltwise.alg = kind2dnnl_kind(kind);
                    eltwise.alpha = 0.f;
                    eltwise.beta = 0.f;
                    eltwise.scale = 1.f;
                } else if (is_convolution_kind()) {
                    convolution.stride = kind == DW_K3S1P1 ? 1 : 2;
                    convolution.dst_dt = dnnl_f32;
                    convolution.oscale = scale_t();
                } else if (is_binary_kind()) {
                    binary.alg = kind2dnnl_kind(kind);
                    binary.src1_dt = dnnl_data_type_undef;
                    binary.policy = policy_t::COMMON;
                }
            }

            kind_t kind;
            union {
                struct {
                    float scale;
                    dnnl_data_type_t dt;
                } sum;
                struct {
                    dnnl_alg_kind_t alg;
                    float alpha, beta, scale;
                } eltwise;
                struct {
                    int stride;
                    dnnl_data_type_t dst_dt;
                    scale_t oscale;
                } convolution;
                struct {
                    dnnl_alg_kind_t alg;
                    dnnl_data_type_t src1_dt;
                    policy_t policy;
                } binary;
            };

            bool is_sum_kind() const;
            bool is_convolution_kind() const;
            bool is_eltwise_kind() const;
            bool is_binary_kind() const;
        };

        post_ops_t() : entry() {}

        int from_str(const std::string &s);

        int len() const { return (int)entry.size(); }
        bool is_def() const { return len() == 0; }

        int find(kind_t kind, int start = 0, int stop = -1) const;
        int eltwise_index() const;
        int convolution_index() const;
        int binary_index() const;

        std::vector<int> get_binary_po_masks() const;

        std::vector<entry_t> entry;
    };

    attr_t() : scratchpad_mode(dnnl_scratchpad_mode_library) {}

    void insert(const scale_t &s) { this->oscale = s; }
    void insert(const arg_scales_t &as) { this->scales = as; }
    void insert(const zero_points_t &zp) { this->zero_points = zp; }
    void insert(const post_ops_t &po) { this->post_ops = po; }
    void insert(dnnl_scratchpad_mode_t sm) { this->scratchpad_mode = sm; }

    scale_t oscale;
    arg_scales_t scales;
    zero_points_t zero_points;
    post_ops_t post_ops;
    dnnl_scratchpad_mode_t scratchpad_mode;

    bool is_def() const;
};
using policy_t = attr_t::policy_t;

void handle_legacy_attr(attr_t &attr, const attr_t &legacy_attr);

int str2attr(attr_t *attr, const char *str);
std::ostream &operator<<(std::ostream &s, const policy_t &policy);
std::ostream &operator<<(std::ostream &s, const attr_t::scale_t &scale);
std::ostream &operator<<(
        std::ostream &s, const attr_t::zero_points_t &zero_points);
std::ostream &operator<<(std::ostream &s, const attr_t::arg_scales_t &scales);
std::ostream &operator<<(std::ostream &s, const attr_t::post_ops_t::kind_t &k);
std::ostream &operator<<(std::ostream &s, const attr_t::post_ops_t &post_ops);
std::ostream &operator<<(std::ostream &s, dnnl_scratchpad_mode_t sm);
std::ostream &operator<<(std::ostream &s, const attr_t &attr);

// A container for additional data and info, not available from user's input at
// parse time, but which are required to create the library attributes.
struct attr_args_t {
    struct entry_t {
        entry_t(const void *vals = NULL, int64_t count = 1, int mask = -1,
                bool runtime = false)
            : vals(vals), count(count), mask(mask), runtime(runtime) {}

        int64_t get_count(policy_t policy) const {
            return (policy == policy_t::COMMON || runtime) ? 1 : count;
        }

        int get_mask(policy_t policy) const {
            return mask == -1 ? attr_t::get_default_mask(policy) : mask;
        }

        const float *get_float_ptr() const {
            return runtime ? &DNNL_RUNTIME_F32_VAL
                           : static_cast<const float *>(vals);
        }

        const void *vals = NULL;
        int64_t count = 1;
        int mask = -1;
        bool runtime = false;
    };

    attr_args_t() = default;

    void prepare_output_scales(
            const attr_t &attr, const void *vals, int64_t count, int mask = -1);

    int prepare_binary_post_op_mds(
            const attr_t &attr, int ndims, const dnnl_dims_t dims);

    entry_t get(int arg) const {
        const auto it = entries.find(arg);
        return it == entries.end() ? entry_t() : it->second;
    }

    dnnl_memory_desc_t get_md(int arg) const {
        const auto it = mds.find(arg);
        return it == mds.end() ? dnnl_memory_desc_t() : it->second;
    }

private:
    void insert(
            int arg, const void *vals, int64_t count, int mask, bool runtime) {
        entries.insert(
                std::make_pair(arg, entry_t(vals, count, mask, runtime)));
    }

    std::map<int, entry_t> entries;
    std::map<int, dnnl_memory_desc_t> mds;
};

struct engine_t {
    engine_t(dnnl_engine_kind_t engine_kind);
    ~engine_t();
    operator dnnl_engine_t() const { return engine_; }

private:
    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(engine_t);
    dnnl_engine_t engine_;
};

struct stream_t {
    stream_t(dnnl_engine_t engine);
    ~stream_t();
    operator dnnl_stream_t() const { return stream_; }

private:
    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(stream_t);
    dnnl_stream_t stream_;
};

std::ostream &dump_global_params(std::ostream &s);

// Validates a tag/meta-tag.
int check_tag(const std::string &tag_, bool check_enum_tags_only = false);

// Validates a tag in abc notation.
int check_abc_tag(const std::string &tag, bool check_enum_tags_only = false);

// Converts a tag/meta-tag to abc notation.
std::string normalize_tag(const std::string &tag, int ndims = -1);

dnnl_primitive_attr_t create_dnnl_attr(
        const attr_t &attr, const attr_args_t &attr_args);

dnnl_engine_kind_t str2engine_kind(const char *str);
dnnl_scratchpad_mode_t str2scratchpad_mode(const char *str);

void maybe_oscale(const attr_t &attr, float &d, float *scales, int64_t oc);
void maybe_zero_point(const attr_t &attr, float &d, const int32_t *zero_points,
        int64_t c, int arg, bool opposite_zero_point = false);
float compute_eltwise_fwd(attr_t::post_ops_t::kind_t kind, float src,
        float scale, float alpha, float beta);
float compute_eltwise_bwd(attr_t::post_ops_t::kind_t kind, float d_dst,
        float src, float alpha, float beta);
float compute_binary(attr_t::post_ops_t::kind_t kind, float src0, float src1);
void maybe_post_ops(const attr_t &attr, float &val, float sum_val,
        const std::vector<float> &v_binary_vals);
inline void maybe_post_ops(
        const attr_t &attr, float &val, float sum_val = 0.f) {
    maybe_post_ops(attr, val, sum_val, std::vector<float>());
}
#endif
