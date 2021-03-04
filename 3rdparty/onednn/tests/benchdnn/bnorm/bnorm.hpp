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

#ifndef BNORM_HPP
#define BNORM_HPP

#include <assert.h>
#include <limits.h>
#include <stdint.h>

#include <iostream>
#include <string>

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
#include "dnnl_memory.hpp"
#include "perf_report.hpp"

namespace bnorm {

enum check_alg_t { ALG_0, ALG_1, ALG_AUTO };
check_alg_t str2check_alg(const char *str);
const char *check_alg2str(check_alg_t alg);

using flags_t = unsigned;
const flags_t NONE = dnnl_normalization_flags_none;
const flags_t GLOB_STATS = dnnl_use_global_stats;
const flags_t USE_SCALESHIFT = dnnl_use_scaleshift;
const flags_t FUSE_NORM_RELU = dnnl_fuse_norm_relu;
flags_t str2flags(const char *str);
std::string flags2str(flags_t flags);

struct desc_t {
    int64_t mb, ic, id, ih, iw;
    float eps;
    const char *name;
    int ndims;
};
int str2desc(desc_t *desc, const char *str);
std::ostream &operator<<(std::ostream &s, const desc_t &d);

struct settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    desc_t desc {};

    std::vector<dir_t> dir {FWD_D};
    std::vector<dnnl_data_type_t> dt {dnnl_f32};
    std::vector<std::string> tag {tag::abx};
    std::vector<flags_t> flags {NONE};
    std::vector<int64_t> mb {0};
    std::vector<bool> inplace {false};
    std::vector<attr_t::post_ops_t> post_ops {attr_t::post_ops_t()};
    std::vector<dnnl_scratchpad_mode_t> scratchpad_mode {
            dnnl_scratchpad_mode_library};
    attr_t attr = {};
    check_alg_t check_alg = ALG_AUTO;
    bool debug_check_ws = false;
    const char *pattern = NULL;

    const char *perf_template_csv
            = "perf,%engine%,%impl%,%name%,%dir%,%dt%,%tag%,%attr%,%flags%,%"
              "DESC%,%-time%,%0time%";
    const char *perf_template_def
            = "perf,%engine%,%impl%,%name%,%prb%,%-time%,%0time%";
    const char *perf_template = perf_template_def;

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t : public desc_t {
    prb_t(const desc_t &desc, int64_t mb, dir_t dir, dnnl_data_type_t dt,
            const std::string &tag, flags_t flags, bool inplace,
            const attr_t &attr, check_alg_t check_alg, bool debug_check_ws)
        : desc_t(desc)
        , check_alg(check_alg)
        , debug_check_ws(debug_check_ws)
        , dir(dir)
        , dt(dt)
        , tag(tag)
        , flags(flags)
        , inplace(inplace)
        , attr(attr) {
        if (mb) this->mb = mb;
    }
    ~prb_t() {}

    check_alg_t check_alg;
    bool debug_check_ws;

    dir_t dir;
    dnnl_data_type_t dt;
    std::string tag;
    flags_t flags;
    bool inplace;
    attr_t attr;

    bool need_ws() const {
        return (flags & FUSE_NORM_RELU) && !(dir & FLAG_INF);
    }
};
std::ostream &operator<<(std::ostream &s, const prb_t &prb);

struct perf_report_t : public base_perf_report_t {
    using base_perf_report_t::base_perf_report_t;

    void report(const prb_t *prb, const res_t *res, const char *prb_str) {
        p_ = prb;
        tag_ = normalize_tag(p_->tag, p_->ndims);
        base_report(res, prb_str);
    }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const desc_t &>(*p_);
    }

    void dump_desc_csv(std::ostream &s) const override {
        s << p_->mb << ',' << p_->ic << ',' << p_->id << ',' << p_->ih << ','
          << p_->iw << ',' << p_->eps;
    }

    void dump_flags(std::ostream &s) const override {
        s << flags2str(p_->flags);
    }

    const attr_t *attr() const override { return &p_->attr; }
    const char *name() const override { return p_->name; }
    const dir_t *dir() const override { return &p_->dir; }
    const dnnl_data_type_t *dt() const override { return &p_->dt; }
    const std::string *tag() const override { return &tag_; }

private:
    const prb_t *p_ = NULL;
    std::string tag_;
};

/* some extra control parameters which shouldn't be placed in prb_t */

inline size_t data_off(const prb_t *prb, int64_t mb, int64_t c, int64_t d,
        int64_t h, int64_t w) {
    return (((mb * prb->ic + c) * prb->id + d) * prb->ih + h) * prb->iw + w;
}

inline void inv_data_off(const prb_t *prb, size_t off, int64_t &mb, int64_t &c,
        int64_t &d, int64_t &h, int64_t &w) {
    w = off % prb->iw;
    off /= prb->iw;
    h = off % prb->ih;
    off /= prb->ih;
    d = off % prb->id;
    off /= prb->id;
    c = off % prb->ic;
    off /= prb->ic;
    mb = off % prb->mb;
    off /= prb->mb;
    assert(off == 0);
}

void compute_ref_fwd(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &mean, const dnn_mem_t &var, const dnn_mem_t &ss,
        dnn_mem_t &ws, dnn_mem_t &dst, dnn_mem_t &src_hat);
void compute_ref_bwd(const prb_t *prb, const dnn_mem_t &src_hat,
        const dnn_mem_t &var, const dnn_mem_t &d_dst, const dnn_mem_t &ss,
        const dnn_mem_t &ws, dnn_mem_t &d_src, dnn_mem_t &d_ss);

int doit(const prb_t *prb, res_t *res);
int bench(int argc, char **argv);

} // namespace bnorm

#endif
