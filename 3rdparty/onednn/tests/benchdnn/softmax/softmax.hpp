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

#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include <iostream>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "perf_report.hpp"

namespace softmax {

enum alg_t { UNDEF, SOFTMAX, LOGSOFTMAX };
alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);
dnnl_alg_kind_t alg2alg_kind(alg_t alg);

struct settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    dims_t dims;

    std::vector<dir_t> dir {FWD_D};
    std::vector<dnnl_data_type_t> dt {dnnl_f32};
    std::vector<std::string> tag {tag::abx};
    std::vector<alg_t> alg {SOFTMAX};
    std::vector<int> axis {1};
    std::vector<int64_t> mb {0};
    std::vector<bool> inplace {false};
    std::vector<dnnl_scratchpad_mode_t> scratchpad_mode {
            dnnl_scratchpad_mode_library};

    const char *perf_template_csv
            = "perf,%engine%,%impl%,%dir%,%dt%,%tag%,%alg%,%axis%,%DESC%,%-"
              "time%,%0time%";
    const char *perf_template_def
            = "perf,%engine%,%impl%,%prb%,%-time%,%0time%";
    const char *perf_template = perf_template_def;

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t {
    prb_t(const dims_t &dims, dir_t dir, dnnl_data_type_t dt,
            const std::string &tag, alg_t alg, int axis, bool inplace,
            const attr_t &attr, int64_t mb = 0)
        : dims(dims)
        , dir(dir)
        , dt(dt)
        , tag(tag)
        , alg(alg)
        , axis(axis)
        , inplace(inplace)
        , attr(attr)
        , ndims((int)dims.size()) {
        if (mb) this->dims[0] = mb;
    }
    ~prb_t() {}

    dims_t dims;
    dir_t dir;
    dnnl_data_type_t dt;
    std::string tag;
    alg_t alg;
    int axis;
    bool inplace;
    attr_t attr;
    int ndims;
};
std::ostream &operator<<(std::ostream &s, const prb_t &prb);

struct perf_report_t : public base_perf_report_t {
    using base_perf_report_t::base_perf_report_t;

    void report(const prb_t *prb, const res_t *res, const char *prb_str) {
        p_ = prb;
        tag_ = normalize_tag(p_->tag, p_->ndims);
        base_report(res, prb_str);
    }

    void dump_alg(std::ostream &s) const override { s << alg2str(p_->alg); }

    void dump_desc(std::ostream &s) const override { s << p_->dims; }

    void dump_desc_csv(std::ostream &s) const override { s << p_->dims; }

    const int *axis() const override { return &p_->axis; }
    const dir_t *dir() const override { return &p_->dir; }
    const dnnl_data_type_t *dt() const override { return &p_->dt; }
    const std::string *tag() const override { return &tag_; }

private:
    const prb_t *p_ = NULL;
    std::string tag_;
};

inline void map_off_to_mb_ic(
        const prb_t *prb, int64_t off, int64_t &mb, int64_t &ic) {
    for (int i = prb->ndims - 1; i > 1; i--)
        off /= prb->dims[i];

    ic = off % prb->dims[1];
    off /= prb->dims[1];
    mb = off % prb->dims[0];
    off /= prb->dims[0];
    assert(off == 0);
}

inline void get_sizes(const prb_t *prb, int64_t &outer_size,
        int64_t &inner_size, int64_t &axis_size) {
    outer_size = inner_size = axis_size = 1;
    for (int i = 0; i < prb->axis; i++)
        outer_size *= prb->dims[i];
    for (int i = prb->axis + 1; i < prb->ndims; i++)
        inner_size *= prb->dims[i];
    axis_size = prb->dims[prb->axis];
}

void compute_ref_fwd(const prb_t *prb, const dnn_mem_t &src, dnn_mem_t &dst);
void compute_ref_bwd(const prb_t *prb, const dnn_mem_t &dst,
        const dnn_mem_t &diff_dst, dnn_mem_t &diff_src);

int doit(const prb_t *prb, res_t *res);
int bench(int argc, char **argv);

} // namespace softmax

#endif
