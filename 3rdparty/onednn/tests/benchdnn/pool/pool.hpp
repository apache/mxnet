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

#ifndef POOL_HPP
#define POOL_HPP

#include <assert.h>
#include <limits.h>
#include <stdint.h>

#include <iostream>

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "perf_report.hpp"

namespace pool {

enum alg_t { MAX, AVG_NP, AVG_P };
alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);
dnnl_alg_kind_t alg2alg_kind(alg_t alg);

struct desc_t {
    int64_t mb, ic;
    int64_t id, ih, iw;
    int64_t od, oh, ow;
    int64_t kd, kh, kw;
    int64_t dd, dh, dw;
    int64_t sd, sh, sw;
    int64_t pd, ph, pw;
    int64_t pd_r, ph_r, pw_r; // End side padding for each dimension

    const char *name;
    int ndims;

    // Initialize dependent opposite-side paddings values from the shape
    // parameters
    void init_pad_r() {
        pw_r = opp_pad(iw, ow, kw, dw, sw, pw);
        ph_r = opp_pad(ih, oh, kh, dh, sh, ph);
        pd_r = opp_pad(id, od, kd, dd, sd, pd);
    }

private:
    int64_t opp_pad(
            int64_t i, int64_t o, int64_t k, int64_t d, int64_t s, int64_t p) {
        return (o - 1) * s - i + ((k - 1) * (d + 1) + 1) - p;
    }
};

int str2desc(desc_t *desc, const char *str);
std::ostream &operator<<(std::ostream &s, const desc_t &d);

/** configuration structure, that controls initial data filling + error check
 *
 * dt defines pooling precision
 *
 * for each type (SRC and DST) the values are filled as follows:
 * if (rand() > f_sparsity) then:
 *     v <-- f_base // it is guaranteed each kernel window
 *                  // has at least one non-zero element
 * else:
 *     v <-- f_min + rand() * f_step % (f_max - f_min)
 *
 * on final check the resulting values should be in [min .. max] range, the
 * relative difference should not exceed eps
 */
typedef struct dt_conf_t {
    dnnl_data_type_t dt;
    double min, max; /* representative */
    int f_min, f_max; /* fill range */
    double eps; /* acceptable error */
} _dt_conf_t[DAT_TOTAL];

extern const _dt_conf_t conf_f32;

const dt_conf_t *str2cfg(const char *str);
std::ostream &operator<<(std::ostream &s, const dt_conf_t *cfg);

struct settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    desc_t desc {};

    std::vector<dir_t> dir {FWD_D};
    std::vector<const dt_conf_t *> cfg {conf_f32};
    std::vector<std::string> tag {tag::abx};
    std::vector<alg_t> alg {MAX};
    std::vector<int64_t> mb {0};
    std::vector<attr_t::post_ops_t> post_ops {attr_t::post_ops_t()};
    std::vector<dnnl_scratchpad_mode_t> scratchpad_mode {
            dnnl_scratchpad_mode_library};

    const char *perf_template_csv
            = "perf,%engine%,%impl%,%name%,%dir%,%cfg%,%tag%,%alg%,%DESC%,%-"
              "time%,%0time%";
    const char *perf_template_def
            = "perf,%engine%,%impl%,%name%,%prb%,%-time%,%0time%";
    const char *perf_template = perf_template_def;

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t : public desc_t {
    prb_t(const desc_t &desc, dir_t dir, const dt_conf_t *cfg,
            const std::string &tag, alg_t alg, const attr_t &attr,
            int64_t mb = 0)
        : desc_t(desc), dir(dir), cfg(cfg), tag(tag), alg(alg), attr(attr) {
        if (mb) this->mb = mb;
    }
    ~prb_t() {}

    dir_t dir;
    const dt_conf_t *cfg;
    std::string tag;
    alg_t alg;
    attr_t attr;

    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(prb_t);
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

    void dump_cfg(std::ostream &s) const override { s << p_->cfg; }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const desc_t &>(*p_);
    }

    void dump_desc_csv(std::ostream &s) const override {
        s << p_->mb << ','

          << p_->ic << ',' << p_->id << ',' << p_->ih << ',' << p_->iw << ','

          << p_->od << ',' << p_->oh << ',' << p_->ow << ','

          << p_->kd << ',' << p_->kh << ',' << p_->kw << ','

          << p_->sd << ',' << p_->sh << ',' << p_->sw << ','

          << p_->pd << ',' << p_->ph << ',' << p_->pw << ','

          << p_->dd << ',' << p_->dh << ',' << p_->dw;
    }

    const char *name() const override { return p_->name; }
    const dir_t *dir() const override { return &p_->dir; }
    const std::string *tag() const override { return &tag_; }

private:
    const prb_t *p_ = NULL;
    std::string tag_;
};

inline int64_t src_off_f(const prb_t *prb, int64_t mb, int64_t ic, int64_t id,
        int64_t ih, int64_t iw) {
    return (((mb * prb->ic + ic) * prb->id + id) * prb->ih + ih) * prb->iw + iw;
}

inline void inv_src_off_f(const prb_t *prb, int64_t off, int64_t &mb,
        int64_t &ic, int64_t &id, int64_t &ih, int64_t &iw) {
    iw = off % prb->iw;
    off /= prb->iw;
    ih = off % prb->ih;
    off /= prb->ih;
    id = off % prb->id;
    off /= prb->id;
    ic = off % prb->ic;
    off /= prb->ic;
    mb = off % prb->mb;
    off /= prb->mb;
    assert(off == 0);
}

inline int64_t dst_off_f(const prb_t *prb, int64_t mb, int64_t ic, int64_t od,
        int64_t oh, int64_t ow) {
    return (((mb * prb->ic + ic) * prb->od + od) * prb->oh + oh) * prb->ow + ow;
}

inline void inv_dst_off_f(const prb_t *prb, int64_t off, int64_t &mb,
        int64_t &ic, int64_t &od, int64_t &oh, int64_t &ow) {
    ow = off % prb->ow;
    off /= prb->ow;
    oh = off % prb->oh;
    off /= prb->oh;
    od = off % prb->od;
    off /= prb->od;
    ic = off % prb->ic;
    off /= prb->ic;
    mb = off % prb->mb;
    off /= prb->mb;
    assert(off == 0);
}

inline int64_t ker_off_f(const prb_t *prb, int64_t kd, int64_t kh, int64_t kw) {
    return (kd * prb->kh + kh) * prb->kw + kw;
}

inline int64_t get_num_summands(
        const prb_t *prb, int64_t d, int64_t h, int64_t w) {
    const int64_t ID = prb->id, IH = prb->ih, IW = prb->iw;
    const int64_t KD = prb->kd, KH = prb->kh, KW = prb->kw;
    const int64_t DD = prb->dd, DH = prb->dh, DW = prb->dw;
    const int64_t PD = prb->pd, PH = prb->ph, PW = prb->pw;
    const int64_t SD = prb->sd, SH = prb->sh, SW = prb->sw;

    auto id_start = d * SD - PD;
    auto ih_start = h * SH - PH;
    auto iw_start = w * SW - PW;
    auto id_end = d * SD - PD + (KD - 1) * DD + KD;
    auto ih_end = h * SH - PH + (KH - 1) * DH + KH;
    auto iw_end = w * SW - PW + (KW - 1) * DW + KW;

    auto id_start_excluded
            = id_start < 0 ? (0 - id_start - 1) / (DD + 1) + 1 : 0;
    auto ih_start_excluded
            = ih_start < 0 ? (0 - ih_start - 1) / (DH + 1) + 1 : 0;
    auto iw_start_excluded
            = iw_start < 0 ? (0 - iw_start - 1) / (DW + 1) + 1 : 0;
    auto id_end_excluded = id_end > ID ? (id_end - ID - 1) / (DD + 1) + 1 : 0;
    auto ih_end_excluded = ih_end > IH ? (ih_end - IH - 1) / (DH + 1) + 1 : 0;
    auto iw_end_excluded = iw_end > IW ? (iw_end - IW - 1) / (DW + 1) + 1 : 0;

    return prb->alg == AVG_P ? KD * KH * KW
                             : (KD - id_start_excluded - id_end_excluded)
                    * (KH - ih_start_excluded - ih_end_excluded)
                    * (KW - iw_start_excluded - iw_end_excluded);
}

void compute_ref_fwd(const prb_t *prb, const dnn_mem_t &src,
        const std::vector<dnn_mem_t> &binary_po, dnn_mem_t &dst, dnn_mem_t &ws);
void compute_ref_bwd(const prb_t *prb, dnn_mem_t &diff_src,
        const dnn_mem_t &diff_dst, const dnn_mem_t &ws);

int compare_src(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res);
int compare_dst(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res);
int fill_src(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res);
int fill_dst(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res);
int fill_ws(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res);

int doit(const prb_t *prb, res_t *res);
int bench(int argc, char **argv);

} // namespace pool

#endif
