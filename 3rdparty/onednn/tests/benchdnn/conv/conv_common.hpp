/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#ifndef CONV_COMMON_HPP
#define CONV_COMMON_HPP

#include <assert.h>
#include <limits.h>
#include <stdint.h>

#include <iostream>

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "perf_report.hpp"

namespace conv {

enum alg_t { DIRECT, WINO, AUTO };
alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);
alg_t alg_kind2alg(dnnl_alg_kind_t alg);

struct desc_t {
    int64_t g, mb;
    int64_t ic, id, ih, iw;
    int64_t oc, od, oh, ow;
    int64_t kd, kh, kw;
    int64_t sd, sh, sw;
    int64_t pd, ph, pw;
    int64_t pd_r, ph_r, pw_r; // End side padding for each dimension
    int64_t dd, dh, dw;
    bool has_groups;

    const char *name;
    int ndims;

    // Initialize dependent opposite-side paddings values
    // from the shape parameters
    void init_pad_r(bool is_deconv) {
        pw_r = opp_pad(is_deconv, iw, ow, kw, sw, pw, dw);
        ph_r = opp_pad(is_deconv, ih, oh, kh, sh, ph, dh);
        pd_r = opp_pad(is_deconv, id, od, kd, sd, pd, dd);
    }

private:
    int64_t opp_pad(bool is_deconv, int64_t i, int64_t o, int64_t k, int64_t s,
            int64_t p, int64_t d) {
        return is_deconv ? (i - 1) * s - o + ((k - 1) * (d + 1) + 1) - p
                         : (o - 1) * s - i + ((k - 1) * (d + 1) + 1) - p;
    }
};

int str2desc(desc_t *desc, const char *str, bool is_deconv);
std::ostream &operator<<(std::ostream &s, const desc_t &d);

/** configuration structure, that controls initial data filling + error check
 *
 * dt defines convolution precision
 *
 * for each type (SRC, WEI, BIA, and DST) the values are filled as follows:
 * if (rand() > f_sparsity) then:
 *     v <-- f_base // it is guaranteed each kernel window
 *                  // has at least one non-zero element
 * else:
 *     v <-- f_min + rand() * f_step % (f_max - f_min)
 *
 *
 * on final check the resulting values should be in [min .. max] range, the
 * relative difference should not exceed eps
 */
typedef struct dt_conf_t {
    dnnl_data_type_t dt;
    double min, max; /* representative */
    int f_min, f_max; /* fill range */
    int f_base; /* fill base, use 0 */
    int f_step; /* fill step, use 1 */
    double f_sparsity; /* amount of non-zeros, default 0.25 */
    double eps; /* acceptable error */
} _dt_conf_t[DAT_TOTAL];

extern const _dt_conf_t conf_f32;

const dt_conf_t *str2cfg(const char *str);
std::ostream &operator<<(std::ostream &s, const dt_conf_t *cfg);
const dt_conf_t *auto_cfg(const alg_t alg, const dt_conf_t *cfg);

struct settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    desc_t desc {};

    std::vector<dir_t> dir {FWD_B};
    std::vector<const dt_conf_t *> cfg {conf_f32};
    std::vector<std::string> stag {tag::any}, wtag {tag::any}, dtag {tag::any};
    std::vector<int64_t> mb {0};
    std::vector<alg_t> alg {DIRECT};
    std::vector<attr_t::scale_t> oscale {attr_t::scale_t()};
    std::vector<attr_t::zero_points_t> zero_points {attr_t::zero_points_t()};
    std::vector<attr_t::post_ops_t> post_ops {attr_t::post_ops_t()};
    std::vector<dnnl_scratchpad_mode_t> scratchpad_mode {
            dnnl_scratchpad_mode_library};
    attr_t attr = {};
    const char *pattern = NULL;

    const char *perf_template_csv
            = "perf,%engine%,%impl%,%name%,%dir%,%cfg%,%alg%,%attr%,%DESC%,"
              "%Gops%,%Gfreq%,%-time%,%-Gflops%,%0time%,%0Gflops%";
    const char *perf_template_def
            = "perf,%engine%,%impl%,%name%,%prb%,%Gops%,%Gfreq%,%-time%,%-"
              "Gflops%,%0time%,%0Gflops%";
    const char *perf_template = perf_template_def;

    void reset() { *this = settings_t(perf_template); }
};

// moved out of prb_t to support fusion
float *generate_oscales(const attr_t::scale_t &oscale, int N);
int32_t *generate_zero_points(
        int arg, const attr_t::zero_points_t &zero_points, int N);

struct prb_t : public desc_t {
    prb_t(const desc_t &desc, dir_t dir, const dt_conf_t *cfg,
            const std::string &stag, const std::string &wtag,
            const std::string &dtag, alg_t alg, const attr_t &attr,
            int64_t mb = 0, bool is_deconv = false)
        : desc_t(desc)
        , dir(dir)
        , cfg(cfg)
        , stag(stag)
        , wtag(wtag)
        , dtag(dtag)
        , alg(alg)
        , attr(attr)
        , ops(0)
        , scales(NULL)
        , src_zp(NULL)
        , dst_zp(NULL)
        , is_deconv(is_deconv) {
        if (mb) this->mb = mb;
        count_ops();
        scales = generate_oscales(attr.oscale, oc);
        src_zp = generate_zero_points(DNNL_ARG_SRC, attr.zero_points, ic);
        dst_zp = generate_zero_points(DNNL_ARG_DST, attr.zero_points, oc);
    }
    ~prb_t() {
        if (scales) zfree(scales);
        if (src_zp) zfree(src_zp);
        if (dst_zp) zfree(dst_zp);
    }

    dir_t dir;
    const dt_conf_t *cfg;
    std::string stag, wtag, dtag;
    alg_t alg;
    attr_t attr;

    double ops;
    float *scales;
    int32_t *src_zp;
    int32_t *dst_zp;
    bool is_deconv;

    void count_ops();

    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(prb_t);
};
std::ostream &operator<<(std::ostream &s, const prb_t &prb);

struct perf_report_t : public base_perf_report_t {
    using base_perf_report_t::base_perf_report_t;

    void report(const prb_t *prb, const res_t *res, const char *prb_str) {
        p_ = prb;
        stag_ = {normalize_tag(p_->stag, p_->ndims)};
        wtag_ = normalize_tag(p_->wtag, p_->ndims);
        dtag_ = normalize_tag(p_->dtag, p_->ndims);
        base_report(res, prb_str);
    }

    void dump_alg(std::ostream &s) const override { s << alg2str(p_->alg); }

    void dump_cfg(std::ostream &s) const override { s << p_->cfg; }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const desc_t &>(*p_);
    }

    void dump_desc_csv(std::ostream &s) const override {
        s << p_->g << ',' << p_->mb << ','

          << p_->ic << ',' << p_->id << ',' << p_->ih << ',' << p_->iw << ','

          << p_->oc << ',' << p_->od << ',' << p_->oh << ',' << p_->ow << ','

          << p_->kd << ',' << p_->kh << ',' << p_->kw << ','

          << p_->sd << ',' << p_->sh << ',' << p_->sw << ','

          << p_->pd << ',' << p_->ph << ',' << p_->pw << ','

          << p_->dd << ',' << p_->dh << ',' << p_->dw;
    }

    double ops() const override { return p_->ops; }
    const attr_t *attr() const override { return &p_->attr; }
    const char *name() const override { return p_->name; }
    const dir_t *dir() const override { return &p_->dir; }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const std::string *wtag() const override { return &wtag_; }
    const std::string *dtag() const override { return &dtag_; }

private:
    const prb_t *p_ = NULL;
    std::vector<std::string> stag_;
    std::string wtag_, dtag_;
};

inline int64_t src_off_f(const prb_t *prb, int64_t mb, int64_t g, int64_t ic,
        int64_t id, int64_t ih, int64_t iw) {
    return (((mb * prb->ic + g * prb->ic / prb->g + ic) * prb->id + id)
                           * prb->ih
                   + ih)
            * prb->iw
            + iw;
}

inline void inv_src_off_f(const prb_t *prb, int64_t off, int64_t &mb,
        int64_t &g, int64_t &ic, int64_t &id, int64_t &ih, int64_t &iw) {
    iw = off % prb->iw;
    off /= prb->iw;
    ih = off % prb->ih;
    off /= prb->ih;
    id = off % prb->id;
    off /= prb->id;
    ic = off % (prb->ic / prb->g);
    off /= (prb->ic / prb->g);
    g = off % prb->g;
    off /= prb->g;
    mb = off % prb->mb;
    off /= prb->mb;
    assert(off == 0);
}

inline int64_t wei_off_f(const prb_t *prb, int64_t g, int64_t oc, int64_t ic,
        int64_t kd, int64_t kh, int64_t kw) {
    return ((((g * prb->oc / prb->g + oc) * prb->ic / prb->g + ic) * prb->kd
                    + kd) * prb->kh
                   + kh)
            * prb->kw
            + kw;
}

inline void inv_wei_off_f(const prb_t *prb, int64_t off, int64_t &g,
        int64_t &oc, int64_t &ic, int64_t &kd, int64_t &kh, int64_t &kw) {
    kw = off % prb->kw;
    off /= prb->kw;
    kh = off % prb->kh;
    off /= prb->kh;
    kd = off % prb->kd;
    off /= prb->kd;
    ic = off % (prb->ic / prb->g);
    off /= (prb->ic / prb->g);
    oc = off % (prb->oc / prb->g);
    off /= (prb->oc / prb->g);
    g = off % prb->g;
    off /= prb->g;
    assert(off == 0);
}

inline int64_t bia_off_f(const prb_t *prb, int64_t g, int64_t oc) {
    return g * prb->oc / prb->g + oc;
}

inline void inv_bia_off_f(
        const prb_t *prb, int64_t off, int64_t &g, int64_t &oc) {
    oc = off % (prb->oc / prb->g);
    off /= (prb->oc / prb->g);
    g = off % prb->g;
    off /= prb->g;
    assert(off == 0);
}

inline int64_t dst_off_f(const prb_t *prb, int64_t mb, int64_t g, int64_t oc,
        int64_t od, int64_t oh, int64_t ow) {
    return (((mb * prb->oc + g * prb->oc / prb->g + oc) * prb->od + od)
                           * prb->oh
                   + oh)
            * prb->ow
            + ow;
}

inline void inv_dst_off_f(const prb_t *prb, int64_t off, int64_t &mb,
        int64_t &g, int64_t &oc, int64_t &od, int64_t &oh, int64_t &ow) {
    ow = off % prb->ow;
    off /= prb->ow;
    oh = off % prb->oh;
    off /= prb->oh;
    od = off % prb->od;
    off /= prb->od;
    oc = off % (prb->oc / prb->g);
    off /= (prb->oc / prb->g);
    g = off % prb->g;
    off /= prb->g;
    mb = off % prb->mb;
    off /= prb->mb;
    assert(off == 0);
}

float oscale(const prb_t *prb, int oc);

void compute_ref_fwd(const prb_t *prb, dnnl_primitive_t c_ref, dnn_mem_t &src_m,
        dnn_mem_t &wei_m, dnn_mem_t &bia_m,
        const std::vector<dnn_mem_t> &binary_po, dnn_mem_t &dst_m);
void compute_ref_bwd_d(const prb_t *prb, dnnl_primitive_t c_ref,
        dnn_mem_t &diff_src_m, dnn_mem_t &wei_m, dnn_mem_t &bia_m,
        const std::vector<dnn_mem_t> &binary_po, dnn_mem_t &diff_dst_m);
void compute_ref_bwd_w(const prb_t *prb, dnnl_primitive_t c_ref,
        dnn_mem_t &src_m, dnn_mem_t &diff_wei_m, dnn_mem_t &diff_bia_m,
        dnn_mem_t &diff_dst_m);

void compute_ref_direct_fwd(const prb_t *prb, dnn_mem_t &src_m,
        dnn_mem_t &wei_m, dnn_mem_t &bia_m,
        const std::vector<dnn_mem_t> &binary_po, dnn_mem_t &dst_m);
void compute_ref_direct_bwd_d(const prb_t *prb, dnn_mem_t &diff_src_m,
        dnn_mem_t &wei_m, dnn_mem_t &bia_m,
        const std::vector<dnn_mem_t> &binary_po, dnn_mem_t &diff_dst_m);
void compute_ref_direct_bwd_w(const prb_t *prb, dnn_mem_t &src_m,
        dnn_mem_t &diff_wei_m, dnn_mem_t &diff_bia_m, dnn_mem_t &diff_dst_m);

void compute_wino_ref_fwd(const prb_t *prb, dnn_mem_t &src_m, dnn_mem_t &wei_m,
        dnn_mem_t &bia_m, dnn_mem_t &dst_m);
void compute_wino_ref_bwd_d(const prb_t *prb, dnn_mem_t &idiff_src_m,
        dnn_mem_t &wei_m, dnn_mem_t &bia_m, dnn_mem_t &diff_dst_m);
void compute_wino_ref_bwd_w(const prb_t *prb, dnn_mem_t &src_m,
        dnn_mem_t &diff_wei_m, dnn_mem_t &diff_bia_m, dnn_mem_t &diff_dst_m);

int compare_src(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *res, bool final_compare = false);
int compare_wei(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *res, bool final_compare = false);
int compare_bia(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *res, bool final_compare = false);
int compare_dst(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *res, bool final_compare = false);
int fill_src(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res);
int fill_wei(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res);
int fill_bia(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res);
int fill_dst(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res);
double get_trust_nz_level(
        const prb_t *prb, data_kind_t kind, bool final_compare);

void compute_ref_bwd_bias(
        const prb_t *prb, dnn_mem_t &diff_bia_m, dnn_mem_t &diff_dst_m);
void compute_bias_fwd(const prb_t *prb, dnn_mem_t &bia_m, dnn_mem_t &dst_m);
void compute_ref_bwd_weights(const prb_t *prb, dnn_mem_t &src_m,
        dnn_mem_t &diff_wei_m, dnn_mem_t &diff_dst_m);

} // namespace conv

#endif
