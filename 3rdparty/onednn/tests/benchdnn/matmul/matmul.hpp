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

#ifndef MATMUL_HPP
#define MATMUL_HPP

#include <algorithm>
#include <bitset>
#include <iostream>
#include <numeric>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "perf_report.hpp"

namespace matmul {

typedef struct dt_conf_t {
    dnnl_data_type_t dt;
    double min, max; /* representative */
    double f_min, f_max; /* fill range */
    int f_base; /* fill base, use 0 */
    double f_sparsity; /* amount of non-zeros, default 0.25 */
    double f_scale; /* fill scale, scaling factor for integer generated data */
    double eps; /* acceptable error */
} _dt_conf_t[DAT_TOTAL];

typedef std::bitset<DNNL_MAX_NDIMS> dims_mask_t;
extern const _dt_conf_t conf_f32;

const int64_t LD_GOOD = INT64_MAX;
const int64_t LD_NONE = INT64_MAX - 1;

struct desc_t {
    desc_t() : is_legacy_desc(false), name(nullptr) {}
    std::vector<dims_t> sdims;
    bool is_legacy_desc = false;
    const char *name = nullptr;
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

    std::vector<const dt_conf_t *> cfg {conf_f32};
    std::vector<std::string> stag {tag::abx}, wtag {tag::abx}, dtag {tag::abx};
    std::vector<int64_t> ld_src {LD_NONE}, ld_wei {LD_NONE}, ld_dst {LD_NONE};
    std::vector<bool> runtime_mb {false}, runtime_m {false}, runtime_n {false},
            runtime_k {false};
    std::vector<dnnl_data_type_t> bia_dt {dnnl_data_type_undef};
    std::vector<int> bia_mask {2};
    std::vector<std::vector<dims_mask_t>> rt_dims_masks {{}};
    std::vector<attr_t::scale_t> oscale {attr_t::scale_t()};
    std::vector<attr_t::zero_points_t> zero_points {attr_t::zero_points_t()};
    std::vector<attr_t::post_ops_t> post_ops {attr_t::post_ops_t()};
    std::vector<dnnl_scratchpad_mode_t> scratchpad_mode {
            dnnl_scratchpad_mode_library};
    attr_t attr = {};

    const char *perf_template_csv
            = "perf,%engine%,%impl%,%name%,%cfg%,%attr%,%DESC%,"
              "%Gops%,%Gfreq%,%-time%,%-Gflops%,%0time%,%0Gflops%";
    const char *perf_template_def
            = "perf,%engine%,%impl%,%name%,%prb%,%Gops%,%Gfreq%,%-time%,%-"
              "Gflops%,%0time%,%0Gflops%";
    const char *perf_template = perf_template_def;

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t : public desc_t {
    prb_t(const desc_t &desc, const dt_conf_t *cfg, const std::string &stag,
            const std::string &wtag, const std::string &dtag, int64_t ld_src,
            int64_t ld_wei, int64_t ld_dst, bool runtime_mb, bool runtime_m,
            bool runtime_n, bool runtime_k, dnnl_data_type_t bia_dt,
            int bia_mask, const std::vector<dims_mask_t> &rt_dims_masks,
            const attr_t &attr)
        : desc_t(desc)
        , cfg(cfg)
        , stag(stag)
        , wtag(wtag)
        , dtag(dtag)
        , ld_src(ld_src)
        , ld_wei(ld_wei)
        , ld_dst(ld_dst)
        , runtime_mb(runtime_mb)
        , runtime_m(runtime_m)
        , runtime_n(runtime_n)
        , runtime_k(runtime_k)
        , bia_dt(bia_dt)
        , bia_mask(bia_mask)
        , rt_dims_masks(rt_dims_masks)
        , attr(attr)
        , scales(NULL) {

        this->rt_dims_masks.resize(2);
        if (IMPLICATION(src_runtime_dim_mask().none()
                            && weights_runtime_dim_mask().none(),
                    runtime_mb || runtime_m || runtime_n || runtime_k)) {
            // legacy desc_t
            set_runtime_dims_masks();
        }

        const auto &srcdims = src_dims();
        const auto &weidims = weights_dims();
        ndims = (int)srcdims.size();
        m = srcdims[ndims - 2];
        k = srcdims.back();
        n = weidims.back();

        init_dst();
        const auto &dstdims = dst_dims();
        const auto nelems = std::accumulate(dstdims.begin(), dstdims.end(),
                (dnnl_dim_t)1, std::multiplies<dnnl_dim_t>());
        ops = 2. * nelems * k;

        generate_oscales();
    }
    ~prb_t() {
        if (scales) zfree(scales);
    }

    int m, n, k;
    const dt_conf_t *cfg;
    int ndims;
    std::string stag, wtag, dtag;
    int64_t ld_src, ld_wei, ld_dst;
    bool runtime_mb, runtime_m, runtime_n, runtime_k;
    dnnl_data_type_t bia_dt;
    int bia_mask;
    std::vector<dims_mask_t> rt_dims_masks;

    attr_t attr;

    double ops;
    float *scales;

    const dims_t &src_dims() const { return sdims[0]; }
    const dims_t &weights_dims() const { return sdims[1]; }
    const dims_t &dst_dims() const { return sdims[2]; }

    const dims_mask_t &src_runtime_dim_mask() const { return rt_dims_masks[0]; }
    const dims_mask_t &weights_runtime_dim_mask() const {
        return rt_dims_masks[1];
    }
    const dims_mask_t &dst_runtime_dim_mask() const { return rt_dims_masks[2]; }

    int src_broadcast_mask() const { return get_broadcast_mask(src_dims()); }

    int weights_broadcast_mask() const {
        return get_broadcast_mask(weights_dims());
    }

    int bias_broadcast_mask() const { return bia_mask; }

    void generate_oscales();

    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(prb_t);

private:
    int get_broadcast_mask(const dims_t &dims_idx) const {
        const dims_t &dims_dst = this->dst_dims();

        int broadcast_mask = 0;
        for (int d = 0; d < ndims; ++d)
            broadcast_mask += dims_dst[d] == dims_idx[d] ? (1 << d) : 0;

        return broadcast_mask;
    }

    void init_dst_dims() {
        if (sdims.size() > 2) return;
        sdims.resize(3);
        auto &dst_dims = sdims.back();
        dst_dims.resize(ndims);

        for (int i = 0; i < ndims - 2; ++i) {
            sdims.back()[i] = MAX2(sdims[0][i], sdims[1][i]);
        }
        sdims.back()[ndims - 2] = m;
        sdims.back()[ndims - 1] = n;
    }

    void init_dst_rt_dims_mask() {
        if (rt_dims_masks.size() > 2) return;

        const auto &src_rt_dim_mask = src_runtime_dim_mask();
        const auto &wei_rt_dim_mask = weights_runtime_dim_mask();
        dims_mask_t dst_rt_dim_mask;

        for (int i = 0; i < ndims - 2; ++i) {
            dst_rt_dim_mask[i] = src_rt_dim_mask[i] | wei_rt_dim_mask[i];
        }

        // m, n mask
        dst_rt_dim_mask[ndims - 2] = src_rt_dim_mask[ndims - 2];
        dst_rt_dim_mask[ndims - 1] = wei_rt_dim_mask[ndims - 1];

        rt_dims_masks.push_back(dst_rt_dim_mask);
    }

    void init_dst() {
        init_dst_dims();
        init_dst_rt_dims_mask();
    }

    // used only for legacy desc support
    void set_runtime_dims_masks() {
        // here we only set src and wei masks. dst mask is computed in init_dst
        const auto ndims = sdims[0].size();
        auto &src_mask = rt_dims_masks[0];
        auto &wei_mask = rt_dims_masks[1];
        if (runtime_mb && ndims == 3) { // else silently ignore
            src_mask[0] = true;
            wei_mask[0] = true;
        }

        if (runtime_m) src_mask[ndims - 2] = true;
        if (runtime_n) wei_mask[ndims - 1] = true;
        if (runtime_k) {
            src_mask[ndims - 1] = true;
            wei_mask[ndims - 2] = true;
        }
    }
};
std::ostream &operator<<(std::ostream &s, const prb_t &prb);

/* some extra control parameters which shouldn't be placed in prb_t */

const dt_conf_t *str2cfg(const char *str);
std::ostream &operator<<(std::ostream &s, const dt_conf_t *cfg);

struct perf_report_t : public base_perf_report_t {
    using base_perf_report_t::base_perf_report_t;

    void report(const prb_t *prb, const res_t *res, const char *prb_str) {
        p_ = prb;
        stag_ = {normalize_tag(p_->stag, p_->ndims)};
        wtag_ = normalize_tag(p_->wtag, p_->ndims);
        dtag_ = normalize_tag(p_->dtag, p_->ndims);
        base_report(res, prb_str);
    }

    void dump_cfg(std::ostream &s) const override { s << p_->cfg; }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const desc_t &>(*p_);
    }

    void dump_desc_csv(std::ostream &s) const override {
        s << static_cast<const desc_t &>(*p_);
    }

    double ops() const override { return p_->ops; }
    const attr_t *attr() const override { return &p_->attr; }
    const char *name() const override { return p_->name; }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const std::string *wtag() const override { return &wtag_; }
    const std::string *dtag() const override { return &dtag_; }

private:
    const prb_t *p_ = NULL;
    std::vector<std::string> stag_;
    std::string wtag_, dtag_;
};

inline int64_t src_off_f(const prb_t *prb, int64_t mb, int64_t m, int64_t k) {
    return (mb * prb->m + m) * prb->k + k;
}

inline int64_t wei_off_f(const prb_t *prb, int64_t mb, int64_t k, int64_t n) {
    return (mb * prb->k + k) * prb->n + n;
}

inline int64_t dst_off_f(const prb_t *prb, int64_t mb, int64_t m, int64_t n) {
    return (mb * prb->m + m) * prb->n + n;
}

void compute_ref(const engine_t &engine_tgt, const prb_t *prb, dnn_mem_t &src_m,
        dnn_mem_t &wei_m, dnn_mem_t &bia_m,
        const std::vector<dnn_mem_t> &binary_po, dnn_mem_t &dst_m);

int doit(const prb_t *prb, res_t *res);

int bench(int argc, char **argv);

} // namespace matmul

#endif
