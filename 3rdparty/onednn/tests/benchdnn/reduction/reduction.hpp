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

#ifndef REDUCTION_HPP
#define REDUCTION_HPP

#include "oneapi/dnnl/dnnl.hpp"

#include "dnn_types.hpp"
#include "dnnl_memory.hpp"
#include "perf_report.hpp"

namespace reduction {

enum alg_t {
    UNDEF,
    MIN,
    MAX,
    MUL,
    SUM,
    MEAN,
    NORM_LP_MAX,
    NORM_LP_SUM,
    NORM_LP_POWER_P_MAX,
    NORM_LP_POWER_P_SUM
};

alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);
dnnl_alg_kind_t alg2alg_kind(alg_t alg);

struct settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    std::vector<dims_t> dims;
    std::vector<dnnl_data_type_t> sdt {dnnl_f32};
    std::vector<dnnl_data_type_t> ddt {dnnl_f32};
    std::vector<std::string> stag {tag::abx};
    std::vector<std::string> dtag {tag::any};
    std::vector<alg_t> alg {alg_t::SUM};
    std::vector<float> p {1.0f}, eps {0.0f};
    std::vector<int64_t> mb {0};

    const char *perf_template_csv
            = "perf,%engine%,%impl%,%sdt%,%ddt%,%stag%,%dtag%,%alg%,%attr%,"
              "%DESC%,%-time%,%0time%";
    const char *perf_template_def
            = "perf,%engine%,%impl%,%prb%,%-time%,%0time%";
    const char *perf_template = perf_template_def;

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t {
    prb_t(const std::vector<dims_t> &dims, dnnl_data_type_t sdt,
            dnnl_data_type_t ddt, const std::string &stag,
            const std::string &dtag, alg_t alg, float p, float eps,
            const attr_t &attr)
        : src_dims(dims[0])
        , dst_dims(dims[1])
        , ndims(static_cast<int>(dims[0].size()))
        , sdt(sdt)
        , ddt(ddt)
        , stag(stag)
        , dtag(dtag)
        , alg(alg)
        , p(p)
        , eps(eps)
        , attr(attr) {}

    dims_t src_dims, dst_dims;
    int ndims;
    dnnl_data_type_t sdt, ddt;
    std::string stag, dtag;
    alg_t alg;
    float p, eps;
    attr_t attr;
};

std::ostream &operator<<(std::ostream &s, const prb_t &prb);

struct perf_report_t : public base_perf_report_t {
    using base_perf_report_t::base_perf_report_t;

    void report(const prb_t *prb, const res_t *r, const char *prb_str) {
        prb_ = prb;
        sdt_.push_back(prb_->sdt);
        stag_.push_back(normalize_tag(prb_->stag, prb_->ndims));
        dtag_ = normalize_tag(prb_->dtag, prb_->ndims);
        base_report(r, prb_str);
    }

    void dump_alg(std::ostream &s) const override { s << alg2str(prb_->alg); }

    void dump_desc(std::ostream &s) const override {
        s << prb_->src_dims << ":" << prb_->dst_dims;
    }

    const attr_t *attr() const override { return &prb_->attr; }
    const std::vector<dnnl_data_type_t> *sdt() const override { return &sdt_; }
    const dnnl_data_type_t *ddt() const override { return &prb_->ddt; }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const std::string *dtag() const override { return &dtag_; }

private:
    const prb_t *prb_ = NULL;
    std::vector<dnnl_data_type_t> sdt_;
    std::vector<std::string> stag_;
    std::string dtag_;
};

void compute_ref(const prb_t *prb, const dnn_mem_t &src, dnn_mem_t &dst);

int doit(const prb_t *prb, res_t *res);
int bench(int argc, char **argv);

} // namespace reduction

#endif
