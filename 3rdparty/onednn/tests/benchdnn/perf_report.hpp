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

#ifndef PERF_REPORT_HPP
#define PERF_REPORT_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <sstream>

#include "dnnl_memory.hpp"
#include "oneapi/dnnl/dnnl.h"

// Please update doc/knobs_perf_report.md in case of any changes!

struct base_perf_report_t {
    base_perf_report_t(const char *perf_template) : pt_(perf_template) {}
    virtual ~base_perf_report_t() {}

    void handle_option(std::ostream &s, const char *&option, const res_t *r,
            const char *prb_str) const {
        const auto &t = r->timer;
        benchdnn_timer_t::mode_t mode = benchdnn_timer_t::min;
        (void)mode;
        double unit = 1e0;
        char c = *option;

        if (c == '-' || c == '0' || c == '+') {
            mode = modifier2mode(c);
            c = *(++option);
        }

        if (c == 'K' || c == 'M' || c == 'G') {
            unit = modifier2unit(c);
            c = *(++option);
        }

#define HANDLE(opt, ...) \
    if (!strncmp(opt "%", option, strlen(opt) + 1)) { \
        __VA_ARGS__; \
        option += strlen(opt) + 1; \
        return; \
    }

        auto get_flops = [&]() -> double {
            if (!t.sec(mode)) return 0;
            return ops() / t.sec(mode) / unit;
        };

        auto get_bw = [&]() -> double { return get_flops(); };

        auto get_freq = [&]() -> double {
            if (!t.sec(mode)) return 0;
            return t.ticks(mode) / t.sec(mode) / unit;
        };

        HANDLE("alg", dump_alg(s));
        HANDLE("cfg", dump_cfg(s));
        HANDLE("desc", dump_desc(s));
        HANDLE("DESC", dump_desc_csv(s));
        HANDLE("engine", dump_engine(s));
        HANDLE("flags", dump_flags(s));
        HANDLE("activation", dump_rnn_activation(s));
        HANDLE("direction", dump_rnn_direction(s));

        HANDLE("attr", if (attr() && !attr()->is_def()) s << *attr());
        HANDLE("axis", if (axis()) s << *axis());
        HANDLE("dir", if (dir()) s << *dir());
        HANDLE("dt", if (dt()) s << *dt());
        HANDLE("group", if (group()) s << *group());
        HANDLE("sdt", if (sdt()) s << *sdt());
        HANDLE("stag", if (stag()) s << *stag());
        HANDLE("name", if (name()) s << name());
        HANDLE("ddt", if (ddt()) s << *ddt());
        HANDLE("dtag", if (dtag()) s << *dtag());
        HANDLE("prop", if (prop()) s << prop2str(*prop()));
        HANDLE("tag", if (tag()) s << *tag());
        HANDLE("stat_tag", if (stat_tag()) s << *stat_tag());
        HANDLE("wtag", if (wtag()) s << *wtag());

        HANDLE("bw", s << get_bw());
        HANDLE("flops", s << get_flops());
        HANDLE("clocks", s << t.ticks(mode) / unit);
        HANDLE("prb", s << prb_str);
        HANDLE("freq", s << get_freq());
        HANDLE("ops", s << ops() / unit);
        HANDLE("time", s << t.ms(mode) / unit);
        HANDLE("impl", s << r->impl_name);

#undef HANDLE

        SAFE_V(FAIL);
    }

    void base_report(const res_t *r, const char *prb_str) const {
        dump_perf_footer();

        std::stringstream ss;

        const char *pt = pt_;
        char c;
        while ((c = *pt++) != '\0') {
            if (c != '%') {
                ss << c;
                continue;
            }
            handle_option(ss, pt, r, prb_str);
        }

        std::string str = ss.str();
        BENCHDNN_PRINT(0, "%s\n", str.c_str());
    };

    /* truly common types */
    virtual double ops() const { return 0.; }
    virtual const attr_t *attr() const { return nullptr; }
    virtual const int *axis() const { return nullptr; }
    virtual const char *name() const { return nullptr; }
    virtual const int64_t *group() const { return nullptr; }
    virtual const dir_t *dir() const { return nullptr; }
    virtual const dnnl_data_type_t *dt() const { return nullptr; }
    virtual const std::vector<dnnl_data_type_t> *sdt() const { return nullptr; }
    virtual const dnnl_data_type_t *ddt() const { return nullptr; }
    virtual const std::string *tag() const { return nullptr; }
    virtual const std::string *stat_tag() const { return nullptr; }
    virtual const std::vector<std::string> *stag() const { return nullptr; }
    virtual const std::string *dtag() const { return nullptr; }
    virtual const std::string *wtag() const { return nullptr; }
    virtual const dnnl_prop_kind_t *prop() const { return nullptr; }

    /* designed to be overloaded in reorder only to match verbose output */
    virtual void dump_engine(std::ostream &s) const { s << engine_tgt_kind; }

    /* primitive-specific properties (but with common interface) */
    virtual void dump_alg(std::ostream &) const { SAFE_V(FAIL); }
    virtual void dump_cfg(std::ostream &) const { SAFE_V(FAIL); }
    virtual void dump_desc(std::ostream &) const { SAFE_V(FAIL); }
    virtual void dump_desc_csv(std::ostream &) const { SAFE_V(FAIL); }
    virtual void dump_flags(std::ostream &) const { SAFE_V(FAIL); }
    virtual void dump_rnn_activation(std::ostream &) const { SAFE_V(FAIL); }
    virtual void dump_rnn_direction(std::ostream &) const { SAFE_V(FAIL); }

private:
    const char *pt_;

    void dump_perf_footer() const {
        static bool footer_printed = false;
        if (!footer_printed) {
            BENCHDNN_PRINT(0, "Output template: %s\n", pt_);
            footer_printed = true;
        }
    }

    static benchdnn_timer_t::mode_t modifier2mode(char c) {
        if (c == '-') return benchdnn_timer_t::min;
        if (c == '0') return benchdnn_timer_t::avg;
        if (c == '+') return benchdnn_timer_t::max;
        return benchdnn_timer_t::min;
    }

    static double modifier2unit(char c) {
        if (c == 'K') return 1e3;
        if (c == 'M') return 1e6;
        if (c == 'G') return 1e9;
        return 1e0;
    }
};

#endif
