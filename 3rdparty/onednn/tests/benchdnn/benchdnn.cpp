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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "parser.hpp"

#include "binary/binary.hpp"
#include "bnorm/bnorm.hpp"
#include "concat/concat.hpp"
#include "conv/conv.hpp"
#include "conv/deconv.hpp"
#include "eltwise/eltwise.hpp"
#include "ip/ip.hpp"
#include "lnorm/lnorm.hpp"
#include "lrn/lrn.hpp"
#include "matmul/matmul.hpp"
#include "pool/pool.hpp"
#include "reduction/reduction.hpp"
#include "reorder/reorder.hpp"
#include "resampling/resampling.hpp"
#include "rnn/rnn.hpp"
#include "self/self.hpp"
#include "shuffle/shuffle.hpp"
#include "softmax/softmax.hpp"
#include "sum/sum.hpp"

int verbose {0};
bool canonical {false};
bool mem_check {true};
std::string skip_impl;
bench_mode_t bench_mode {CORR};
stat_t benchdnn_stat {0};
const char *driver_name = "";

double max_ms_per_prb {3e3};
int min_times_per_prb {5};
int fix_times_per_prb {0};

bool fast_ref_gpu {true};
bool allow_enum_tags_only {true};

int main(int argc, char **argv) {
    using namespace parser;

    if (argc < 2) {
        fprintf(stderr, "err: no arguments passed\n");
        return 1;
    }

    --argc;
    ++argv;

    init_fp_mode();

    for (; argc > 0; --argc, ++argv)
        if (!parse_bench_settings(argv[0])) break;

    if (!strcmp("--self", argv[0])) {
        self::bench(--argc, ++argv);
    } else if (!strcmp("--conv", argv[0])) {
        conv::bench(--argc, ++argv);
    } else if (!strcmp("--deconv", argv[0])) {
        deconv::bench(--argc, ++argv);
    } else if (!strcmp("--ip", argv[0])) {
        ip::bench(--argc, ++argv);
    } else if (!strcmp("--shuffle", argv[0])) {
        shuffle::bench(--argc, ++argv);
    } else if (!strcmp("--reorder", argv[0])) {
        reorder::bench(--argc, ++argv);
    } else if (!strcmp("--bnorm", argv[0])) {
        bnorm::bench(--argc, ++argv);
    } else if (!strcmp("--lnorm", argv[0])) {
        lnorm::bench(--argc, ++argv);
    } else if (!strcmp("--rnn", argv[0])) {
        rnn::bench(--argc, ++argv);
    } else if (!strcmp("--softmax", argv[0])) {
        softmax::bench(--argc, ++argv);
    } else if (!strcmp("--pool", argv[0])) {
        pool::bench(--argc, ++argv);
    } else if (!strcmp("--sum", argv[0])) {
        sum::bench(--argc, ++argv);
    } else if (!strcmp("--eltwise", argv[0])) {
        eltwise::bench(--argc, ++argv);
    } else if (!strcmp("--concat", argv[0])) {
        concat::bench(--argc, ++argv);
    } else if (!strcmp("--lrn", argv[0])) {
        lrn::bench(--argc, ++argv);
    } else if (!strcmp("--binary", argv[0])) {
        binary::bench(--argc, ++argv);
    } else if (!strcmp("--matmul", argv[0])) {
        matmul::bench(--argc, ++argv);
    } else if (!strcmp("--resampling", argv[0])) {
        resampling::bench(--argc, ++argv);
    } else if (!strcmp("--reduction", argv[0])) {
        reduction::bench(--argc, ++argv);
    } else {
        fprintf(stderr, "err: unknown driver\n");
    }

    printf("tests:%d passed:%d "
           "skipped:%d mistrusted:%d unimplemented:%d "
           "failed:%d listed:%d\n",
            benchdnn_stat.tests, benchdnn_stat.passed, benchdnn_stat.skipped,
            benchdnn_stat.mistrusted, benchdnn_stat.unimplemented,
            benchdnn_stat.failed, benchdnn_stat.listed);
    if (bench_mode & PERF) {
        printf("total perf: min(ms):%g avg(ms):%g\n",
                benchdnn_stat.ms[benchdnn_timer_t::min],
                benchdnn_stat.ms[benchdnn_timer_t::avg]);
    }

    return !!benchdnn_stat.failed;
}
