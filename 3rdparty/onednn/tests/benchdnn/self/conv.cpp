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

#include <stdlib.h>
#include <string.h>

#include "conv/conv.hpp"
#include "self/self.hpp"

using namespace conv;

namespace self {

static int check_simple_enums() {
    /* alg */
    CHECK_CASE_STR_EQ(alg2str(alg_t::AUTO), "auto");
    CHECK_CASE_STR_NE(alg2str(alg_t::AUTO), "autox");

    CHECK_CASE_STR_EQ(alg2str(alg_t::DIRECT), "direct");
    CHECK_CASE_STR_NE(alg2str(alg_t::DIRECT), "directx");

    CHECK_CASE_STR_EQ(alg2str(alg_t::WINO), "wino");
    CHECK_CASE_STR_NE(alg2str(alg_t::WINO), "winox");

    CHECK_EQ(str2alg("auto"), alg_t::AUTO);
    CHECK_EQ(str2alg("AUTO"), alg_t::AUTO);

    CHECK_EQ(str2alg("direct"), alg_t::DIRECT);
    CHECK_EQ(str2alg("DIRECT"), alg_t::DIRECT);

    CHECK_EQ(str2alg("wino"), alg_t::WINO);
    CHECK_EQ(str2alg("WINO"), alg_t::WINO);

    return OK;
}

void conv() {
    RUN(check_simple_enums());
}

} // namespace self
