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

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_amx_tilecfg_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_amx_tilecfg_t)

    // TODO: Need to check status
    jit_amx_tilecfg_t() { create_kernel(); }

    void tile_configure(const char *palette) const { (*this)(palette); }

private:
    void generate() override {
        preamble();

        tilerelease();
        ldtilecfg(ptr[abi_param1]);

        postamble();
    }
};

void amx_tile_configure(const char palette[64]) {
    static const jit_amx_tilecfg_t tilecfg;
    tilecfg.tile_configure(palette);
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
