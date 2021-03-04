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

#ifndef CPU_REF_CONCAT_HPP
#define CPU_REF_CONCAT_HPP

#include "common/engine.hpp"
#include "common/primitive.hpp"
#include "common/reorder_pd.hpp"
#include "common/stream.hpp"

#include "cpu/cpu_concat_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct ref_concat_t : public primitive_t {
    struct pd_t : public cpu_concat_pd_t {
        pd_t(const primitive_attr_t *attr, const memory_desc_t *dst_md, int n,
                int concat_dim, const memory_desc_t *src_mds)
            : cpu_concat_pd_t(attr, dst_md, n, concat_dim, src_mds)
            , tent_dst_md_(types::zero_md()) {}
        pd_t(const pd_t &rhs) : cpu_concat_pd_t(rhs) { copy(rhs); }
        ~pd_t() = default;

        DECLARE_CONCAT_PD_T("ref:any", ref_concat_t);

        status_t init(engine_t *engine) {
            status_t status = cpu_concat_pd_t::init();
            if (status != status::success) {
                assert(dst_md_.format_kind != format_kind::undef);
                status = dnnl_memory_desc_init_by_strides(&tent_dst_md_,
                        dst_md_.ndims, dst_md_.dims, dst_md_.data_type,
                        nullptr);
                if (status != status::success) return status::unimplemented;

                status = cpu_concat_pd_t::init(&tent_dst_md_);
                if (status != status::success) return status::unimplemented;
            }

            for (int i = 0; i < n_; ++i) {
                auto r_impls = engine->get_reorder_implementation_list(
                        src_md(i), src_image_md(i));
                for (auto r = r_impls; *r; ++r) {
                    primitive_attr_t r_attr; /* alpha == 1. */
                    r_attr.set_scratchpad_mode(scratchpad_mode::user);
                    reorder_pd_t *r_pd = nullptr;

                    if ((*r)(&r_pd, engine, &r_attr, engine, src_md(i), engine,
                                src_image_md(i))
                            == status::success) {
                        reorder_pds_.emplace_back(r_pd);
                        break;
                    }
                }
            }
            if (reorder_pds_.size() != (size_t)n_) return status::unimplemented;

            if (use_tent_dst()) {
                assert(tent_dst_md_.format_kind != format_kind::undef);
                assert(dst_md_.format_kind != format_kind::undef);

                auto r_impls = engine->get_reorder_implementation_list(
                        &tent_dst_md_, &dst_md_);
                for (auto r = r_impls; *r; ++r) {
                    primitive_attr_t r_attr;
                    r_attr.set_scratchpad_mode(scratchpad_mode::user);
                    reorder_pd_t *r_pd = nullptr;
                    if ((*r)(&r_pd, engine, &r_attr, engine, &tent_dst_md_,
                                engine, &dst_md_)
                            == status::success) {
                        reorder_pds_.emplace_back(r_pd);
                        break;
                    }
                }
            }
            init_scratchpad();
            return status;
        }

        // if dst is forced and cannot be used directly.
        bool use_tent_dst() const { return !types::is_zero_md(&tent_dst_md_); }

        std::vector<std::unique_ptr<primitive_desc_t>> reorder_pds_;
        memory_desc_t tent_dst_md_;

    private:
        void copy(const pd_t &rhs) {
            tent_dst_md_ = rhs.tent_dst_md_;
            for (size_t i = 0; i < rhs.reorder_pds_.size(); ++i)
                reorder_pds_.emplace_back(rhs.reorder_pds_[i]->clone());
        }

        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            if (use_tent_dst()) {
                const memory_desc_wrapper tent_dst_d(&tent_dst_md_);
                scratchpad.book(memory_tracking::names::key_concat_tent_dst,
                        tent_dst_d.size(), 1, tent_dst_d.data_type_size());
            }

            for (size_t i = 0; i < reorder_pds_.size(); i++) {
                scratchpad.book(key_nested_multiple + (int)i,
                        reorder_pds_[i]->scratchpad_registry());
            }
        }
    };

    ref_concat_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        const size_t n = pd()->reorder_pds_.size();
        reorders_.resize(n);
        for (size_t i = 0; i < n; ++i)
            pd()->reorder_pds_[i]->create_primitive(reorders_[i], engine);
        return status::success;
    }

    ~ref_concat_t() = default;

    status_t execute(const exec_ctx_t &ctx) const override {
        using namespace memory_tracking::names;
        engine_t *engine = ctx.stream()->engine();
        const auto n = pd()->n_inputs();

        auto execute_reorder = [&](const std::shared_ptr<primitive_t> &reorder,
                                       const memory_arg_t &src,
                                       const memory_arg_t &dst, int r_num) {
            exec_args_t r_args;
            r_args[DNNL_ARG_SRC] = src;
            r_args[DNNL_ARG_DST] = dst;
            exec_ctx_t r_ctx(ctx, std::move(r_args));

            nested_scratchpad_t ns(ctx, key_nested_multiple + r_num, reorder);
            r_ctx.set_scratchpad_grantor(ns.grantor());
            reorder->execute(r_ctx);
        };

        if (pd()->use_tent_dst()) {
            using namespace memory_tracking::names;
            auto scratchpad = ctx.get_scratchpad_grantor();
            auto tent_dst_storage
                    = scratchpad.get_memory_storage(key_concat_tent_dst);

            for (int i = 0; i < n; ++i) {
                memory_t tent_dst_i(engine, pd()->src_image_md(i),
                        tent_dst_storage->clone(), false);
                execute_reorder(reorders_[i],
                        ctx.args().at(DNNL_ARG_MULTIPLE_SRC + i),
                        {&tent_dst_i, false}, i);
            }

            memory_t tent_dst(engine, &pd()->tent_dst_md_,
                    tent_dst_storage->clone(), false);
            execute_reorder(reorders_[n], {&tent_dst, true},
                    ctx.args().at(DNNL_ARG_DST), n);
        } else {
            auto &dst_mem_storage = CTX_OUT_STORAGE(DNNL_ARG_DST);
            for (int i = 0; i < n; ++i) {
                memory_t tent_dst_i(engine, pd()->src_image_md(i),
                        dst_mem_storage.clone(), false);
                execute_reorder(reorders_[i],
                        ctx.args().at(DNNL_ARG_MULTIPLE_SRC + i),
                        {&tent_dst_i, false}, i);
            }
        }
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::vector<std::shared_ptr<primitive_t>> reorders_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
