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

#ifndef CPU_REF_FUSED_CONVOLUTION_HPP
#define CPU_REF_FUSED_CONVOLUTION_HPP

#include "common/primitive.hpp"
#include "common/primitive_iterator.hpp"
#include "common/stream.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/dw_convolution_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct ref_fused_convolution_fwd_t : public primitive_t {

    struct arg_cache_t {
        struct arg_info_t {
            int op_arg;
            bool is_ctx_arg;
            bool is_const;
            union {
                size_t offset;
                int ctx_arg;
            };
            memory_desc_t md;
        };

        void append_ctx_arg(int op_arg, int ctx_arg) {
            arg_info_t arg_info;
            arg_info.op_arg = op_arg;
            arg_info.is_ctx_arg = true;
            arg_info.is_const = false; // unused
            arg_info.ctx_arg = ctx_arg;
            arg_info.md = glob_zero_md;
            info_.push_back(arg_info);
        }

        void append_inout_arg(int arg, size_t offset, const memory_desc_t *md,
                bool is_const) {
            arg_info_t arg_info;
            arg_info.op_arg = arg;
            arg_info.is_ctx_arg = false;
            arg_info.is_const = is_const;
            arg_info.offset = offset;
            arg_info.md = *md;
            info_.push_back(arg_info);
        }

        void append_ctx_arg(int arg) { append_ctx_arg(arg, arg); }

        const std::vector<arg_info_t> &info() const { return info_; }

    private:
        std::vector<arg_info_t> info_;
    };

    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd) {
            name_ = "ref_fused_convolution:any";
        }

        pd_t(const pd_t &other) : cpu_convolution_fwd_pd_t(other) {
            copy_from(other);
        }

        DECLARE_COMMON_PD_T(name_.c_str(), ref_fused_convolution_fwd_t);

        virtual status_t init(engine_t *engine) {
            bool ok = true && is_fwd()
                    && (attr()->post_ops_.find(primitive_kind::sum) == -1);

            if (!ok) return status::unimplemented;

            CHECK(init_ops(engine));
            init_name();
            return status::success;
        }

        const memory_desc_t *src_md(int index = 0) const override {
            return op_pds_.front()->src_md(index);
        }

        const memory_desc_t *dst_md(int index = 0) const override {
            return op_pds_.back()->dst_md(index);
        }

        const memory_desc_t *weights_md(int index = 0) const override {
            return op_pds_.front()->weights_md(index); // for now
        }

        const memory_desc_t *arg_md(int index = 0) const override {
            switch (index) { // for now
                case DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS:
                    return op_pds_.back()->weights_md(0);
                case DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS:
                    return op_pds_.back()->weights_md(1);
                default: return convolution_fwd_pd_t::arg_md(index);
            }
        }

        arg_usage_t arg_usage(int arg) const override {

            if (utils::one_of(arg, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS,
                        DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS))
                return arg_usage_t::input;

            return convolution_fwd_pd_t::arg_usage(arg);
        }

        size_t user_scratchpad_size_;
        std::vector<std::unique_ptr<primitive_desc_t>> op_pds_;
        std::vector<arg_cache_t> args_;

    private:
        std::string name_;
        const unsigned int max_fusions_ = 1;

        status_t append_op(primitive_desc_t *op_pd, size_t &sp_begin,
                size_t &sp_end, engine_t *engine) {
            auto from_md = op_pds_.back()->dst_md();
            auto to_md = op_pd->src_md();

            if (*from_md != *to_md) {
                //TODO: Find a test-case for this
                reorder_pd_t *r_pd = nullptr;
                auto r_impls = engine->get_reorder_implementation_list(
                        from_md, to_md);
                for (auto r = r_impls; *r; ++r) {
                    primitive_attr_t attr;
                    attr.set_scratchpad_mode(scratchpad_mode::user);
                    if ((*r)(&r_pd, engine, &attr, engine, from_md, engine,
                                to_md)
                            == status::success) {
                        op_pds_.emplace_back((primitive_desc_t *)r_pd);
                        break;
                    }
                }
                if (!r_pd) return status::unimplemented;

                arg_cache_t arg_cache;
                arg_cache.append_inout_arg(
                        DNNL_ARG_FROM, sp_begin, from_md, true);
                arg_cache.append_inout_arg(DNNL_ARG_TO, sp_end, to_md, false);
                args_.push_back(arg_cache);

                // Increment scratchpad offsets
                sp_begin = sp_end;
                sp_end += memory_desc_wrapper(to_md).size();

                user_scratchpad_size_ = nstl::max<size_t>(user_scratchpad_size_,
                        op_pds_.back()->scratchpad_size(scratchpad_mode::user));
            }

            op_pds_.emplace_back(op_pd);
            user_scratchpad_size_ = nstl::max<size_t>(user_scratchpad_size_,
                    op_pds_.back()->scratchpad_size(scratchpad_mode::user));
            return status::success;
        }

        status_t init_ops(engine_t *engine) {
            using namespace data_type;
            primitive_attr_t root_attr(*attr());
            if (!root_attr.is_initialized()) return status::out_of_memory;
            root_attr.set_scratchpad_mode(scratchpad_mode::user);
            auto po_op_iter
                    = attr()->post_ops_.find(primitive_kind::convolution);
            if (po_op_iter == -1) return status::unimplemented;

            primitive_attr_t attr_1x1(*attr());
            // erase post-ops after fusion as they will be handled separately
            auto &e = attr_1x1.post_ops_.entry_;
            e.erase(e.begin() + po_op_iter, e.end());

            attr_1x1.set_scratchpad_mode(scratchpad_mode::user);

            dnnl_primitive_desc_iterator it(
                    engine, op_desc(), &attr_1x1, nullptr);
            if (!it.is_initialized()) return status::out_of_memory;
            ++it;
            primitive_desc_t *root_pd = it.fetch_once();
            if (!root_pd) return status::unimplemented;
            op_pds_.emplace_back(root_pd);
            // Scratchpad offsets. Simulate offset computation so that offset
            // computation can be avoided during execution.
            size_t inout_sp_offset_begin = 0;
            size_t inout_sp_offset_end = 0;
            user_scratchpad_size_
                    = root_pd->scratchpad_size(scratchpad_mode::user);

            // Create arg cache for the root pd
            arg_cache_t arg_cache;
            arg_cache.append_ctx_arg(DNNL_ARG_SRC);
            arg_cache.append_ctx_arg(DNNL_ARG_WEIGHTS);
            if (desc()->bias_desc.data_type != data_type::undef)
                arg_cache.append_ctx_arg(DNNL_ARG_BIAS);
            arg_cache.append_inout_arg(DNNL_ARG_DST, inout_sp_offset_end,
                    root_pd->dst_md(), false);
            args_.push_back(arg_cache);

            // Increment scratchpad offsets
            inout_sp_offset_begin = inout_sp_offset_end;
            inout_sp_offset_end
                    += memory_desc_wrapper(root_pd->dst_md()).size();

            const auto &po = attr()->post_ops_;
            const auto &end = po.len();

            unsigned int fusion_ops = 0;
            // Loop through the post-ops untill we reach the end
            // (if we have more than one op to fuse later)
            while (po_op_iter < end) {
                if (fusion_ops++ > max_fusions_) return status::unimplemented;

                const auto &prev_op_pd = op_pds_.back();

                if (po.entry_[po_op_iter].kind == primitive_kind::convolution) {
                    if (prev_op_pd->kind() != primitive_kind::convolution)
                        return status::unimplemented;
                    auto conv_pd = reinterpret_cast<convolution_pd_t *>(
                            prev_op_pd.get());
                    bool ok = true && is_fwd()
                            && utils::everyone_is(1, conv_pd->KD(),
                                    conv_pd->KH(), conv_pd->KW());
                    if (!ok) return status::unimplemented;

                    convolution_desc_t cd_dw;
                    primitive_attr_t attr_dw;
                    primitive_desc_t *append_conv_pd;
                    CHECK(get_depthwise_conv_desc(cd_dw, *(conv_pd->dst_md()),
                            root_attr, attr_dw, po_op_iter));
                    dnnl_primitive_desc_iterator it(
                            engine, (op_desc_t *)&cd_dw, &attr_dw, nullptr);
                    if (!it.is_initialized()) return status::out_of_memory;
                    ++it;
                    append_conv_pd = (it.fetch_once());
                    if (!append_conv_pd) return status::unimplemented;

                    auto status = append_op(append_conv_pd,
                            inout_sp_offset_begin, inout_sp_offset_end, engine);
                    if (status != status::success) {
                        delete append_conv_pd;
                        return status;
                    }

                    const auto &op = op_pds_.back();
                    arg_cache_t arg_cache;
                    arg_cache.append_inout_arg(DNNL_ARG_SRC,
                            inout_sp_offset_begin, op->src_md(), true);
                    arg_cache.append_ctx_arg(DNNL_ARG_DST);
                    arg_cache.append_ctx_arg(DNNL_ARG_WEIGHTS,
                            DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
                    if (op->weights_md(1)->data_type != data_type::undef)
                        arg_cache.append_ctx_arg(DNNL_ARG_BIAS,
                                DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);

                    args_.push_back(arg_cache);

                } else // other fused ops
                    return status::unimplemented;

                while (++po_op_iter < end) {
                    if (utils::one_of(po.entry_[po_op_iter].kind,
                                primitive_kind::convolution))
                        break;
                }
            }

            assert(!op_pds_.empty());

            CHECK(init_scratchpad_memory(inout_sp_offset_end));

            return status::success;
        }

        status_t init_scratchpad_memory(size_t inout_buffer_size) {

            auto scratchpad = scratchpad_registry().registrar();

            scratchpad.book(memory_tracking::names::key_fusion_inout_buffer,
                    inout_buffer_size, 1, 16);
            scratchpad.book(
                    memory_tracking::names::key_fusion_forward_scratchpad,
                    user_scratchpad_size_, 1, 16);
            return status::success;
        }

        void init_name() {
            for (const auto &op_pd : op_pds_) {
                name_.append(":");
                name_.append(op_pd->name());
            }
            return;
        }

        void copy_from(const pd_t &other) {
            user_scratchpad_size_ = other.user_scratchpad_size_;
            op_pds_.clear();
            for (const auto &other_op_pd : other.op_pds_)
                op_pds_.emplace_back(other_op_pd->clone());
            args_ = other.args_;
            name_ = other.name_;
            return;
        }
    };

    ref_fused_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        const auto &op_pds = pd()->op_pds_;
        for (auto &op_pd : op_pds) {
            std::shared_ptr<primitive_t> p;
            op_pd->create_primitive(p, engine);
            primitives_.emplace_back(p);
        }
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        engine_t *engine = ctx.stream()->engine();
        const auto scratchpad = ctx.get_scratchpad_grantor();

        const auto inout_buffer = scratchpad.get_memory_storage(
                memory_tracking::names::key_fusion_inout_buffer);

        const auto &ctx_args = ctx.args();
        const auto op_count = primitives_.size();
        std::vector<std::unique_ptr<memory_t>> inout_memory;

        for (size_t i = 0; i < op_count; ++i) {
            const auto &op = primitives_[i];
            const auto &arg_cache = pd()->args_[i];

            exec_args_t exec_args;

            for (const auto &arg_info : arg_cache.info()) {
                if (arg_info.is_ctx_arg) {
                    exec_args[arg_info.op_arg] = ctx_args.at(arg_info.ctx_arg);
                } else {
                    inout_memory.emplace_back(new memory_t(engine, &arg_info.md,
                            inout_buffer->get_sub_storage(arg_info.offset,
                                    memory_desc_wrapper(arg_info.md).size()),
                            false));
                    exec_args[arg_info.op_arg].mem = inout_memory.back().get();
                    exec_args[arg_info.op_arg].is_const = arg_info.is_const;
                }
            }

            exec_ctx_t op_ctx(ctx, std::move(exec_args));

            nested_scratchpad_t ns(ctx,
                    memory_tracking::names::key_fusion_forward_scratchpad, op);
            op_ctx.set_scratchpad_grantor(ns.grantor());
            CHECK(op->execute(op_ctx));
        }

        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::vector<std::shared_ptr<primitive_t>> primitives_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
