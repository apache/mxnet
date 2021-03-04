/*******************************************************************************
* Copyright 2017 - 2020 Intel Corporation
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

#include <functional>
#include <new>

#include "oneapi/dnnl/dnnl_types.h"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"

#include "cpu/x64/jit_uni_pooling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace jit_uni_pooling_utils {

struct trans_wrapper_t {
    trans_wrapper_t(data_type_t inp_dt, dim_t inp_str, data_type_t out_dt,
            dim_t out_str, dim_t ysize, dim_t xsize)
        : inp_dt_size_(types::data_type_size(inp_dt))
        , out_dt_size_(types::data_type_size(out_dt))
        , inp_str_(inp_str)
        , out_str_(out_str)
        , nb_x_(xsize / 8)
        , nb_y_(ysize / 8)
        , x_tail_(xsize % 8)
        , y_tail_(ysize % 8) {
        using namespace cpu::x64::tr;

        auto create_ker = [=](dim_t ys, dim_t y_inp_str, dim_t y_out_str,
                                  dim_t xs, dim_t x_inp_str, dim_t x_out_str) {
            tr::prb_t prb;
            kernel_t::desc_t desc;

            prb.ndims = 2;
            prb.ioff = 0;
            prb.ooff = 0;
            prb.scale_type = scale_type_t::NONE;
            prb.beta = 0;
            prb.nodes[0].ss = prb.nodes[1].ss = 1;

            prb.itype = inp_dt;
            prb.otype = out_dt;

            prb.nodes[0].n = ys;
            prb.nodes[0].is = y_inp_str;
            prb.nodes[0].os = y_out_str;

            prb.nodes[1].n = xs;
            prb.nodes[1].is = x_inp_str;
            prb.nodes[1].os = x_out_str;

            kernel_t::desc_init(desc, prb, 2);
            return kernel_t::create(desc);
        };

        if (nb_x_ * nb_y_ > 0)
            ker_.reset(create_ker(8, inp_str_, 1, 8, 1, out_str_));

        if (x_tail_)
            ker_x_tail_.reset(create_ker(8, inp_str_, 1, x_tail_, 1, out_str_));

        if (y_tail_)
            ker_y_tail_.reset(
                    create_ker(y_tail_, inp_str_, 1, xsize, 1, out_str_));
    }

    status_t create_kernel() {
        if (ker_) CHECK(ker_->create_kernel());
        if (ker_x_tail_) CHECK(ker_x_tail_->create_kernel());
        if (ker_y_tail_) CHECK(ker_y_tail_->create_kernel());
        return status::success;
    }

    void exec(const void *inp, void *out) {
        dim_t x_blocked = nb_x_ * 8;
        dim_t y_blocked = nb_y_ * 8;

        auto call_ker = [&](tr::kernel_t &ker, dim_t inp_y, dim_t inp_x,
                                dim_t out_y, dim_t out_x) {
            tr::call_param_t cp;
            cp.scale = nullptr;

            dim_t inp_off = (inp_y * inp_str_ + inp_x) * inp_dt_size_;
            dim_t out_off = (out_y * out_str_ + out_x) * out_dt_size_;
            cp.in = (uint8_t *)inp + inp_off;
            cp.out = (uint8_t *)out + out_off;
            (ker)(&cp);
        };

        for (dim_t by = 0; by < nb_y_; by++) {
            for (dim_t bx = 0; bx < nb_x_; bx++)
                call_ker(*ker_, 8 * by, 8 * bx, 8 * bx, 8 * by);

            if (x_tail_)
                call_ker(*ker_x_tail_, 8 * by, x_blocked, x_blocked, 8 * by);
        }
        if (y_tail_) call_ker(*ker_y_tail_, y_blocked, 0, 0, y_blocked);
    }

    ~trans_wrapper_t() = default;

private:
    std::unique_ptr<tr::kernel_t> ker_;
    std::unique_ptr<tr::kernel_t> ker_x_tail_;
    std::unique_ptr<tr::kernel_t> ker_y_tail_;

    const size_t inp_dt_size_;
    const size_t out_dt_size_;

    const dim_t inp_str_;
    const dim_t out_str_;
    const dim_t nb_x_;
    const dim_t nb_y_;
    const dim_t x_tail_;
    const dim_t y_tail_;
};

struct trans_context_t {
    std::unique_ptr<trans_wrapper_t> src_trans_ = nullptr;
    std::unique_ptr<trans_wrapper_t> src_tail_trans_ = nullptr;
    std::unique_ptr<trans_wrapper_t> ind_trans_ = nullptr;
    std::unique_ptr<trans_wrapper_t> ind_tail_trans_ = nullptr;
    std::unique_ptr<trans_wrapper_t> dst_trans_ = nullptr;
    std::unique_ptr<trans_wrapper_t> dst_tail_trans_ = nullptr;
    status_t create_kernel() {
        if (src_trans_) CHECK(src_trans_->create_kernel());
        if (src_tail_trans_) CHECK(src_tail_trans_->create_kernel());
        if (ind_trans_) CHECK(ind_trans_->create_kernel());
        if (ind_tail_trans_) CHECK(ind_tail_trans_->create_kernel());
        if (dst_trans_) CHECK(dst_trans_->create_kernel());
        if (dst_tail_trans_) CHECK(dst_tail_trans_->create_kernel());
        return status::success;
    }
};

static void trans_exec(trans_wrapper_t *trans, trans_wrapper_t *trans_tail,
        dim_t cs, const void *inp, void *out, dim_t c_block) {

    if (cs == c_block)
        trans->exec(inp, out);
    else
        trans_tail->exec(inp, out);
};

template <typename src_data_t, typename dst_data_t>
struct transpose_ncsp_to_block_fmt_t {
    transpose_ncsp_to_block_fmt_t(trans_wrapper_t *transposer,
            trans_wrapper_t *transposer_tail, const src_data_t *src_nscp_base,
            const memory_desc_wrapper &src_nscp_desc,
            dst_data_t *__restrict dst_blocked_base, dim_t block_size,
            const jit_pool_conf_t &jpp, std::size_t offset_multiplier = 1u)
        : transposer_(transposer)
        , transposer_tail_(transposer_tail)
        , c_without_padding_(jpp.c_without_padding)
        , c_block_(jpp.c_block)
        , src_nscp_base_(src_nscp_base)
        , src_nscp_desc_(src_nscp_desc)
        , dst_blocked_base_(dst_blocked_base)
        , block_size_(block_size)
        , offset_multiplier_(offset_multiplier) {}

    void operator()(std::size_t ithr, int n, int b_c) const {
        const dim_t cs
                = nstl::min(c_without_padding_ - b_c * c_block_, c_block_);
        const src_data_t *src_nscp = src_nscp_base_
                + src_nscp_desc_.blk_off(n, b_c * c_block_, 0)
                        * offset_multiplier_;
        dst_data_t *dst_blocked
                = dst_blocked_base_ + ithr * block_size_ * offset_multiplier_;
        trans_exec(transposer_, transposer_tail_, cs, src_nscp, dst_blocked,
                c_block_);
    }

private:
    trans_wrapper_t *transposer_;
    trans_wrapper_t *transposer_tail_;
    const int c_without_padding_;
    const int c_block_;
    const src_data_t *src_nscp_base_;
    const memory_desc_wrapper &src_nscp_desc_;
    dst_data_t *__restrict dst_blocked_base_;
    const dim_t block_size_;
    std::size_t offset_multiplier_;
};

template <typename src_data_t, typename dst_data_t>
struct transpose_block_fmt_to_ncsp_t {

    transpose_block_fmt_to_ncsp_t(trans_wrapper_t *transposer,
            trans_wrapper_t *transposer_tail,
            const src_data_t *__restrict src_blocked_base, dim_t block_size,
            dst_data_t *dst_ncsp_base, const memory_desc_wrapper &dst_nscp_desc,
            const jit_pool_conf_t &jpp, std::size_t offset_multiplier = 1u)
        : transposer_(transposer)
        , transposer_tail_(transposer_tail)
        , c_without_padding_(jpp.c_without_padding)
        , c_block_(jpp.c_block)
        , src_blocked_base_(src_blocked_base)
        , block_size_(block_size)
        , dst_ncsp_base_(dst_ncsp_base)
        , dst_nscp_desc_(dst_nscp_desc)
        , offset_multiplier_(offset_multiplier) {}

    void operator()(std::size_t ithr, int n, int b_c) const {
        const dim_t cs
                = nstl::min(c_without_padding_ - b_c * c_block_, c_block_);
        const src_data_t *src_blocked
                = src_blocked_base_ + ithr * block_size_ * offset_multiplier_;
        dst_data_t *dst_ncsp = dst_ncsp_base_
                + dst_nscp_desc_.blk_off(n, b_c * c_block_, 0)
                        * offset_multiplier_;
        trans_exec(transposer_, transposer_tail_, cs, src_blocked, dst_ncsp,
                c_block_);
    }

private:
    trans_wrapper_t *transposer_;
    trans_wrapper_t *transposer_tail_;
    const int c_without_padding_;
    const int c_block_;
    const src_data_t *__restrict src_blocked_base_;
    const dim_t block_size_;
    dst_data_t *dst_ncsp_base_;
    const memory_desc_wrapper &dst_nscp_desc_;
    std::size_t offset_multiplier_;
};

template <typename wsp_data_t, impl::data_type_t d_type>
class transpose_facade_base_t {
public:
    transpose_facade_base_t(const jit_pool_conf_t &jpp,
            const memory_desc_wrapper &src_d, const memory_desc_wrapper &dst_d,
            const memory_desc_wrapper &indices_d, const char *indices,
            const data_type_t wsp_dt, const exec_ctx_t &ctx)
        : src_sp_(static_cast<dim_t>(jpp.id) * jpp.ih * jpp.iw)
        , dst_sp_(static_cast<dim_t>(jpp.od) * jpp.oh * jpp.ow)
        , src_slice_(src_sp_ * jpp.c_block)
        , dst_slice_(dst_sp_ * jpp.c_block)
        , transpose_src_(jpp.tag_kind == jptg_ncsp)
        , transpose_dst_(jpp.tag_kind == jptg_ncsp)
        , src_d_(src_d)
        , dst_d_(dst_d)
        , indices_d_(indices_d)
        , ind_dt_size_(
                  indices ? types::data_type_size(indices_d_.data_type()) : 0)
        , cvt_slice_src_wsp_(nullptr)
        , cvt_slice_dst_wsp_(nullptr)
        , cvt_slice_ind_wsp_(nullptr)
        , execute_transpose_input_(nullptr)
        , execute_transpose_output_(nullptr) {

        auto scratchpad = ctx.get_scratchpad_grantor();

        if (transpose_src_)
            cvt_slice_src_wsp_ = scratchpad.template get<wsp_data_t>(
                    memory_tracking::names::key_pool_src_plain2blocked_cvt);

        if (transpose_dst_) {
            cvt_slice_dst_wsp_ = scratchpad.template get<wsp_data_t>(
                    memory_tracking::names::key_pool_dst_plain2blocked_cvt);
            cvt_slice_ind_wsp_ = scratchpad.template get<char>(
                    memory_tracking::names::key_pool_ind_plain2blocked_cvt);
        }
    }

    inline bool should_transpose_src() const noexcept { return transpose_src_; }
    inline bool should_transpose_dst() const noexcept { return transpose_dst_; }

    const void *get_src_addr(
            std::size_t ithr, int ih, const jit_pool_conf_t &jpp) const {
        const wsp_data_t *const wsp = cvt_slice_src_wsp_ + ithr * src_slice_;
        return static_cast<const void *>(&wsp[ih * jpp.iw * jpp.c_block]);
    }

    const void *get_dst_addr(
            std::size_t ithr, int oh, const jit_pool_conf_t &jpp) const {
        const wsp_data_t *const wsp = cvt_slice_dst_wsp_ + ithr * dst_slice_;
        return static_cast<const void *>(&wsp[oh * jpp.ow * jpp.c_block]);
    }

    const void *get_indices_addr(
            std::size_t ithr, int oh, const jit_pool_conf_t &jpp) const {
        const char *const wsp
                = cvt_slice_ind_wsp_ + ithr * dst_slice_ * ind_dt_size_;
        return static_cast<const void *>(
                &wsp[oh * jpp.ow * jpp.c_block * ind_dt_size_]);
    }

    const void *get_src_addr_3d(std::size_t ithr, int id, int ih,
            const jit_pool_conf_t &jpp) const {
        const wsp_data_t *const wsp = cvt_slice_src_wsp_ + ithr * src_slice_;
        return static_cast<const void *>(&wsp[ih * jpp.iw * jpp.c_block
                + id * jpp.ih * jpp.iw * jpp.c_block]);
    }

    const void *get_dst_addr_3d(std::size_t ithr, int od, int oh,
            const jit_pool_conf_t &jpp) const {
        const wsp_data_t *const wsp = cvt_slice_dst_wsp_ + ithr * dst_slice_;
        return static_cast<const void *>(&wsp[oh * jpp.ow * jpp.c_block
                + od * jpp.oh * jpp.ow * jpp.c_block]);
    }

    const void *get_indices_addr_3d(std::size_t ithr, int od, int oh,
            const jit_pool_conf_t &jpp) const {
        const char *const wsp
                = cvt_slice_ind_wsp_ + ithr * dst_slice_ * ind_dt_size_;
        return static_cast<const void *>(
                &wsp[oh * jpp.ow * jpp.c_block * ind_dt_size_
                        + od * jpp.oh * jpp.ow * jpp.c_block * ind_dt_size_]);
    }

    void execute_transpose_input(std::size_t ithr, int n, int b_c) const {
        execute_transpose_input_(ithr, n, b_c);
    }

    void execute_transpose_output(std::size_t ithr, int n, int b_c) const {
        execute_transpose_output_(ithr, n, b_c);
    }

protected:
    const dim_t src_sp_;
    const dim_t dst_sp_;
    const dim_t src_slice_;
    const dim_t dst_slice_;

    const bool transpose_src_;
    const bool transpose_dst_;

    const memory_desc_wrapper &src_d_;
    const memory_desc_wrapper &dst_d_;
    const memory_desc_wrapper &indices_d_;
    const size_t ind_dt_size_;

    wsp_data_t *__restrict cvt_slice_src_wsp_;
    wsp_data_t *__restrict cvt_slice_dst_wsp_;
    char *__restrict cvt_slice_ind_wsp_;

    std::function<void(std::size_t, int, int)> execute_transpose_input_;
    std::function<void(std::size_t, int, int)> execute_transpose_output_;
};

template <typename data_t, typename wsp_data_t, impl::data_type_t d_type>
class fwd_pooling_transpose_facade_t
    : public transpose_facade_base_t<wsp_data_t, d_type> {
public:
    fwd_pooling_transpose_facade_t(const jit_pool_conf_t &jpp,
            trans_context_t *trans_ctx, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &dst_d,
            const memory_desc_wrapper &indices_d, const data_type_t wsp_dt,
            const data_t *src, data_t *dst, char *indices,
            const exec_ctx_t &ctx)
        : transpose_facade_base_t<wsp_data_t, d_type>(
                jpp, src_d, dst_d, indices_d, indices, wsp_dt, ctx) {

        if (this->should_transpose_src()) {
            this->execute_transpose_input_
                    = transpose_ncsp_to_block_fmt_t<data_t, wsp_data_t>(
                            trans_ctx->src_trans_.get(),
                            trans_ctx->src_tail_trans_.get(), src, this->src_d_,
                            this->cvt_slice_src_wsp_, this->src_slice_, jpp);
        }

        if (this->should_transpose_dst()) {
            using namespace std::placeholders;
            this->execute_transpose_output_ = std::bind(
                    [=](const transpose_block_fmt_to_ncsp_t<wsp_data_t, data_t>
                                    &trans_dst,
                            transpose_block_fmt_to_ncsp_t<char, char>
                                    &trans_indices,
                            std::size_t ithr, int n, int b_c) {
                        trans_dst(ithr, n, b_c);
                        if (indices) trans_indices(ithr, n, b_c);
                    },
                    transpose_block_fmt_to_ncsp_t<wsp_data_t, data_t>(
                            trans_ctx->dst_trans_.get(),
                            trans_ctx->dst_tail_trans_.get(),
                            this->cvt_slice_dst_wsp_, this->dst_slice_, dst,
                            this->dst_d_, jpp, 1u),
                    transpose_block_fmt_to_ncsp_t<char, char>(
                            trans_ctx->ind_trans_.get(),
                            trans_ctx->ind_tail_trans_.get(),
                            this->cvt_slice_ind_wsp_, this->dst_slice_, indices,
                            this->indices_d_, jpp, this->ind_dt_size_),
                    _1, _2, _3);
        }
    }
};

template <typename data_t, typename wsp_data_t, impl::data_type_t d_type>
class bwd_pooling_transpose_facade_t
    : public transpose_facade_base_t<wsp_data_t, d_type> {
public:
    bwd_pooling_transpose_facade_t(const jit_pool_conf_t &jpp,
            trans_context_t *trans_ctx, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &dst_d,
            const memory_desc_wrapper &indices_d, const data_type_t wsp_dt,
            data_t *src, const data_t *dst, const char *indices,
            const exec_ctx_t &ctx)
        : transpose_facade_base_t<wsp_data_t, d_type>(
                jpp, src_d, dst_d, indices_d, indices, wsp_dt, ctx)
        , c_tail_(jpp.c_without_padding % jpp.c_block) {

        if (this->should_transpose_src())
            this->execute_transpose_output_
                    = transpose_block_fmt_to_ncsp_t<wsp_data_t, data_t>(
                            trans_ctx->src_trans_.get(),
                            trans_ctx->src_tail_trans_.get(),
                            this->cvt_slice_src_wsp_, this->src_slice_, src,
                            this->src_d_, jpp, 1u);

        if (this->should_transpose_dst()) {
            using namespace std::placeholders;

            this->execute_transpose_input_ = std::bind(
                    [=](const transpose_ncsp_to_block_fmt_t<data_t, wsp_data_t>
                                    &trans_dst,
                            transpose_ncsp_to_block_fmt_t<char, char>
                                    &trans_indices,
                            std::size_t ithr, int n, int b_c) {
                        trans_dst(ithr, n, b_c);
                        if (indices) trans_indices(ithr, n, b_c);
                    },
                    transpose_ncsp_to_block_fmt_t<data_t, wsp_data_t>(
                            trans_ctx->dst_trans_.get(),
                            trans_ctx->dst_tail_trans_.get(), dst, this->dst_d_,
                            this->cvt_slice_dst_wsp_, this->dst_slice_, jpp),
                    transpose_ncsp_to_block_fmt_t<char, char>(
                            trans_ctx->ind_trans_.get(),
                            trans_ctx->ind_tail_trans_.get(), indices,
                            this->indices_d_, this->cvt_slice_ind_wsp_,
                            this->dst_slice_, jpp, this->ind_dt_size_),
                    _1, _2, _3);
        }
    }

    inline bool should_fill_input_c_tail_with_zeros() const noexcept {
        return this->should_transpose_dst() && c_tail_ != 0;
    }

    void fill_input_c_tail_with_zeros(
            std::size_t ithr, const jit_pool_conf_t &jpp) const {

        wsp_data_t *__restrict wsp_ptr
                = this->cvt_slice_dst_wsp_ + ithr * this->dst_slice_;
        for_(dim_t s = 0; s < this->dst_sp_; s++)
        for (dim_t c = c_tail_; c < jpp.c_block; c++)
            wsp_ptr[s * jpp.c_block + c] = 0.f;

        char *__restrict ind_ptr = this->cvt_slice_ind_wsp_
                + ithr * this->dst_slice_ * this->ind_dt_size_;
        for_(dim_t s = 0; s < this->dst_sp_; s++)
        for_(dim_t c = c_tail_; c < jpp.c_block; c++)
        for (size_t i = 0; i < this->ind_dt_size_; i++)
            ind_ptr[(s * jpp.c_block + c) * this->ind_dt_size_ + i] = 0;
    }

private:
    const dim_t c_tail_;
};

} // namespace jit_uni_pooling_utils

template <cpu_isa_t isa, impl::data_type_t d_type>
jit_uni_pooling_fwd_t<isa, d_type>::jit_uni_pooling_fwd_t(const pd_t *apd)
    : primitive_t(apd), kernel_(nullptr), trans_ctx_(nullptr) {}

template <cpu_isa_t isa, impl::data_type_t d_type>
status_t jit_uni_pooling_fwd_t<isa, d_type>::init(engine_t *engine) {

    CHECK(safe_ptr_assign(kernel_,
            new (std::nothrow) jit_uni_pool_kernel<isa>(
                    pd()->jpp_, pd()->invariant_dst_md())));

    if (pd()->jpp_.tag_kind == jptg_ncsp) CHECK(init_ncsp_trans_ctx());
    return kernel_->create_kernel();
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_pooling_fwd_t<isa, d_type>::init_ncsp_trans_ctx() {
    using namespace dnnl::impl;
    using namespace jit_uni_pooling_utils;

    const auto &jpp = pd()->jpp_;
    trans_ctx_ = utils::make_unique<trans_context_t>();
    const dim_t src_sp = static_cast<dim_t>(jpp.id) * jpp.ih * jpp.iw;
    const dim_t dst_sp = static_cast<dim_t>(jpp.od) * jpp.oh * jpp.ow;
    const auto res = std::div(jpp.c_without_padding, jpp.c_block);
    const dim_t &nb_c = res.quot;
    const dim_t &c_tail = res.rem;
    const memory_desc_wrapper indices_d = pd()->workspace_md();
    const bool have_indices = indices_d.data_type() != data_type::undef;
    static constexpr auto wsp_dt = wsp_dt_;

    if (nb_c) {
        trans_ctx_->src_trans_ = utils::make_unique<trans_wrapper_t>(
                d_type, src_sp, wsp_dt, jpp.c_block, jpp.c_block, src_sp);
        trans_ctx_->dst_trans_ = utils::make_unique<trans_wrapper_t>(
                wsp_dt, jpp.c_block, d_type, dst_sp, dst_sp, jpp.c_block);
        if (have_indices)
            trans_ctx_->ind_trans_ = utils::make_unique<trans_wrapper_t>(
                    indices_d.data_type(), jpp.c_block, indices_d.data_type(),
                    dst_sp, dst_sp, jpp.c_block);
    }

    if (c_tail) {
        trans_ctx_->src_tail_trans_ = utils::make_unique<trans_wrapper_t>(
                d_type, src_sp, wsp_dt, jpp.c_block, c_tail, src_sp);
        trans_ctx_->dst_tail_trans_ = utils::make_unique<trans_wrapper_t>(
                wsp_dt, jpp.c_block, d_type, dst_sp, dst_sp, c_tail);
        if (have_indices)
            trans_ctx_->ind_tail_trans_ = utils::make_unique<trans_wrapper_t>(
                    indices_d.data_type(), jpp.c_block, indices_d.data_type(),
                    dst_sp, dst_sp, c_tail);
    }

    return trans_ctx_->create_kernel();
}

template <cpu_isa_t isa, impl::data_type_t d_type>
jit_uni_pooling_fwd_t<isa, d_type>::~jit_uni_pooling_fwd_t() = default;

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pooling_fwd_t<isa, d_type>::execute_forward(const data_t *src,
        data_t *dst, char *indices, const exec_ctx_t &ctx) const {

    const memory_desc_wrapper src_d = pd()->src_md();
    const memory_desc_wrapper dst_d = pd()->dst_md();
    const memory_desc_wrapper indices_d = pd()->workspace_md();
    const auto ind_dt_size
            = indices ? types::data_type_size(indices_d.data_type()) : 0;
    const auto &jpp = pd()->jpp_;
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(jpp.post_ops, ctx);

    using wsp_data_t = typename prec_traits<wsp_dt_>::type;
    using namespace jit_uni_pooling_utils;

    const auto transpose_facade
            = fwd_pooling_transpose_facade_t<data_t, wsp_data_t, d_type>(jpp,
                    trans_ctx_.get(), src_d, dst_d, indices_d, wsp_dt_, src,
                    dst, indices, ctx);

    const auto trans_src = transpose_facade.should_transpose_src();
    const auto trans_dst = transpose_facade.should_transpose_dst();

    const auto ker = [&](std::size_t ithr, int n, int b_c, int oh, int ur_bc) {
        assert(ur_bc == jpp.ur_bc || ur_bc == jpp.ur_bc_tail);
        auto arg = jit_pool_call_s();

        const int ij = oh * jpp.stride_h;
        const int i_t_overflow = nstl::max(0, jpp.t_pad - ij);
        const int i_b_overflow
                = nstl::max(jpp.ih, ij + jpp.kh - jpp.t_pad) - jpp.ih;
        const int ih = nstl::max(ij - jpp.t_pad, 0);
        assert(IMPLICATION(pd()->ndims() == 3, utils::everyone_is(0, ih, oh)));
        const int c_off = ((jpp.tag_kind == jptg_nspc) ? jpp.c_block : 1) * b_c;
        const int c_elem_off = jpp.c_block * b_c;

        if (trans_src)
            arg.src = transpose_facade.get_src_addr(ithr, ih, jpp);
        else
            arg.src = static_cast<const void *>(
                    &src[src_d.blk_off(n, c_off, ih)]);

        if (trans_dst)
            arg.dst = transpose_facade.get_dst_addr(ithr, oh, jpp);
        else
            arg.dst = static_cast<const void *>(
                    &dst[dst_d.blk_off(n, c_off, oh)]);

        if (indices) {
            if (trans_dst)
                arg.indices = transpose_facade.get_indices_addr(ithr, oh, jpp);
            else {
                const size_t ind_off = indices_d.blk_off(n, c_off, oh);
                arg.indices = static_cast<const void *>(
                        &indices[ind_off * ind_dt_size]);
            }
        }
        arg.kh_padding = jpp.kh - i_t_overflow - i_b_overflow;
        arg.kh_padding_shift = i_t_overflow * jpp.kw;
        arg.ker_area_h = static_cast<float>(jpp.kh
                - nstl::max(0, oh * jpp.stride_h - jpp.t_pad + jpp.kh - jpp.ih)
                - nstl::max(0, jpp.t_pad - oh * jpp.stride_h));
        arg.ur_bc = ur_bc;
        arg.b_c = b_c;
        arg.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();
        arg.c_elem_off = c_elem_off;
        (*kernel_)(&arg);
    };

    if (jpp.tag_kind == jptg_nspc) {
        const auto nb2_c = utils::div_up(jpp.nb_c, jpp.ur_bc);
        parallel_nd(jpp.mb, jpp.oh, nb2_c, [&](int n, int oh, int b2_c) {
            const auto b_c = b2_c * jpp.ur_bc;
            const auto ur_bc = nstl::min(jpp.ur_bc, jpp.nb_c - b_c);
            ker(0, n, b_c, oh, ur_bc);
        });
    } else {
        if (trans_src || trans_dst) {
            // ncsp format
            parallel_nd_ext(0, jpp.mb, jpp.nb_c,
                    [&](int ithr, int nthr, int n, int b_c) {
                        if (trans_src)
                            transpose_facade.execute_transpose_input(
                                    ithr, n, b_c);
                        for (int oh = 0; oh < jpp.oh; ++oh)
                            ker(ithr, n, b_c, oh, 1);
                        if (trans_dst)
                            transpose_facade.execute_transpose_output(
                                    ithr, n, b_c);
                    });
        } else {
            // nChw16c, nChw8c format
            parallel(0, [&](std::size_t ithr, std::size_t nthr) {
                const std::size_t work_amount
                        = static_cast<std::size_t>(jpp.mb) * jpp.nb_c * jpp.oh;
                if (ithr >= work_amount) return;

                std::size_t start {0}, end {0};
                int n {0}, b_c {0}, oh {0};

                balance211(work_amount, nthr, ithr, start, end);
                utils::nd_iterator_init(
                        start, n, jpp.mb, b_c, jpp.nb_c, oh, jpp.oh);

                for (std::size_t iwork = start; iwork < end; ++iwork) {
                    ker(ithr, n, b_c, oh, 1);
                    utils::nd_iterator_step(
                            n, jpp.mb, b_c, jpp.nb_c, oh, jpp.oh);
                }
            });
        }
    }
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pooling_fwd_t<isa, d_type>::execute_forward_3d(const data_t *src,
        data_t *dst, char *indices, const exec_ctx_t &ctx) const {

    const auto &jpp = pd()->jpp_;
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper indices_d(pd()->workspace_md());
    const size_t ind_dt_size
            = indices ? types::data_type_size(indices_d.data_type()) : 0;
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(jpp.post_ops, ctx);

    using wsp_data_t = typename prec_traits<wsp_dt_>::type;
    using namespace jit_uni_pooling_utils;
    static constexpr int first_ithr = 0;

    const auto transpose_facade
            = fwd_pooling_transpose_facade_t<data_t, wsp_data_t, d_type>(jpp,
                    trans_ctx_.get(), src_d, dst_d, indices_d, wsp_dt_, src,
                    dst, indices, ctx);

    const auto trans_src = transpose_facade.should_transpose_src();
    const auto trans_dst = transpose_facade.should_transpose_dst();

    auto ker = [&](int n, int b_c, int od, int oh, int id, int d_t_overflow,
                       int d_b_overflow, int ur_bc, int ithr) {
        assert(ur_bc == jpp.ur_bc || ur_bc == jpp.ur_bc_tail);
        auto arg = jit_pool_call_s();

        const int ij = oh * jpp.stride_h;
        const int i_t_overflow = nstl::max(0, jpp.t_pad - ij);
        const int i_b_overflow
                = nstl::max(jpp.ih, ij + jpp.kh - jpp.t_pad) - jpp.ih;
        const int ih = nstl::max(ij - jpp.t_pad, 0);
        const int c_off = ((jpp.tag_kind == jptg_nspc) ? jpp.c_block : 1) * b_c;

        if (trans_src)
            arg.src = transpose_facade.get_src_addr_3d(ithr, id, ih, jpp);
        else
            arg.src = &src[src_d.blk_off(n, c_off, id, ih)];

        if (trans_dst)
            arg.dst = transpose_facade.get_dst_addr_3d(ithr, od, oh, jpp);
        else
            arg.dst = &dst[dst_d.blk_off(n, c_off, od, oh)];

        if (indices) {
            if (trans_dst) {
                arg.indices = transpose_facade.get_indices_addr_3d(
                        ithr, od, oh, jpp);
            } else {
                const size_t ind_off = indices_d.blk_off(n, c_off, od, oh);
                arg.indices = &indices[ind_off * ind_dt_size];
            }
        }

        arg.kd_padding = jpp.kd - d_t_overflow - d_b_overflow;
        arg.kh_padding = jpp.kh - i_t_overflow - i_b_overflow;
        arg.kh_padding_shift
                = i_t_overflow * jpp.kw + d_t_overflow * jpp.kw * jpp.kh;
        arg.kd_padding_shift = (i_t_overflow + i_b_overflow) * jpp.kw;
        arg.ker_area_h = (float)(jpp.kh
                                 - nstl::max(0,
                                         oh * jpp.stride_h - jpp.t_pad + jpp.kh
                                                 - jpp.ih)
                                 - nstl::max(0, jpp.t_pad - oh * jpp.stride_h))
                * (jpp.kd
                        - nstl::max(0,
                                od * jpp.stride_d - jpp.f_pad + jpp.kd - jpp.id)
                        - nstl::max(0, jpp.f_pad - od * jpp.stride_d));

        arg.ur_bc = ur_bc;
        arg.b_c = b_c;
        arg.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();
        arg.c_elem_off = jpp.c_block * b_c;
        (*kernel_)(&arg);
    };

    if (jpp.tag_kind == jptg_nspc) {
        const auto nb2_c = utils::div_up(jpp.nb_c, jpp.ur_bc);
        parallel_nd(jpp.mb, jpp.od, nb2_c, [&](int n, int od, int b2_c) {
            const auto b_c = b2_c * jpp.ur_bc;
            const auto ur_bc = nstl::min(jpp.ur_bc, jpp.nb_c - b_c);

            const int ik = od * jpp.stride_d;
            const int d_t_overflow = nstl::max(0, jpp.f_pad - ik);
            const int d_b_overflow
                    = nstl::max(jpp.id, ik + jpp.kd - jpp.f_pad) - jpp.id;
            const int id = nstl::max(ik - jpp.f_pad, 0);
            for (int oh = 0; oh < jpp.oh; ++oh) {
                ker(n, b_c, od, oh, id, d_t_overflow, d_b_overflow, ur_bc,
                        first_ithr);
            }
        });
    } else {
        if (trans_src || trans_dst) {
            parallel_nd_ext(0, jpp.mb, jpp.nb_c,
                    [&](int ithr, int nthr, int n, int b_c) {
                        if (trans_src)
                            transpose_facade.execute_transpose_input(
                                    ithr, n, b_c);

                        for (int od = 0; od < jpp.od; ++od) {
                            const int ik = od * jpp.stride_d;
                            const int d_t_overflow
                                    = nstl::max(0, jpp.f_pad - ik);
                            const int d_b_overflow
                                    = nstl::max(jpp.id, ik + jpp.kd - jpp.f_pad)
                                    - jpp.id;
                            const int id = nstl::max(ik - jpp.f_pad, 0);
                            for (int oh = 0; oh < jpp.oh; ++oh) {
                                ker(n, b_c, od, oh, id, d_t_overflow,
                                        d_b_overflow, 1, ithr);
                            }
                        }

                        if (trans_dst)
                            transpose_facade.execute_transpose_output(
                                    ithr, n, b_c);
                    });
        } else {
            parallel_nd(jpp.mb, jpp.nb_c, jpp.od, [&](int n, int b_c, int od) {
                const int ik = od * jpp.stride_d;
                const int d_t_overflow = nstl::max(0, jpp.f_pad - ik);
                const int d_b_overflow
                        = nstl::max(jpp.id, ik + jpp.kd - jpp.f_pad) - jpp.id;
                const int id = nstl::max(ik - jpp.f_pad, 0);
                for (int oh = 0; oh < jpp.oh; ++oh) {
                    ker(n, b_c, od, oh, id, d_t_overflow, d_b_overflow, 1,
                            first_ithr);
                }
            });
        }
    }
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_pooling_bwd_t<isa, d_type>::jit_uni_pooling_bwd_t(const pd_t *apd)
    : primitive_t(apd)
    , kernel_(utils::make_unique<jit_uni_pool_kernel<isa>>(
              pd()->jpp_, pd()->invariant_dst_md()))
    , trans_ctx_(nullptr) {}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_pooling_bwd_t<isa, d_type>::~jit_uni_pooling_bwd_t() = default;

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_pooling_bwd_t<isa, d_type>::init_ncsp_trans_ctx() {
    using namespace dnnl::impl;
    using namespace jit_uni_pooling_utils;

    const auto &jpp = pd()->jpp_;
    trans_ctx_ = utils::make_unique<trans_context_t>();
    const dim_t diff_src_sp = static_cast<dim_t>(jpp.id) * jpp.ih * jpp.iw;
    const dim_t diff_dst_sp = static_cast<dim_t>(jpp.od) * jpp.oh * jpp.ow;
    const auto res = std::div(jpp.c_without_padding, jpp.c_block);
    const dim_t &nb_c = res.quot;
    const dim_t &c_tail = res.rem;
    const memory_desc_wrapper indices_d = pd()->workspace_md();
    const bool have_indices = indices_d.data_type() != data_type::undef;
    static constexpr auto wsp_dt = wsp_dt_;

    if (nb_c) {
        trans_ctx_->dst_trans_ = utils::make_unique<trans_wrapper_t>(d_type,
                diff_dst_sp, wsp_dt, jpp.c_block, jpp.c_block, diff_dst_sp);
        trans_ctx_->src_trans_ = utils::make_unique<trans_wrapper_t>(wsp_dt,
                jpp.c_block, d_type, diff_src_sp, diff_src_sp, jpp.c_block);
        if (have_indices)
            trans_ctx_->ind_trans_ = utils::make_unique<trans_wrapper_t>(
                    indices_d.data_type(), diff_dst_sp, indices_d.data_type(),
                    jpp.c_block, jpp.c_block, diff_dst_sp);
    }
    if (c_tail) {
        trans_ctx_->dst_tail_trans_ = utils::make_unique<trans_wrapper_t>(
                d_type, diff_dst_sp, wsp_dt, jpp.c_block, c_tail, diff_dst_sp);
        trans_ctx_->src_tail_trans_ = utils::make_unique<trans_wrapper_t>(
                wsp_dt, jpp.c_block, d_type, diff_src_sp, diff_src_sp, c_tail);
        if (have_indices)
            trans_ctx_->ind_tail_trans_ = utils::make_unique<trans_wrapper_t>(
                    indices_d.data_type(), diff_dst_sp, indices_d.data_type(),
                    jpp.c_block, c_tail, diff_dst_sp);
    }

    return trans_ctx_->create_kernel();
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_pooling_bwd_t<isa, d_type>::init(engine_t *engine) {
    if (pd()->jpp_.tag_kind == jptg_ncsp) CHECK(init_ncsp_trans_ctx());
    return kernel_->create_kernel();
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pooling_bwd_t<isa, d_type>::execute_backward(
        const data_t *diff_dst, const char *indices, data_t *diff_src,
        const exec_ctx_t &ctx) const {

    using namespace jit_uni_pooling_utils;
    using wsp_data_t = typename prec_traits<wsp_dt_>::type;

    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper indices_d(pd()->workspace_md());
    const size_t ind_dt_size
            = indices ? types::data_type_size(indices_d.data_type()) : 0;
    const auto &jpp = pd()->jpp_;
    const auto transpose_facade
            = jit_uni_pooling_utils::bwd_pooling_transpose_facade_t<data_t,
                    wsp_data_t, d_type>(jpp, trans_ctx_.get(), diff_src_d,
                    diff_dst_d, indices_d, wsp_dt_, diff_src, diff_dst, indices,
                    ctx);

    auto get_first_ih = [&](int oh) {
        return nstl::min(nstl::max(oh * jpp.stride_h - jpp.t_pad, 0), jpp.ih);
    };

    auto get_last_ih = [&](int oh) {
        return nstl::min(
                nstl::max(oh * jpp.stride_h - jpp.t_pad + jpp.kh, 0), jpp.ih);
    };
    const auto ker = [&](int ithr, int n, int b_c, int oh, int ur_bc) {
        auto arg = jit_pool_call_s();

        const int ih = get_first_ih(oh);
        assert(IMPLICATION(pd()->ndims() == 3, utils::everyone_is(0, ih, oh)));
        assert(pd()->ndims() != 3 || utils::everyone_is(0, ih, oh));

        const auto c_off = jpp.is_plain() ? b_c * jpp.c_block : b_c;
        if (transpose_facade.should_transpose_src())
            arg.src = transpose_facade.get_src_addr(ithr, ih, jpp);
        else
            arg.src = &diff_src[diff_src_d.blk_off(n, c_off, ih)];

        if (transpose_facade.should_transpose_dst())
            arg.dst = transpose_facade.get_dst_addr(ithr, oh, jpp);
        else
            arg.dst = &diff_dst[diff_dst_d.blk_off(n, c_off, oh)];

        if (indices) {
            if (transpose_facade.should_transpose_dst())
                arg.indices = transpose_facade.get_indices_addr(ithr, oh, jpp);

            else {
                const size_t ind_off = indices_d.blk_off(n, c_off, oh);
                arg.indices = &indices[ind_off * ind_dt_size];
            }
        }

        const int zero_ih_start = (oh == 0) ? 0 : get_last_ih(oh - 1);
        const int zero_ih_end = (oh == jpp.oh - 1) ? jpp.ih : get_last_ih(oh);

        arg.zero_id = 1;
        arg.zero_ih = zero_ih_end - zero_ih_start;
        if (transpose_facade.should_transpose_src())
            arg.zero_ptr
                    = transpose_facade.get_src_addr(ithr, zero_ih_start, jpp);
        else
            arg.zero_ptr
                    = &diff_src[diff_src_d.blk_off(n, c_off, zero_ih_start, 0)];

        const int i_t_overflow = nstl::max(0, jpp.t_pad - oh * jpp.stride_h);
        const int i_b_overflow
                = nstl::max(jpp.ih, oh * jpp.stride_h + jpp.kh - jpp.t_pad)
                - jpp.ih;
        arg.kh_padding = jpp.kh - i_t_overflow - i_b_overflow;
        arg.kh_padding_shift = i_t_overflow * jpp.kw;
        arg.ker_area_h = static_cast<float>(jpp.kh
                - nstl::max(0, oh * jpp.stride_h - jpp.t_pad + jpp.kh - jpp.ih)
                - nstl::max(0, jpp.t_pad - oh * jpp.stride_h));

        arg.ur_bc = ur_bc;
        arg.b_c = b_c;
        (*kernel_)(&arg);
    };

    auto process_block = [&](int ithr, int n, int b_c, int ur_bc) {
        if (transpose_facade.should_transpose_dst())
            transpose_facade.execute_transpose_input(ithr, n, b_c);

        for (int oh = 0; oh < jpp.oh; ++oh)
            ker(ithr, n, b_c, oh, ur_bc);

        if (transpose_facade.should_transpose_src())
            transpose_facade.execute_transpose_output(ithr, n, b_c);
    };

    parallel(0, [&](int ithr, int nthr) {
        const auto nb2_c = utils::div_up(jpp.nb_c, jpp.ur_bc);
        const std::size_t work_amount
                = static_cast<std::size_t>(jpp.mb) * nb2_c;
        if (static_cast<std::size_t>(ithr) >= work_amount) return;

        if (transpose_facade.should_fill_input_c_tail_with_zeros())
            transpose_facade.fill_input_c_tail_with_zeros(ithr, jpp);

        std::size_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        int n {0}, b2_c {0};
        utils::nd_iterator_init(start, n, jpp.mb, b2_c, nb2_c);
        for (size_t iwork = start; iwork < end; ++iwork) {
            const auto b_c = b2_c * jpp.ur_bc;
            const auto ur_bc = nstl::min(jpp.ur_bc, jpp.nb_c - b_c);

            process_block(ithr, n, b_c, ur_bc);
            utils::nd_iterator_step(n, jpp.mb, b2_c, nb2_c);
        }
    });
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pooling_bwd_t<isa, d_type>::execute_backward_3d(
        const data_t *diff_dst, const char *indices, data_t *diff_src,
        const exec_ctx_t &ctx) const {
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper indices_d(pd()->workspace_md());
    const size_t ind_dt_size
            = indices ? types::data_type_size(indices_d.data_type()) : 0;

    const auto &jpp = pd()->jpp_;

    using wsp_data_t = typename prec_traits<wsp_dt_>::type;
    using namespace jit_uni_pooling_utils;
    static constexpr int first_ithr = 0;

    const auto transpose_facade
            = bwd_pooling_transpose_facade_t<data_t, wsp_data_t, d_type>(jpp,
                    trans_ctx_.get(), diff_src_d, diff_dst_d, indices_d,
                    wsp_dt_, diff_src, diff_dst, indices, ctx);

    const auto trans_src = transpose_facade.should_transpose_src();
    const auto trans_dst = transpose_facade.should_transpose_dst();

    auto get_last_ih = [&](int oh) {
        return nstl::min(
                nstl::max(oh * jpp.stride_h - jpp.t_pad + jpp.kh, 0), jpp.ih);
    };

    auto get_last_id = [&](int od) {
        return nstl::min(
                nstl::max(od * jpp.stride_d - jpp.f_pad + jpp.kd, 0), jpp.id);
    };

    auto ker = [&](int n, int b_c, int od, int oh, int id, int d_t_overflow,
                       int d_b_overflow, bool zero_inp, int kd, int ur_bc,
                       int ithr) {
        auto arg = jit_pool_call_s();

        const int ij = oh * jpp.stride_h;
        const int i_t_overflow = nstl::max(0, jpp.t_pad - ij);
        const int i_b_overflow
                = nstl::max(jpp.ih, ij + jpp.kh - jpp.t_pad) - jpp.ih;
        const int ih = nstl::max(ij - jpp.t_pad, 0);
        const int c_off = ((jpp.tag_kind == jptg_nspc) ? jpp.c_block : 1) * b_c;

        if (trans_src)
            arg.src = transpose_facade.get_src_addr_3d(ithr, id + kd, ih, jpp);
        else
            arg.src = (const void *)&diff_src[diff_src_d.blk_off(
                    n, c_off, id + kd, ih)];

        if (trans_dst)
            arg.dst = transpose_facade.get_dst_addr_3d(ithr, od, oh, jpp);
        else
            arg.dst = (const void
                            *)&diff_dst[diff_dst_d.blk_off(n, c_off, od, oh)];

        if (indices) {
            if (trans_dst) {
                arg.indices = transpose_facade.get_indices_addr_3d(
                        ithr, od, oh, jpp);
            } else {
                const size_t ind_off = indices_d.blk_off(n, c_off, od, oh);
                arg.indices = (const void *)&indices[ind_off * ind_dt_size];
            }
        }

        if (zero_inp) {
            const int zero_id_start = (od == 0) ? 0 : get_last_id(od - 1);
            const int zero_id_end
                    = (od == jpp.od - 1) ? jpp.id : get_last_id(od);

            arg.zero_id = zero_id_end - zero_id_start;

            const int zero_ih_start = (oh == 0) ? 0 : get_last_ih(oh - 1);
            const int zero_ih_end
                    = (oh == jpp.oh - 1) ? jpp.ih : get_last_ih(oh);
            arg.zero_ih = zero_ih_end - zero_ih_start;

            if (trans_src)
                arg.zero_ptr = transpose_facade.get_src_addr_3d(
                        ithr, zero_id_start, zero_ih_start, jpp);
            else
                arg.zero_ptr = &diff_src[diff_src_d.blk_off(
                        n, c_off, zero_id_start, zero_ih_start, 0)];
        } else {
            arg.zero_id = 0;
            arg.zero_ih = 0;
        }

        arg.kd_padding = jpp.kd - d_t_overflow - d_b_overflow;
        arg.kh_padding = jpp.kh - i_t_overflow - i_b_overflow;
        arg.kh_padding_shift = i_t_overflow * jpp.kw
                + d_t_overflow * jpp.kw * jpp.kh + kd * jpp.kw * jpp.kh;
        arg.kd_padding_shift = (i_t_overflow + i_b_overflow) * jpp.kw;
        arg.ker_area_h = (float)(jpp.kh
                                 - nstl::max(0,
                                         oh * jpp.stride_h - jpp.t_pad + jpp.kh
                                                 - jpp.ih)
                                 - nstl::max(0, jpp.t_pad - oh * jpp.stride_h))
                * (jpp.kd
                        - nstl::max(0,
                                od * jpp.stride_d - jpp.f_pad + jpp.kd - jpp.id)
                        - nstl::max(0, jpp.f_pad - od * jpp.stride_d));

        arg.ur_bc = ur_bc;
        arg.b_c = b_c;
        (*kernel_)(&arg);
    };

    auto process_simple = [&](int n, int b_c, int od, int ur_bc, int ithr) {
        const int ik = od * jpp.stride_d;
        const int d_t_overflow = nstl::max(0, jpp.f_pad - ik);
        const int d_b_overflow
                = nstl::max(jpp.id, ik + jpp.kd - jpp.f_pad) - jpp.id;
        const int id = nstl::max(ik - jpp.f_pad, 0);

        for (int oh = 0; oh < jpp.oh; ++oh) {
            ker(n, b_c, od, oh, id, d_t_overflow, d_b_overflow, true, 0, ur_bc,
                    ithr);
        }
    };

    if (jpp.simple_alg) {
        if (jpp.tag_kind == jptg_nspc) {
            const auto nb2_c = utils::div_up(jpp.nb_c, jpp.ur_bc);
            parallel_nd(jpp.mb, jpp.od, nb2_c, [&](int n, int od, int b2_c) {
                const auto b_c = b2_c * jpp.ur_bc;
                const auto ur_bc = nstl::min(jpp.ur_bc, jpp.nb_c - b_c);
                process_simple(n, b_c, od, ur_bc, first_ithr);
            });
        } else {
            assert(jpp.ur_bc == 1);
            if (trans_src || trans_dst) {
                parallel_nd_ext(0, jpp.mb, jpp.nb_c,
                        [&](int ithr, int nthr, int n, int b_c) {
                            if (trans_src)
                                transpose_facade.execute_transpose_input(
                                        ithr, n, b_c);
                            for (int od = 0; od < jpp.od; ++od) {
                                process_simple(n, b_c, od, 1, ithr);
                            }
                            if (trans_dst)
                                transpose_facade.execute_transpose_output(
                                        ithr, n, b_c);
                        });
            } else {
                parallel_nd(
                        jpp.mb, jpp.nb_c, jpp.od, [&](int n, int b_c, int od) {
                            process_simple(n, b_c, od, 1, first_ithr);
                        });
            }
        }
    } else {
        const data_t zero_val = 0;
        if (jpp.tag_kind == jptg_nspc) {
            const size_t chunk_size = (size_t)jpp.ih * jpp.iw * jpp.c;
            parallel_nd(jpp.mb, jpp.id, [&](int n, int id) {
                const size_t offset = ((size_t)n * jpp.id + id) * chunk_size;
                PRAGMA_OMP_SIMD()
                for (size_t idx = 0; idx < chunk_size; ++idx)
                    diff_src[offset + idx] = zero_val;
            });
        } else {
            if (!trans_src) {
                const size_t chunk_size
                        = (size_t)jpp.id * jpp.ih * jpp.iw * jpp.c_block;
                parallel_nd_ext(0, jpp.mb, jpp.nb_c,
                        [&](int ithr, int nthr, int n, int b_c) {
                            const size_t offset
                                    = ((size_t)n * jpp.nb_c + b_c) * chunk_size;
                            PRAGMA_OMP_SIMD()
                            for (size_t idx = 0; idx < chunk_size; ++idx)
                                diff_src[offset + idx] = zero_val;
                        });
            }
        }

        const auto nb2_c = utils::div_up(jpp.nb_c, jpp.ur_bc);
        if (trans_src || trans_dst) {
            parallel_nd_ext(
                    0, jpp.mb, nb2_c, [&](int ithr, int nthr, int n, int b2_c) {
                        const auto b_c = b2_c * jpp.ur_bc;

                        if (trans_dst) {
                            transpose_facade.execute_transpose_input(
                                    ithr, n, b_c);

                            size_t block_size = jpp.c_block * jpp.id * jpp.ih
                                    * jpp.iw * jpp.dt_size;

                            const void *src = transpose_facade.get_src_addr_3d(
                                    ithr, 0, 0, jpp);
                            std::memset((void *)src, zero_val, block_size);
                        }

                        for (int kd = 0; kd < jpp.kd; ++kd) {
                            const auto ur_bc
                                    = nstl::min(jpp.ur_bc, jpp.nb_c - b_c);
                            for (int od = 0; od < jpp.od; ++od) {
                                const int ik = od * jpp.stride_d;
                                const int d_t_overflow
                                        = nstl::max(0, jpp.f_pad - ik);
                                const int d_b_overflow
                                        = nstl::max(jpp.id,
                                                  ik + jpp.kd - jpp.f_pad)
                                        - jpp.id;
                                if (kd >= jpp.kd - d_t_overflow - d_b_overflow)
                                    continue;
                                const int id = nstl::max(ik - jpp.f_pad, 0);
                                for (int oh = 0; oh < jpp.oh; ++oh) {
                                    ker(n, b_c, od, oh, id, d_t_overflow,
                                            d_b_overflow, false, kd, ur_bc,
                                            ithr);
                                }
                            }
                        }

                        if (trans_src)
                            transpose_facade.execute_transpose_output(
                                    ithr, n, b_c);
                    });
        } else {
            for (int kd = 0; kd < jpp.kd; ++kd) {
                parallel_nd(jpp.mb, nb2_c, [&](int n, int b2_c) {
                    const auto b_c = b2_c * jpp.ur_bc;
                    const auto ur_bc = nstl::min(jpp.ur_bc, jpp.nb_c - b_c);
                    for (int od = 0; od < jpp.od; ++od) {
                        const int ik = od * jpp.stride_d;
                        const int d_t_overflow = nstl::max(0, jpp.f_pad - ik);
                        const int d_b_overflow
                                = nstl::max(jpp.id, ik + jpp.kd - jpp.f_pad)
                                - jpp.id;
                        if (kd >= jpp.kd - d_t_overflow - d_b_overflow)
                            continue;
                        const int id = nstl::max(ik - jpp.f_pad, 0);
                        for (int oh = 0; oh < jpp.oh; ++oh) {
                            ker(n, b_c, od, oh, id, d_t_overflow, d_b_overflow,
                                    false, kd, ur_bc, first_ithr);
                        }
                    }
                });
            }
        }
    }
}

template struct jit_uni_pooling_fwd_t<sse41, data_type::f32>;
template struct jit_uni_pooling_bwd_t<sse41, data_type::f32>;
template struct jit_uni_pooling_fwd_t<avx, data_type::f32>;
template struct jit_uni_pooling_bwd_t<avx, data_type::f32>;
template struct jit_uni_pooling_fwd_t<avx2, data_type::f32>;
template struct jit_uni_pooling_bwd_t<avx2, data_type::f32>;
template struct jit_uni_pooling_fwd_t<avx512_common, data_type::f32>;
template struct jit_uni_pooling_bwd_t<avx512_common, data_type::f32>;
template struct jit_uni_pooling_fwd_t<avx512_core, data_type::f32>;
template struct jit_uni_pooling_bwd_t<avx512_core, data_type::f32>;
template struct jit_uni_pooling_fwd_t<avx512_core, data_type::bf16>;
template struct jit_uni_pooling_bwd_t<avx512_core, data_type::bf16>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
