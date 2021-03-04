/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

/// @file
/// C++ API

#ifndef ONEAPI_DNNL_DNNL_HPP
#define ONEAPI_DNNL_DNNL_HPP

#include "oneapi/dnnl/dnnl_config.h"

/// @cond DO_NOT_DOCUMENT_THIS
#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.h"

/// @endcond

// __cpp_exceptions is referred from
// https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_exceptions.html
// gcc < 5 does not define __cpp_exceptions but __EXCEPTIONS,
// Microsoft C++ Compiler does not provide an option to disable exceptions
#ifndef DNNL_ENABLE_EXCEPTIONS
#if __cpp_exceptions || __EXCEPTIONS \
        || (defined(_MSC_VER) && !defined(__clang__))
#define DNNL_ENABLE_EXCEPTIONS 1
#else
#define DNNL_ENABLE_EXCEPTIONS 0
#endif
#endif

#if defined(__GNUC__) || defined(__clang__)
#define DNNL_TRAP() __builtin_trap()
#elif defined(__INTEL_COMPILER) || defined(_MSC_VER)
#define DNNL_TRAP() __debugbreak()
#else
#error "unknown compiler"
#endif

#if DNNL_ENABLE_EXCEPTIONS
#define DNNL_THROW_ERROR(status, msg) throw error(status, msg)
#else
#include <cstdio>
#define DNNL_THROW_ERROR(status, msg) \
    do { \
        fputs(msg, stderr); \
        DNNL_TRAP(); \
    } while (0)
#endif

/// @addtogroup dnnl_api oneDNN API
/// @{

/// oneDNN namespace
namespace dnnl {

/// @addtogroup dnnl_api_utils Utilities
/// Utility types and definitions.
/// @{

/// oneDNN exception class.
///
/// This class captures the status returned by a failed C API function and
/// the error message from the call site.
struct error : public std::exception {
    dnnl_status_t status;
    const char *message;

    /// Constructs an instance of an exception class.
    ///
    /// @param status The error status returned by a C API function.
    /// @param message The error message.
    error(dnnl_status_t status, const char *message)
        : status(status), message(message) {}

    /// Returns the explanatory string.
    const char *what() const noexcept override { return message; }

    /// A convenience function for wrapping calls to C API functions. Checks
    /// the return status and throws an dnnl::error in case of failure.
    ///
    /// @param status The error status returned by a C API function.
    /// @param message The error message.
    static void wrap_c_api(dnnl_status_t status, const char *message) {
        if (status != dnnl_success) DNNL_THROW_ERROR(status, message);
    }
};

/// @cond DO_NOT_DOCUMENT_THIS
template <typename T>
void validate_container_size(const T &v, const char *error_message,
        int min_size = 1, int max_size = -1) {
    const int size = (int)v.size();
    if (size < min_size || (max_size >= 0 && size > max_size))
        DNNL_THROW_ERROR(dnnl_invalid_arguments, error_message);
}
/// @endcond

/// A class that provides the destructor for a oneDNN C API handle.
template <typename T>
struct handle_traits {};

/// oneDNN C API handle wrapper class.
///
/// This class is used as the base class for primitive (dnnl::primitive),
/// engine (dnnl::engine), and stream (dnnl::stream) classes, as well as
/// others. An object of the dnnl::handle class can be passed by value.
///
/// A handle can be weak, in which case it follows std::weak_ptr semantics.
/// Otherwise, it follows `std::shared_ptr` semantics.
///
/// @note
///     The implementation stores oneDNN C API handles in a `std::shared_ptr`
///     with deleter set to a dummy function in the weak mode.
///
template <typename T, typename traits = handle_traits<T>>
struct handle {
private:
    static dnnl_status_t dummy_destructor(T) { return dnnl_success; }
    std::shared_ptr<typename std::remove_pointer<T>::type> data_ {0};

protected:
    bool operator==(const T other) const { return other == data_.get(); }
    bool operator!=(const T other) const { return !(*this == other); }

public:
    /// Constructs an empty handle object.
    ///
    /// @warning
    ///     Uninitialized object cannot be used in most library calls and is
    ///     equivalent to a null pointer. Any attempt to use its methods, or
    ///     passing it to the other library function, will cause an exception
    ///     to be thrown.
    handle() = default;

    /// Copy constructor.
    handle(const handle<T, traits> &) = default;
    /// Assignment operator.
    handle<T, traits> &operator=(const handle<T, traits> &) = default;
    /// Move constructor.
    handle(handle<T, traits> &&) = default;
    /// Move assignment operator.
    handle<T, traits> &operator=(handle<T, traits> &&) = default;

    /// Constructs a handle wrapper object from a C API handle.
    ///
    /// @param t The C API handle to wrap.
    /// @param weak A flag specifying whether to construct a weak wrapper;
    ///     defaults to @c false.
    explicit handle(T t, bool weak = false) { reset(t, weak); }

    /// Resets the handle wrapper objects to wrap a new C API handle.
    ///
    /// @param t The new value of the C API handle.
    /// @param weak A flag specifying whether the wrapper should be weak;
    ///     defaults to @c false.
    void reset(T t, bool weak = false) {
        data_.reset(t, weak ? &dummy_destructor : traits::destructor);
    }

    /// Returns the underlying C API handle.
    ///
    /// @param allow_empty A flag signifying whether the method is allowed to
    ///     return an empty (null) object without throwing an exception.
    /// @returns The underlying C API handle.
    T get(bool allow_empty = false) const {
        T result = data_.get();
        if (allow_empty == false && result == nullptr)
            DNNL_THROW_ERROR(
                    dnnl_invalid_arguments, "object is not initialized");
        return result;
    }

    /// Converts a handle to the underlying C API handle type. Does not throw
    /// and returns `nullptr` if the object is empty.
    ///
    /// @returns The underlying C API handle.
    explicit operator T() const { return get(true); }

    /// Checks whether the object is not empty.
    ///
    /// @returns Whether the object is not empty.
    explicit operator bool() const { return get(true) != nullptr; }

    /// Equality operator.
    ///
    /// @param other Another handle wrapper.
    /// @returns @c true if this and the other handle wrapper manage the same
    ///     underlying C API handle, and @c false otherwise. Empty handle
    ///     objects are considered to be equal.
    bool operator==(const handle<T, traits> &other) const {
        return other.data_.get() == data_.get();
    }

    /// Inequality operator.
    ///
    /// @param other Another handle wrapper.
    /// @returns @c true if this and the other handle wrapper manage different
    ///     underlying C API handles, and @c false otherwise. Empty handle
    ///     objects are considered to be equal.
    bool operator!=(const handle &other) const { return !(*this == other); }
};

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<dnnl_memory_t> {
    static dnnl_status_t destructor(dnnl_memory_t p) {
        return dnnl_memory_destroy(p);
    }
};

template <>
struct handle_traits<dnnl_primitive_desc_t> {
    static dnnl_status_t destructor(dnnl_primitive_desc_t p) {
        return dnnl_primitive_desc_destroy(p);
    }
};

template <>
struct handle_traits<dnnl_primitive_t> {
    static dnnl_status_t destructor(dnnl_primitive_t p) {
        return dnnl_primitive_destroy(p);
    }
};

template <>
struct handle_traits<dnnl_primitive_desc_iterator_t> {
    static dnnl_status_t destructor(dnnl_primitive_desc_iterator_t p) {
        return dnnl_primitive_desc_iterator_destroy(p);
    }
};
/// @endcond

/// @} dnnl_api_utils

struct stream;
struct memory;
struct primitive_desc;

/// @addtogroup dnnl_api_primitives Primitives
/// Compute primitives
/// @sa @ref dev_guide_basic_concepts
/// @{

/// @addtogroup dnnl_api_primitives_common Common
/// Common operations to create, destroy and inspect primitives
/// @{

/// Base class for all computational primitives.
struct primitive : public handle<dnnl_primitive_t> {
    /// Kinds of primitives supported by the library.
    enum class kind {
        /// Undefined primitive
        undef = dnnl_undefined_primitive,
        /// A reorder primitive.
        reorder = dnnl_reorder,
        /// A shuffle primitive.
        shuffle = dnnl_shuffle,
        /// A (out-of-place) tensor concatenation primitive.
        concat = dnnl_concat,
        /// A summation primitive.
        sum = dnnl_sum,
        /// A convolution primitive.
        convolution = dnnl_convolution,
        /// A deconvolution primitive.
        deconvolution = dnnl_deconvolution,
        /// An element-wise primitive.
        eltwise = dnnl_eltwise,
        /// A softmax primitive.
        softmax = dnnl_softmax,
        /// A pooling primitive.
        pooling = dnnl_pooling,
        /// An LRN primitive.
        lrn = dnnl_lrn,
        /// A batch normalization primitive.
        batch_normalization = dnnl_batch_normalization,
        /// A layer normalization primitive.
        layer_normalization = dnnl_layer_normalization,
        /// An inner product primitive.
        inner_product = dnnl_inner_product,
        /// An RNN primitive.
        rnn = dnnl_rnn,
        /// A binary primitive.
        binary = dnnl_binary,
        /// A logsoftmax primitive.
        logsoftmax = dnnl_logsoftmax,
        /// A matmul (matrix multiplication) primitive.
        matmul = dnnl_matmul,
        /// A resampling primitive.
        resampling = dnnl_resampling,
        /// A pooling version 2 primitive.
        pooling_v2 = dnnl_pooling_v2,
        /// A reduction primitive.
        reduction = dnnl_reduction,
    };

    using handle::handle;

    /// Default constructor. Constructs an empty object.
    primitive() = default;

    /// Constructs a primitive from a C API primitive descriptor.
    ///
    /// @param c_pd C API primitive descriptor.
    primitive(const_dnnl_primitive_desc_t c_pd);

    /// Constructs a primitive from a primitive descriptor.
    ///
    /// @param pd Primitive descriptor.
    primitive(const primitive_desc &pd);

    /// Returns the C API primitive descriptor of the underlying C API
    /// primitive.
    ///
    /// @returns The underlying C API primitive descriptor.
    inline const_dnnl_primitive_desc_t get_primitive_desc() const;

    /// Returns the kind of the primitive.
    ///
    /// @returns The primitive kind.
    inline kind get_kind() const;

    /// Executes computations specified by the primitive in a specified stream.
    ///
    /// Arguments are passed via an arguments map containing <index,
    /// memory object> pairs. The index must be one of the `DNNL_ARG_*` values
    /// such as `DNNL_ARG_SRC`, and the memory must have a memory descriptor
    /// matching the one returned by
    /// primitive_desc::query_md(#query::exec_arg_md, index) unless using
    /// dynamic shapes (see #DNNL_RUNTIME_DIM_VAL).
    ///
    /// @param astream Stream object. The stream must belong to the same engine
    ///     as the primitive.
    /// @param args Arguments map.
    void execute(const stream &astream,
            const std::unordered_map<int, memory> &args) const;
};

/// Converts primitive kind enum value from C++ API to C API type.
///
/// @param akind C++ API primitive kind enum value.
/// @returns Corresponding C API primitive kind enum value.
inline dnnl_primitive_kind_t convert_to_c(primitive::kind akind) {
    return static_cast<dnnl_primitive_kind_t>(akind);
}

const_dnnl_primitive_desc_t primitive::get_primitive_desc() const {
    const_dnnl_primitive_desc_t pd;
    error::wrap_c_api(dnnl_primitive_get_primitive_desc(get(), &pd),
            "could not get a primitive descriptor from a primitive");
    return pd;
}

dnnl::primitive::kind primitive::get_kind() const {
    const_dnnl_primitive_desc_t pd = get_primitive_desc();
    // TODO (Roma): the code below is only needed because get_primitive_desc
    // returns a C type.
    dnnl_primitive_kind_t kind;
    error::wrap_c_api(dnnl_primitive_desc_query(
                              pd, dnnl_query_primitive_kind, 0, (void *)&kind),
            "could not get a primitive kind from a primitive descriptor");
    return static_cast<dnnl::primitive::kind>(kind);
}

/// @} dnnl_api_primitives_common

/// @addtogroup dnnl_api_attributes
///
/// A container for parameters that extend primitives behavior.
///
/// Attributes can also contain Post-ops, which are computations executed
/// after the primitive.
///
/// @sa @ref dev_guide_attributes
/// @sa @ref dev_guide_attributes_post_ops
///
/// @{

/// Scratchpad mode
enum class scratchpad_mode {
    /// The library manages the scratchpad allocation according to the policy
    /// specified by the `DNNL_ENABLE_CONCURRENT_EXEC`
    /// [build option](@ref dev_guide_build_options) (default).
    ///
    /// When `DNNL_ENABLE_CONCURRENT_EXEC=OFF` (default), the library
    /// scratchpad is common to all primitives to reduce the memory footprint.
    /// This configuration comes with limited thread-safety properties, namely
    /// primitives can be created and executed in parallel but cannot migrate
    /// between threads (in other words, each primitive should be executed in
    /// the same thread it was created in).
    ///
    /// When `DNNL_ENABLE_CONCURRENT_EXEC=ON`, the library scratchpad is
    /// private to each primitive. The memory footprint is larger than when
    /// using `DNNL_ENABLE_CONCURRENT_EXEC=OFF` but different primitives can be
    /// created and run concurrently (the same primitive cannot be run
    /// concurrently from two different threads though).
    library = dnnl_scratchpad_mode_library,
    /// The user manages the scratchpad allocation by querying and providing
    /// the scratchpad memory to primitives. This mode is thread-safe as long
    /// as the scratchpad buffers are not used concurrently by two primitive
    /// executions.
    user = dnnl_scratchpad_mode_user,
};

/// Converts a scratchpad mode enum value from C++ API to C API type.
///
/// @param mode C++ API scratchpad mode enum value.
/// @returns Corresponding C API scratchpad mode enum value.
inline dnnl_scratchpad_mode_t convert_to_c(scratchpad_mode mode) {
    return static_cast<dnnl_scratchpad_mode_t>(mode);
}

/// Propagation kind.
enum class prop_kind {
    /// Undefined propagation kind.
    undef = dnnl_prop_kind_undef,
    /// Forward data propagation (training mode). In this mode, primitives
    /// perform computations necessary for subsequent backward propagation.
    forward_training = dnnl_forward_training,
    /// Forward data propagation (inference mode). In this mode, primitives
    /// perform only computations that are necessary for inference and omit
    /// computations that are necessary only for backward propagation.
    forward_inference = dnnl_forward_inference,
    /// Forward data propagation,
    /// alias for #dnnl::prop_kind::forward_inference.
    forward_scoring = dnnl_forward_scoring,
    /// Forward data propagation,
    /// alias for #dnnl::prop_kind::forward_training.
    forward = dnnl_forward,
    /// Backward propagation (with respect to all parameters).
    backward = dnnl_backward,
    /// Backward data propagation.
    backward_data = dnnl_backward_data,
    /// Backward weights propagation.
    backward_weights = dnnl_backward_weights,
    /// Backward bias propagation.
    backward_bias = dnnl_backward_bias
};

/// Converts propagation kind enum value from C++ API to C API type.
///
/// @param akind C++ API propagation kind enum value.
/// @returns Corresponding C API propagation kind enum value.
inline dnnl_prop_kind_t convert_to_c(prop_kind akind) {
    return static_cast<dnnl_prop_kind_t>(akind);
}

/// Kinds of algorithms.
enum class algorithm {
    /// Undefined algorithm
    undef = dnnl_alg_kind_undef,
    /// Convolution algorithm that is chosen to be either direct or Winograd
    /// automatically
    convolution_auto = dnnl_convolution_auto,
    /// Direct convolution
    convolution_direct = dnnl_convolution_direct,
    /// Winograd convolution
    convolution_winograd = dnnl_convolution_winograd,
    /// Direct deconvolution
    deconvolution_direct = dnnl_deconvolution_direct,
    /// Winograd deconvolution
    deconvolution_winograd = dnnl_deconvolution_winograd,
    /// Elementwise: rectified linear unit (ReLU)
    eltwise_relu = dnnl_eltwise_relu,
    /// Elementwise: hyperbolic tangent non-linearity (tanh)
    eltwise_tanh = dnnl_eltwise_tanh,
    /// Elementwise: exponential linear unit (ELU)
    eltwise_elu = dnnl_eltwise_elu,
    /// Elementwise: square
    eltwise_square = dnnl_eltwise_square,
    /// Elementwise: abs
    eltwise_abs = dnnl_eltwise_abs,
    /// Elementwise: square root
    eltwise_sqrt = dnnl_eltwise_sqrt,
    /// Elementwise: swish (\f$x \cdot sigmoid(a \cdot x)\f$)
    eltwise_swish = dnnl_eltwise_swish,
    /// Elementwise: linear
    eltwise_linear = dnnl_eltwise_linear,
    /// Elementwise: bounded_relu
    eltwise_bounded_relu = dnnl_eltwise_bounded_relu,
    /// Elementwise: soft_relu
    eltwise_soft_relu = dnnl_eltwise_soft_relu,
    /// Elementwise: logistic
    eltwise_logistic = dnnl_eltwise_logistic,
    /// Elementwise: exponent
    eltwise_exp = dnnl_eltwise_exp,
    /// Elementwise: gelu
    /// alias for #dnnl::algorithm::eltwise_gelu_tanh
    eltwise_gelu = dnnl_eltwise_gelu,
    /// Elementwise: tanh-based gelu
    eltwise_gelu_tanh = dnnl_eltwise_gelu_tanh,
    /// Elementwise: erf-based gelu
    eltwise_gelu_erf = dnnl_eltwise_gelu_erf,
    /// Elementwise: natural logarithm
    eltwise_log = dnnl_eltwise_log,
    /// Elementwise: clip
    eltwise_clip = dnnl_eltwise_clip,
    /// Elementwise: pow
    eltwise_pow = dnnl_eltwise_pow,
    /// Elementwise: round
    eltwise_round = dnnl_eltwise_round,
    /// Elementwise: rectified linar unit (ReLU) (dst for backward)
    eltwise_relu_use_dst_for_bwd = dnnl_eltwise_relu_use_dst_for_bwd,
    /// Elementwise: hyperbolic tangent non-linearity (tanh) (dst for backward)
    eltwise_tanh_use_dst_for_bwd = dnnl_eltwise_tanh_use_dst_for_bwd,
    /// Elementwise: exponential linear unit (ELU) (dst for backward)
    eltwise_elu_use_dst_for_bwd = dnnl_eltwise_elu_use_dst_for_bwd,
    /// Elementwise: square root (dst for backward)
    eltwise_sqrt_use_dst_for_bwd = dnnl_eltwise_sqrt_use_dst_for_bwd,
    /// Elementwise: logistic (dst for backward)
    eltwise_logistic_use_dst_for_bwd = dnnl_eltwise_logistic_use_dst_for_bwd,
    /// Elementwise: exponent (dst for backward)
    eltwise_exp_use_dst_for_bwd = dnnl_eltwise_exp_use_dst_for_bwd,
    /// Local response normalization (LRN) across multiple channels
    lrn_across_channels = dnnl_lrn_across_channels,
    /// LRN within a single channel
    lrn_within_channel = dnnl_lrn_within_channel,
    /// Max pooling
    pooling_max = dnnl_pooling_max,
    /// Average pooling exclude padding,
    /// alias for #dnnl::algorithm::pooling_avg_include_padding
    pooling_avg = dnnl_pooling_avg,
    /// Average pooling include padding
    pooling_avg_include_padding = dnnl_pooling_avg_include_padding,
    /// Average pooling exclude padding
    pooling_avg_exclude_padding = dnnl_pooling_avg_exclude_padding,
    /// RNN cell
    vanilla_rnn = dnnl_vanilla_rnn,
    /// LSTM cell
    vanilla_lstm = dnnl_vanilla_lstm,
    /// GRU cell
    vanilla_gru = dnnl_vanilla_gru,
    /// GRU cell with linear before reset. Differs from the vanilla GRU
    /// in how the new memory gate is calculated:
    /// \f$c_t = tanh(W_c*x_t + b_{c_x} + r_t*(U_c*h_{t-1}+b_{c_h})) \f$
    /// LRB GRU expects 4 bias tensors on input:
    /// \f$[b_{u}, b_{r}, b_{c_x}, b_{c_h}]\f$
    lbr_gru = dnnl_lbr_gru,
    /// Binary add
    binary_add = dnnl_binary_add,
    /// Binary mul
    binary_mul = dnnl_binary_mul,
    /// Binary max
    binary_max = dnnl_binary_max,
    /// Binary min
    binary_min = dnnl_binary_min,
    /// Binary div
    binary_div = dnnl_binary_div,
    /// Nearest Neighbor resampling method
    resampling_nearest = dnnl_resampling_nearest,
    /// Linear (Bilinear, Trilinear) resampling method
    resampling_linear = dnnl_resampling_linear,
    /// Reduction using max operation
    reduction_max = dnnl_reduction_max,
    /// Reduction using min operation
    reduction_min = dnnl_reduction_min,
    /// Reduction using sum operation
    reduction_sum = dnnl_reduction_sum,
    /// Reduction using mul operation
    reduction_mul = dnnl_reduction_mul,
    /// Reduction using mean operation
    reduction_mean = dnnl_reduction_mean,
    /// Reduction using norm_lp_max operation
    reduction_norm_lp_max = dnnl_reduction_norm_lp_max,
    /// Reduction using norm_lp_sum operation
    reduction_norm_lp_sum = dnnl_reduction_norm_lp_sum,
    /// Reduction using norm_lp_power_p_max operation
    reduction_norm_lp_power_p_max = dnnl_reduction_norm_lp_power_p_max,
    /// Reduction using norm_lp_power_p_sum operation
    reduction_norm_lp_power_p_sum = dnnl_reduction_norm_lp_power_p_sum,
};

/// Converts algorithm kind enum value from C++ API to C API type.
/// @param aalgorithm C++ API algorithm kind enum value.
/// @returns Corresponding C API algorithm kind enum value.
inline dnnl_alg_kind_t convert_to_c(algorithm aalgorithm) {
    return static_cast<dnnl_alg_kind_t>(aalgorithm);
}

/// @} dnnl_api_attributes

/// @addtogroup dnnl_api_primitives_common
/// @{

/// Flags for normalization primitives.
enum class normalization_flags : unsigned {
    /// Use no normalization flags. If specified, the library computes mean and
    /// variance on forward propagation for training and inference, outputs them
    /// on forward propagation for training, and computes the respective
    /// derivatives on backward propagation.
    none = dnnl_normalization_flags_none,

    /// Use global statistics. If specified, the library uses mean and
    /// variance provided by the user as an input on forward propagation and
    /// does not compute their derivatives on backward propagation. Otherwise,
    /// the library computes mean and variance on forward propagation for
    /// training and inference, outputs them on forward propagation for
    /// training, and computes the respective derivatives on backward
    /// propagation.
    use_global_stats = dnnl_use_global_stats,

    /// Use scale and shift parameters. If specified, the user is expected to
    /// pass scale and shift as inputs on forward propagation. On backward
    /// propagation of type #dnnl::prop_kind::backward, the library computes
    /// their derivatives. If not specified, the scale and shift parameters
    /// are not used by the library in any way.
    use_scale_shift = dnnl_use_scaleshift,

    /// Fuse normalization with ReLU. On training, normalization will require
    /// the workspace to implement backward propagation. On inference, the
    /// workspace is not required and behavior is the same as when normalization
    /// is fused with ReLU using the post-ops API.
    fuse_norm_relu = dnnl_fuse_norm_relu
};

/// Converts normalization flags enum value from C++ API to C API type.
/// @param flags C++ API normalization flags enum value.
/// @returns Corresponding C API normalization flags enum value.
inline dnnl_normalization_flags_t convert_to_c(normalization_flags flags) {
    return static_cast<dnnl_normalization_flags_t>(flags);
}

/// @} dnnl_api_primitives_common

/// @addtogroup dnnl_api_rnn
/// @{

/// RNN cell flags.
enum class rnn_flags : unsigned {
    /// Undefined RNN flags
    undef = dnnl_rnn_flags_undef
};

/// Converts RNN cell flags enum value from C++ API to C API type.
/// @param flags C++ API RNN cell flags enum value.
/// @returns Corresponding C API RNN cell flags enum value.
inline dnnl_rnn_flags_t convert_to_c(rnn_flags flags) {
    return static_cast<dnnl_rnn_flags_t>(flags);
}

#define DNNL_DEFINE_BITMASK_OPS(enum_name) \
    inline enum_name operator|(enum_name lhs, enum_name rhs) { \
        return static_cast<enum_name>( \
                static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs)); \
    } \
\
    inline enum_name operator&(enum_name lhs, enum_name rhs) { \
        return static_cast<enum_name>( \
                static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs)); \
    } \
\
    inline enum_name operator^(enum_name lhs, enum_name rhs) { \
        return static_cast<enum_name>( \
                static_cast<unsigned>(lhs) ^ static_cast<unsigned>(rhs)); \
    } \
\
    inline enum_name &operator|=(enum_name &lhs, enum_name rhs) { \
        lhs = static_cast<enum_name>( \
                static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs)); \
        return lhs; \
    } \
\
    inline enum_name &operator&=(enum_name &lhs, enum_name rhs) { \
        lhs = static_cast<enum_name>( \
                static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs)); \
        return lhs; \
    } \
\
    inline enum_name &operator^=(enum_name &lhs, enum_name rhs) { \
        lhs = static_cast<enum_name>( \
                static_cast<unsigned>(lhs) ^ static_cast<unsigned>(rhs)); \
        return lhs; \
    } \
\
    inline enum_name operator~(enum_name rhs) { \
        return static_cast<enum_name>(~static_cast<unsigned>(rhs)); \
    }

DNNL_DEFINE_BITMASK_OPS(normalization_flags)
DNNL_DEFINE_BITMASK_OPS(rnn_flags)

/// A direction of RNN primitive execution
enum class rnn_direction {
    /// Unidirectional execution of RNN primitive from left to right.
    unidirectional_left2right = dnnl_unidirectional_left2right,
    /// Unidirectional execution of RNN primitive from right to left.
    unidirectional_right2left = dnnl_unidirectional_right2left,
    /// Bidirectional execution of RNN primitive with concatenation of the
    /// results.
    bidirectional_concat = dnnl_bidirectional_concat,
    /// Bidirectional execution of RNN primitive with summation of the
    /// results.
    bidirectional_sum = dnnl_bidirectional_sum,
    /// Alias for #dnnl::rnn_direction::unidirectional_left2right
    unidirectional = dnnl_unidirectional,
};

/// Converts RNN direction enum value from C++ API to C API type.
/// @param dir C++ API RNN direction enum value.
/// @returns Corresponding C API RNN direction enum value.
inline dnnl_rnn_direction_t convert_to_c(rnn_direction dir) {
    return static_cast<dnnl_rnn_direction_t>(dir);
}

/// @} dnnl_api_rnn

/// @addtogroup dnnl_api_primitives_common
/// @{

/// Primitive descriptor query specification.
///
/// In general, queries are not used with the C++ API because most queries are
/// implemented as class members.
///
/// See @ref dnnl_query_t for more information.
enum class query {
    /// no query
    undef = dnnl_query_undef,

    /// execution engine
    engine = dnnl_query_engine,
    /// primitive kind
    primitive_kind = dnnl_query_primitive_kind,

    /// number of inputs expected
    num_of_inputs_s32 = dnnl_query_num_of_inputs_s32,
    /// number of outputs expected
    num_of_outputs_s32 = dnnl_query_num_of_outputs_s32,

    /// runtime estimation (seconds), unimplemented
    time_estimate_f64 = dnnl_query_time_estimate_f64,
    /// memory required for scratchpad (bytes)
    ///
    /// @sa @ref dev_guide_attributes_scratchpad
    memory_consumption_s64 = dnnl_query_memory_consumption_s64,

    /// scratchpad engine
    ///
    /// engine to be used for creating scratchpad memory
    scratchpad_engine = dnnl_query_scratchpad_engine,

    /// reorder source engine
    reorder_src_engine = dnnl_query_reorder_src_engine,
    /// reorder destination engine
    reorder_dst_engine = dnnl_query_reorder_dst_engine,

    /// implementation name
    impl_info_str = dnnl_query_impl_info_str,

    /// propagation kind
    prop_kind = dnnl_query_prop_kind,

    /// operation descriptor
    op_d = dnnl_query_op_d,
    /// convolution descriptor
    convolution_d = dnnl_query_convolution_d,
    /// deconvolution descriptor
    deconvolution_d = dnnl_query_deconvolution_d,
    /// shuffle descriptor
    shuffle_d = dnnl_query_shuffle_d,
    /// eltwise descriptor
    eltwise_d = dnnl_query_eltwise_d,
    /// softmax descriptor
    softmax_d = dnnl_query_softmax_d,
    /// pooling descriptor
    pooling_d = dnnl_query_pooling_d,
    /// lrn descriptor
    lrn_d = dnnl_query_lrn_d,
    /// batch normalization descriptor
    batch_normalization_d = dnnl_query_batch_normalization_d,
    /// layer normalization descriptor
    layer_normalization_d = dnnl_query_layer_normalization_d,
    /// inner product descriptor
    inner_product_d = dnnl_query_inner_product_d,
    /// rnn descriptor
    rnn_d = dnnl_query_rnn_d,
    /// binary descriptor
    binary_d = dnnl_query_binary_d,
    /// logsoftmax descriptor
    logsoftmax_d = dnnl_query_logsoftmax_d,
    /// matmul descriptor
    matmul_d = dnnl_query_matmul_d,
    /// resampling descriptor
    resampling_d = dnnl_query_resampling_d,
    /// reduction descriptor
    reduction_d = dnnl_query_reduction_d,

    /// source memory desc
    src_md = dnnl_query_src_md,
    /// source gradient (diff) memory desc
    diff_src_md = dnnl_query_diff_src_md,
    /// weights memory descriptor desc
    weights_md = dnnl_query_weights_md,
    /// weights gradient (diff) memory desc
    diff_weights_md = dnnl_query_diff_weights_md,
    /// destination memory desc
    dst_md = dnnl_query_dst_md,
    /// destination gradient (diff) memory desc
    diff_dst_md = dnnl_query_diff_dst_md,
    /// workspace memory desc
    workspace_md = dnnl_query_workspace_md,
    /// scratchpad memory desc
    scratchpad_md = dnnl_query_scratchpad_md,
    /// memory desc of an execute argument
    exec_arg_md = dnnl_query_exec_arg_md,
};

/// Converts query enum value from C++ API to C API type.
/// @param aquery C++ API query enum value.
/// @returns Corresponding C API query enum value.
inline dnnl_query_t convert_to_c(query aquery) {
    return static_cast<dnnl_query_t>(aquery);
}

/// @} dnnl_api_primitives_common

/// @} dnnl_api_primitives

/// @addtogroup dnnl_api_engine Engine
///
/// An abstraction of a computational device: a CPU, a specific GPU
/// card in the system, etc. Most primitives are created to execute
/// computations on one specific engine. The only exceptions are reorder
/// primitives that transfer data between two different engines.
///
/// @sa @ref dev_guide_basic_concepts
///
/// @{

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<dnnl_engine_t> {
    static dnnl_status_t destructor(dnnl_engine_t p) {
        return dnnl_engine_destroy(p);
    }
};
/// @endcond

/// An execution engine.
struct engine : public handle<dnnl_engine_t> {
    friend struct primitive;
    friend struct reorder;

    /// Kinds of engines.
    enum class kind {
        /// An unspecified engine
        any = dnnl_any_engine,
        /// CPU engine
        cpu = dnnl_cpu,
        /// GPU engine
        gpu = dnnl_gpu,
    };

    using handle::handle;

    /// Constructs an empty engine. An empty engine cannot be used in any
    /// operations.
    engine() = default;

    /// Returns the number of engines of a certain kind.
    ///
    /// @param akind The kind of engines to count.
    /// @returns The number of engines of the specified kind.
    static size_t get_count(kind akind) {
        return dnnl_engine_get_count(convert_to_c(akind));
    }

    /// Constructs an engine.
    ///
    /// @param akind The kind of engine to construct.
    /// @param index The index of the engine. Must be less than the value
    ///     returned by #get_count() for this particular kind of engine.
    engine(kind akind, size_t index) {
        dnnl_engine_t engine;
        error::wrap_c_api(
                dnnl_engine_create(&engine, convert_to_c(akind), index),
                "could not create an engine");
        reset(engine);
    }

    /// Constructs an engine based on a primitive from the primitive
    /// descriptor @p pd by querying its engine.
    ///
    /// @param pd The primitive descriptor to query.
    engine(const handle<dnnl_primitive_desc_t> &pd) {
        dnnl_engine_t c_engine;
        error::wrap_c_api(
                dnnl_primitive_desc_query(pd.get(),
                        dnnl::convert_to_c(dnnl::query::engine), 0, &c_engine),
                "could not get an engine from a primitive_desc");
        reset(c_engine, true);
    }

    /// Returns the kind of the engine.
    /// @returns The kind of the engine.
    kind get_kind() const {
        dnnl_engine_kind_t kind;
        error::wrap_c_api(dnnl_engine_get_kind(get(), &kind),
                "could not get kind of an engine");
        return static_cast<engine::kind>(kind);
    }

    /// Returns the engine of a primitive descriptor.
    ///
    /// @param pd The primitive descriptor to query.
    /// @returns A weak handle to the engine that the primitive descriptor was
    ///     created with.
    template <typename primitive_desc>
    static engine query(const primitive_desc &pd) {
        return query(pd, dnnl::query::engine);
    }

private:
    static dnnl_engine_kind_t convert_to_c(kind akind) {
        return static_cast<dnnl_engine_kind_t>(akind);
    }

    template <typename primitive_desc>
    static engine query(const primitive_desc &pd, dnnl::query what) {
        dnnl_engine_t c_engine;
        error::wrap_c_api(dnnl_primitive_desc_query(pd.get(),
                                  dnnl::convert_to_c(what), 0, &c_engine),
                "could not get an engine from a primitive_desc");
        return engine(c_engine, true);
    }
};

/// Converts engine kind enum value from C++ API to C API type.
///
/// @param akind C++ API engine kind enum value.
/// @returns Corresponding C API engine kind enum value.
inline dnnl_engine_kind_t convert_to_c(engine::kind akind) {
    return static_cast<dnnl_engine_kind_t>(akind);
}

/// @} dnnl_api_engine

/// @addtogroup dnnl_api_stream Stream
///
/// An encapsulation of execution context tied to a particular engine.
///
/// @sa @ref dev_guide_basic_concepts
///
/// @{

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<dnnl_stream_t> {
    static dnnl_status_t destructor(dnnl_stream_t p) {
        return dnnl_stream_destroy(p);
    }
};
/// @endcond

/// An execution stream.
struct stream : public handle<dnnl_stream_t> {
    using handle::handle;

    /// Stream flags. Can be combined using the bitwise OR operator.
    enum class flags : unsigned {
        /// In-order execution.
        in_order = dnnl_stream_in_order,
        /// Out-of-order execution.
        out_of_order = dnnl_stream_out_of_order,
        /// Default stream configuration.
        default_flags = dnnl_stream_default_flags,
    };

    /// Constructs an empty stream. An empty stream cannot be used in any
    /// operations.
    stream() = default;

    /// Constructs a stream for the specified engine and with behavior
    /// controlled by the specified flags.
    ///
    /// @param aengine Engine to create the stream on.
    /// @param aflags Flags controlling stream behavior.
    stream(const engine &aengine, flags aflags = flags::default_flags) {
        dnnl_stream_t stream;
        error::wrap_c_api(dnnl_stream_create(&stream, aengine.get(),
                                  static_cast<dnnl_stream_flags_t>(aflags)),
                "could not create a stream");
        reset(stream);
    }

    /// Returns the associated engine.
    engine get_engine() const {
        dnnl_engine_t c_engine;
        error::wrap_c_api(dnnl_stream_get_engine(get(), &c_engine),
                "could not get an engine from a stream object");
        return engine(c_engine, true);
    }

    /// Waits for all primitives executing in the stream to finish.
    /// @returns The stream itself.
    stream &wait() {
        error::wrap_c_api(
                dnnl_stream_wait(get()), "could not wait on a stream");
        return *this;
    }
};

DNNL_DEFINE_BITMASK_OPS(stream::flags)

/// @} dnnl_api_stream

/// @addtogroup dnnl_api_memory Memory
///
/// A container that describes and stores data. Memory objects can contain
/// data of various types and formats. There are two levels of abstraction:
///
/// 1. **Memory descriptor** -- engine-agnostic logical description of data
///     (number of dimensions, dimension sizes, and data type), and,
///     optionally, the information about the physical format of data in
///     memory. If this information is not known yet, a memory descriptor can
///     be created with #dnnl::memory::format_tag::any. This allows
///     compute-intensive primitives to choose the best format for
///     computation. The user is responsible for reordering the data into the
///     chosen format when formats do not match.
///
///     A memory descriptor can be initialized either by specifying dimensions
///     and a memory format tag or strides for each of them, or by
///     manipulating the dnnl_memory_desc_t structure directly.
///
///     @warning
///         The latter approach requires understanding how the physical data
///         representation is mapped to the structure and is discouraged. This
///         topic is discussed in @ref dev_guide_understanding_memory_formats.
///
///     The user can query the amount of memory required by a memory
///     descriptor using the #dnnl::memory::desc::get_size() function. The
///     size of data in general cannot be computed as the product of
///     dimensions multiplied by the size of the data type. So users are
///     required to use this function for better code portability.
///
///     Two memory descriptors can be compared using the equality and
///     inequality operators.  The comparison is especially useful when
///     checking whether it is necessary to reorder data from the user's data
///     format to a primitive's format.
///
/// 2. **Memory object** -- an engine-specific object that handles the memory
///     buffer and its description (a memory descriptor). For the CPU engine or
///     with USM, the memory buffer handle is simply a pointer to @c void. The
///     memory buffer can be queried using #dnnl::memory::get_data_handle() and
///     set using #dnnl::memory::set_data_handle(). The underlying SYCL buffer,
///     when used, can be queried using #dnnl::sycl_interop::get_buffer and set
///     using #dnnl::sycl_interop::set_buffer. A memory object can also be
///     queried for the underlying memory descriptor and for its engine using
///     #dnnl::memory::get_desc() and dnnl::memory::get_engine().
///
/// Along with ordinary memory descriptors with all dimensions being positive,
/// the library supports *zero-volume*  memory descriptors with one or more
/// dimensions set to zero. This is used to support the NumPy\* convention.
/// If a zero-volume memory is passed to a primitive, the primitive typically
/// does not perform any computations with this memory. For example:
///
/// - A concatenation primitive would ignore all memory object with zeroes in
///   the concat dimension / axis.
///
/// - A forward convolution with a source memory object with zero in the
///   minibatch dimension would always produce a destination memory object
///   with a zero in the minibatch dimension and perform no computations.
///
/// - However, a forward convolution with a zero in one of the weights
///   dimensions is ill-defined and is considered to be an error by the
///   library because there is no clear definition of what the output values
///   should be.
///
/// Memory buffer of a zero-volume memory is never accessed.
///
/// @{

/// Memory object.
///
/// A memory object encapsulates a handle to a memory buffer allocated on a
/// specific engine, tensor dimensions, data type, and memory format, which is
/// the way tensor indices map to offsets in linear memory space. Memory
/// objects are passed to primitives during execution.
struct memory : public handle<dnnl_memory_t> {
    using handle::handle;

    /// Integer type for representing dimension sizes and indices.
    typedef dnnl_dim_t dim;
    /// Vector of dimensions. Implementations are free to force a limit on the
    /// vector's length.
    typedef std::vector<dim> dims;

    /// Helper function that validates that an `std::vector` of dimensions can
    /// be safely converted to the C API array ::dnnl_dims_t. Throws if
    /// validation fails.
    ///
    /// @param v Vector of dimensions.
    /// @param min_size Minimum expected size of the vector.
    template <typename T>
    static void validate_dims(const std::vector<T> &v, int min_size = 0) {
        validate_container_size(
                v, "dimensions are invalid", min_size, DNNL_MAX_NDIMS);
    }

    /// Data type specification.
    enum class data_type {
        /// Undefined data type (used for empty memory descriptors).
        undef = dnnl_data_type_undef,
        /// [16-bit/half-precision floating point](https://en.wikipedia.org/wiki/Half-precision_floating-point_format).
        f16 = dnnl_f16,
        /// non-standard
        /// [16-bit floating point with 7-bit mantissa](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format).
        bf16 = dnnl_bf16,
        /// [32-bit/single-precision floating point](https://en.wikipedia.org/wiki/Single-precision_floating-point_format).
        f32 = dnnl_f32,
        /// 32-bit signed integer.
        s32 = dnnl_s32,
        /// 8-bit signed integer.
        s8 = dnnl_s8,
        /// 8-bit unsigned integer.
        u8 = dnnl_u8,
    };

    /// Memory format kind
    enum class format_kind {
        /// Undefined memory format kind, used for empty memory descriptors.
        undef = dnnl_format_kind_undef,
        /// Unspecified format kind.
        /// The primitive selects a format automatically.
        any = dnnl_format_kind_any,
        /// A tensor in a generic format described by the stride and blocking
        /// values in each dimension. See @ref dnnl_blocking_desc_t for more
        /// information.
        blocked = dnnl_blocked,
        /// Weights format used in 8-bit Winograd convolution.
        wino = dnnl_format_kind_wino,
        /// Packed weights format used in RNN.
        packed = dnnl_format_kind_rnn_packed,
    };

    /// Memory format tag specification.
    ///
    /// Memory format tags can be further divided into two categories:
    ///
    ///  - Domain-agnostic names, i.e. names that do not depend on the tensor
    ///    usage in the specific primitive. These names use letters from `a`
    ///    to `f` to denote logical dimensions and form the order in which the
    ///    dimensions are laid in memory. For example,
    ///    #dnnl::memory::format_tag::ab is used to denote a 2D tensor where the
    ///    second logical dimension (denoted as `b`) is the innermost, i.e.
    ///    has stride = 1, and the first logical dimension (`a`) is laid out in
    ///    memory with stride equal to the size of the second dimension. On the
    ///    other hand, #dnnl::memory::format_tag::ba is the transposed version
    ///    of the same tensor: the outermost dimension (`a`) becomes the
    ///    innermost one.
    ///
    ///  - Domain-specific names, i.e. names that make sense only in the
    ///    context of a certain domain, such as CNN. These names are
    ///    aliases to the corresponding domain-agnostic tags and used mostly
    ///    for convenience. For example, #dnnl::memory::format_tag::nc
    ///    is used to denote 2D CNN activations tensor memory format, where
    ///    the channels dimension is the innermost one and the batch dimension
    ///    is the outermost one. Moreover, #dnnl::memory::format_tag::nc is
    ///    an alias for #dnnl::memory::format_tag::ab, because for
    ///    CNN primitives the logical dimensions of activations tensors come
    ///    in order: batch, channels, spatial.  In other words, batch
    ///    corresponds to the first logical dimension (`a`), and channels
    ///    correspond to the second one (`b`).
    ///
    /// The following domain-specific notation applies to memory format tags:
    ///  - @c 'n' denotes the mini-batch dimension
    ///  - @c 'c' denotes a channels dimension
    ///  - When there are multiple channel dimensions (for example,
    ///    in convolution weights tensor), @c 'i' and @c 'o' denote dimensions
    ///    of input and output channels
    ///  - @c 'g' denotes a groups dimension for convolution weights
    ///  - @c 'd', @c 'h', and @c 'w' denote spatial depth, height, and width
    ///    respectively
    ///
    /// See @ref dnnl_format_tag_t for a detailed description.
    enum class format_tag {
        /// Undefined memory format tag
        undef = dnnl_format_tag_undef,
        /// Placeholder memory format tag. Used to instruct the primitive to
        /// select a format automatically.
        any = dnnl_format_tag_any,

        /// plain 1D tensor
        a = dnnl_a,

        /// plain 2D tensor
        ab = dnnl_ab,
        /// permuted 2D tensor
        ba = dnnl_ba,

        /// plain 3D tensor
        abc = dnnl_abc,
        /// permuted 3D tensor
        acb = dnnl_acb,
        /// permuted 3D tensor
        bac = dnnl_bac,
        /// permuted 3D tensor
        bca = dnnl_bca,
        /// permuted 3D tensor
        cba = dnnl_cba,

        /// plain 4D tensor
        abcd = dnnl_abcd,
        /// permuted 4D tensor
        abdc = dnnl_abdc,
        /// permuted 4D tensor
        acdb = dnnl_acdb,
        /// permuted 4D tensor
        bacd = dnnl_bacd,
        /// permuted 4D tensor
        bcda = dnnl_bcda,
        /// permuted 4D tensor
        cdba = dnnl_cdba,
        /// permuted 4D tensor
        dcab = dnnl_dcab,

        /// plain 5D tensor
        abcde = dnnl_abcde,
        /// permuted 5D tensor
        abdec = dnnl_abdec,
        /// permuted 5D tensor
        acbde = dnnl_acbde,
        /// permuted 5D tensor
        acdeb = dnnl_acdeb,
        /// permuted 5D tensor
        bacde = dnnl_bacde,
        /// permuted 5D tensor
        bcdea = dnnl_bcdea,
        /// permuted 5D tensor
        cdeba = dnnl_cdeba,
        /// permuted 5D tensor
        decab = dnnl_decab,
        /// permuted 5D tensor
        abced = dnnl_abced,

        /// plain 6D tensor
        abcdef = dnnl_abcdef,
        /// plain 6D tensor
        acbdef = dnnl_acbdef,
        /// plain 6D tensor
        defcab = dnnl_defcab,
        /// permuted 6D tensor
        abcdfe = dnnl_abcdfe,

        /// plain 7D tensor
        abcdefg = dnnl_abcdefg,
        /// permuted 7D tensor
        abcdegf = dnnl_abcdegf,

        /// plain 8D tensor
        abcdefgh = dnnl_abcdefgh,
        /// permuted 8D tensor
        abcdefhg = dnnl_abcdefhg,

        /// plain 9D tensor
        abcdefghi = dnnl_abcdefghi,
        /// permuted 9D tensor
        abcdefgih = dnnl_abcdefgih,

        /// plain 10D tensor
        abcdefghij = dnnl_abcdefghij,
        /// permuted 10D tensor
        abcdefghji = dnnl_abcdefghji,

        /// plain 11D tensor
        abcdefghijk = dnnl_abcdefghijk,
        /// permuted 11D tensor
        abcdefghikj = dnnl_abcdefghikj,

        /// plain 12D tensor
        abcdefghijkl = dnnl_abcdefghijkl,
        /// permuted 12D tensor
        abcdefghijlk = dnnl_abcdefghijlk,

        /// 1D tensor; an alias for #dnnl::memory::format_tag::a
        x = a,
        /// 2D CNN activations tensor; an alias for #dnnl::memory::format_tag::ab
        nc = ab,
        /// 2D CNN activations tensor; an alias for #dnnl::memory::format_tag::ba
        cn = ba,
        /// 2D RNN statistics tensor; an alias for #dnnl::memory::format_tag::ab
        tn = ab,
        /// 2D RNN statistics tensor; an alias for #dnnl::memory::format_tag::ba
        nt = ba,
        /// 3D CNN activations tensor; an alias for #dnnl::memory::format_tag::abc
        ncw = abc,
        /// 3D CNN activations tensor; an alias for #dnnl::memory::format_tag::acb
        nwc = acb,
        /// 4D CNN activations tensor; an alias for #dnnl::memory::format_tag::abcd
        nchw = abcd,
        /// 4D CNN activations tensor; an alias for #dnnl::memory::format_tag::acdb
        nhwc = acdb,
        /// 4D CNN activations tensor; an alias for #dnnl::memory::format_tag::bcda
        chwn = bcda,
        /// 5D CNN activations tensor; an alias for #dnnl::memory::format_tag::abcde
        ncdhw = abcde,
        /// 5D CNN activations tensor; an alias for #dnnl::memory::format_tag::acdeb
        ndhwc = acdeb,

        /// 2D CNN weights tensor; an alias for #dnnl::memory::format_tag::ab
        oi = ab,
        /// 2D CNN weights tensor; an alias for #dnnl::memory::format_tag::ba
        io = ba,
        /// 3D CNN weights tensor; an alias for #dnnl::memory::format_tag::abc
        oiw = abc,
        /// 3D CNN weights tensor; an alias for #dnnl::memory::format_tag::acb
        owi = acb,
        /// 3D CNN weights tensor; an alias for #dnnl::memory::format_tag::cba
        wio = cba,
        /// 3D CNN weights tensor; an alias for #dnnl::memory::format_tag::bca
        iwo = bca,
        /// 4D CNN weights tensor; an alias for #dnnl::memory::format_tag::abcd
        oihw = abcd,
        /// 4D CNN weights tensor; an alias for #dnnl::memory::format_tag::cdba
        hwio = cdba,
        /// 4D CNN weights tensor; an alias for #dnnl::memory::format_tag::acdb
        ohwi = acdb,
        /// 4D CNN weights tensor; an alias for #dnnl::memory::format_tag::bcda
        ihwo = bcda,
        /// 4D CNN weights tensor; an alias for #dnnl::memory::format_tag::bacd
        iohw = bacd,
        /// 5D CNN weights tensor; an alias for #dnnl::memory::format_tag::abcde
        oidhw = abcde,
        /// 5D CNN weights tensor; an alias for #dnnl::memory::format_tag::cdeba
        dhwio = cdeba,
        /// 5D CNN weights tensor; an alias for #dnnl::memory::format_tag::acdeb
        odhwi = acdeb,
        /// 5D CNN weights tensor; an alias for #dnnl::memory::format_tag::bacde
        iodhw = bacde,
        /// 5D CNN weights tensor; an alias for #dnnl::memory::format_tag::bcdea
        idhwo = bcdea,

        /// 4D CNN weights tensor with groups; an alias for #dnnl::memory::format_tag::abcd
        goiw = abcd,
        /// 4D CNN weights tensor with groups; an alias for #dnnl::memory::format_tag::dcab
        wigo = dcab,
        /// 5D CNN weights tensor with groups; an alias for #dnnl::memory::format_tag::abcde
        goihw = abcde,
        /// 5D CNN weights tensor with groups; an alias for #dnnl::memory::format_tag::decab
        hwigo = decab,
        /// 5D CNN weights tensor with groups; an alias for #dnnl::memory::format_tag::acbde
        giohw = acbde,
        /// 6D CNN weights tensor with groups; an alias for #dnnl::memory::format_tag::abcdef
        goidhw = abcdef,
        /// 6D CNN weights tensor with groups; an alias for #dnnl::memory::format_tag::abcdef
        giodhw = acbdef,
        /// 6D CNN weights tensor with groups; an alias for #dnnl::memory::format_tag::defcab
        dhwigo = defcab,

        /// 3D RNN data tensor in the format (seq_length, batch, input channels).
        tnc = abc,
        /// 3D RNN data tensor in the format (batch, seq_length, input channels).
        ntc = bac,
        /// 4D RNN states tensor in the format (num_layers, num_directions,
        /// batch, state channels).
        ldnc = abcd,
        /// 5D RNN weights tensor in the format (num_layers, num_directions,
        ///  input_channels, num_gates, output_channels).
        ///
        ///  - For LSTM cells, the gates order is input, forget, candidate
        ///    and output gate.
        ///  - For GRU cells, the gates order is update, reset and output gate.
        ldigo = abcde,
        /// 5D RNN weights tensor in the format (num_layers, num_directions,
        /// num_gates, output_channels, input_channels).
        ///
        ///  - For LSTM cells, the gates order is input, forget, candidate
        ///    and output gate.
        ///  - For GRU cells, the gates order is update, reset and output gate.
        ldgoi = abdec,
        /// 4D LSTM projection tensor in the format (num_layers, num_directions,
        /// num_channels_in_hidden_state, num_channels_in_recurrent_projection).
        ldio = abcd,
        /// 4D LSTM projection tensor in the format (num_layers, num_directions,
        /// num_channels_in_recurrent_projection, num_channels_in_hidden_state).
        ldoi = abdc,
        /// 4D RNN bias tensor in the format (num_layers, num_directions,
        /// num_gates, output_channels).
        ///
        ///  - For LSTM cells, the gates order is input, forget, candidate
        ///    and output gate.
        ///  - For GRU cells, the gates order is update, reset and output gate.
        ldgo = abcd,

        // Opaque blocked formats

        Abc16a = dnnl_Abc16a,
        ABc16a16b = dnnl_ABc16a16b,
        ABc4a4b = dnnl_ABc4a4b,
        aBc16b = dnnl_aBc16b,
        aBc32b = dnnl_aBc32b,
        ABc16b16a = dnnl_ABc16b16a,
        Abc4a = dnnl_Abc4a,
        aBc4b = dnnl_aBc4b,
        ABc4b16a4b = dnnl_ABc4b16a4b,
        ABc2b8a4b = dnnl_ABc2b8a4b,
        ABc16b16a4b = dnnl_ABc16b16a4b,
        ABc16b16a2b = dnnl_ABc16b16a2b,
        ABc4b4a = dnnl_ABc4b4a,
        ABc8a16b2a = dnnl_ABc8a16b2a,
        ABc8a8b = dnnl_ABc8a8b,
        ABc8a4b = dnnl_ABc8a4b,
        aBc8b = dnnl_aBc8b,
        ABc8b16a2b = dnnl_ABc8b16a2b,
        ABc8b8a = dnnl_ABc8b8a,
        Abcd8a = dnnl_Abcd8a,
        Abcd16a = dnnl_Abcd16a,
        Abcd32a = dnnl_Abcd32a,
        ABcd16a16b = dnnl_ABcd16a16b,
        aBcd16b = dnnl_aBcd16b,
        aBcd32b = dnnl_aBcd32b,
        ABcd16b16a = dnnl_ABcd16b16a,
        aBCd16b16c = dnnl_aBCd16b16c,
        aBCd16c16b = dnnl_aBCd16c16b,
        Abcd4a = dnnl_Abcd4a,
        aBcd4b = dnnl_aBcd4b,
        ABcd4b16a4b = dnnl_ABcd4b16a4b,
        ABcd2b8a4b = dnnl_ABcd2b8a4b,
        ABcd4b4a = dnnl_ABcd4b4a,
        ABcd4a4b = dnnl_ABcd4a4b,
        aBCd4c16b4c = dnnl_aBCd4c16b4c,
        aBCd2c8b4c = dnnl_aBCd2c8b4c,
        ABcd16b16a4b = dnnl_ABcd16b16a4b,
        ABcd16b16a2b = dnnl_ABcd16b16a2b,
        aBCd16c16b4c = dnnl_aBCd16c16b4c,
        aBCd16c16b2c = dnnl_aBCd16c16b2c,
        aBCd4c4b = dnnl_aBCd4c4b,
        aBCd4b4c = dnnl_aBCd4b4c,
        ABcd8a16b2a = dnnl_ABcd8a16b2a,
        ABcd8a8b = dnnl_ABcd8a8b,
        ABcd8a4b = dnnl_ABcd8a4b,
        /// 4D tensor blocked by 2nd dimension with block size 8
        aBcd8b = dnnl_aBcd8b,
        ABcd8b16a2b = dnnl_ABcd8b16a2b,
        aBCd8b16c2b = dnnl_aBCd8b16c2b,
        /// 4D tensor blocked by 1st and 2nd dimension with block size 8
        ABcd8b8a = dnnl_ABcd8b8a,
        aBCd8b8c = dnnl_aBCd8b8c,
        aBCd8b4c = dnnl_aBCd8b4c,
        aBCd8c16b2c = dnnl_aBCd8c16b2c,
        aBCd8c8b = dnnl_aBCd8c8b,
        Abcde16a = dnnl_Abcde16a,
        Abcde32a = dnnl_Abcde32a,
        ABcde16a16b = dnnl_ABcde16a16b,
        aBcde16b = dnnl_aBcde16b,
        aBcde32b = dnnl_aBcde32b,
        ABcde16b16a = dnnl_ABcde16b16a,
        aBCde16b16c = dnnl_aBCde16b16c,
        aBCde16c16b = dnnl_aBCde16c16b,
        aBCde2c8b4c = dnnl_aBCde2c8b4c,
        Abcde4a = dnnl_Abcde4a,
        aBcde4b = dnnl_aBcde4b,
        ABcde4b4a = dnnl_ABcde4b4a,
        ABcde4a4b = dnnl_ABcde4a4b,
        aBCde4b4c = dnnl_aBCde4b4c,
        aBCde4c16b4c = dnnl_aBCde4c16b4c,
        aBCde16c16b4c = dnnl_aBCde16c16b4c,
        aBCde16c16b2c = dnnl_aBCde16c16b2c,
        aBCde4c4b = dnnl_aBCde4c4b,
        Abcde8a = dnnl_Abcde8a,
        ABcde8a8b = dnnl_ABcde8a8b,
        ABcde8a4b = dnnl_ABcde8a4b,
        aBcde8b = dnnl_aBcde8b,
        ABcde8b16a2b = dnnl_ABcde8b16a2b,
        ABcde4b16a4b = dnnl_ABcde4b16a4b,
        ABcde2b8a4b = dnnl_ABcde2b8a4b,
        aBCde8b16c2b = dnnl_aBCde8b16c2b,
        ABcde8b8a = dnnl_ABcde8b8a,
        aBCde8b8c = dnnl_aBCde8b8c,
        aBCde8b4c = dnnl_aBCde8b4c,
        ABcd4a8b8a4b = dnnl_ABcd4a8b8a4b,
        ABcd2a8b8a2b = dnnl_ABcd2a8b8a2b,
        aBCde4b8c8b4c = dnnl_aBCde4b8c8b4c,
        aBCde2b8c8b2c = dnnl_aBCde2b8c8b2c,
        aBCde8c16b2c = dnnl_aBCde8c16b2c,
        aBCde8c8b = dnnl_aBCde8c8b,
        aBcdef16b = dnnl_aBcdef16b,
        aBCdef16b16c = dnnl_aBCdef16b16c,
        aBCdef16c16b = dnnl_aBCdef16c16b,
        aBcdef4b = dnnl_aBcdef4b,
        aBCdef2c8b4c = dnnl_aBCdef2c8b4c,
        aBCdef4c4b = dnnl_aBCdef4c4b,
        aBCdef4b4c = dnnl_aBCdef4b4c,
        aBCdef8b8c = dnnl_aBCdef8b8c,
        aBCdef8b4c = dnnl_aBCdef8b4c,
        aBCdef8c16b2c = dnnl_aBCdef8c16b2c,
        aBCdef4c16b4c = dnnl_aBCdef4c16b4c,
        aBCdef8c8b = dnnl_aBCdef8c8b,
        aBdc16b = dnnl_aBdc16b,
        aBdc4b = dnnl_aBdc4b,
        aBdc8b = dnnl_aBdc8b,
        aBdec16b = dnnl_aBdec16b,
        aBdec4b = dnnl_aBdec4b,
        aBdec8b = dnnl_aBdec8b,
        aBdefc16b = dnnl_aBdefc16b,
        aCBdef16c16b = dnnl_aCBdef16c16b,
        aCBdef16b16c = dnnl_aCBdef16b16c,
        aBdefc4b = dnnl_aBdefc4b,
        aBdefc8b = dnnl_aBdefc8b,
        Acb16a = dnnl_Acb16a,
        Acb4a = dnnl_Acb4a,
        Acb8a = dnnl_Acb8a,
        aCBd16b16c = dnnl_aCBd16b16c,
        aCBd16c16b = dnnl_aCBd16c16b,
        aCBde16b16c = dnnl_aCBde16b16c,
        aCBde16c16b = dnnl_aCBde16c16b,
        Acdb16a = dnnl_Acdb16a,
        Acdb4a = dnnl_Acdb4a,
        Acdb8a = dnnl_Acdb8a,
        Acdeb16a = dnnl_Acdeb16a,
        Acdeb4a = dnnl_Acdeb4a,
        Acdeb8a = dnnl_Acdeb8a,
        BAc16a16b = dnnl_BAc16a16b,
        BAc16b16a = dnnl_BAc16b16a,
        BAcd16a16b = dnnl_BAcd16a16b,
        BAcd16b16a = dnnl_BAcd16b16a,
        ABcd32a32b = dnnl_ABcd32a32b,
        BAcde16b16a = dnnl_BAcde16b16a,
        BAcde16a16b = dnnl_BAcde16a16b,
        aBdec32b = dnnl_aBdec32b,
        Abcdef16a = dnnl_Abcdef16a,
        Abcdef32a = dnnl_Abcdef32a,
        Acdb32a = dnnl_Acdb32a,
        aBCd2b4c2b = dnnl_aBCd2b4c2b,
        aBCde2b4c2b = dnnl_aBCde2b4c2b,
        aBCdef2b4c2b = dnnl_aBCdef2b4c2b,
        aBCd2c4b2c = dnnl_aBCd2c4b2c,
        aBCde2c4b2c = dnnl_aBCde2c4b2c,
        aBCdef2c4b2c = dnnl_aBCdef2c4b2c,
        aBCd4b8c2b = dnnl_aBCd4b8c2b,
        aBCde4b8c2b = dnnl_aBCde4b8c2b,
        aBCdef4b8c2b = dnnl_aBCdef4b8c2b,
        aBCd4c8b2c = dnnl_aBCd4c8b2c,
        aBCde4c8b2c = dnnl_aBCde4c8b2c,
        aBCdef4c8b2c = dnnl_aBCdef4c8b2c,

        format_tag_last = dnnl_format_tag_last,

        nCdhw16c = dnnl_nCdhw16c,
        nCdhw4c = dnnl_nCdhw4c,
        nCdhw8c = dnnl_nCdhw8c,
        nChw16c = dnnl_nChw16c,
        nChw4c = dnnl_nChw4c,
        nChw8c = dnnl_nChw8c,
        nCw16c = dnnl_nCw16c,
        nCw4c = dnnl_nCw4c,
        nCw8c = dnnl_nCw8c,
        NCw16n16c = dnnl_NCw16n16c,
        NChw16n16c = dnnl_NChw16n16c,
        NCdhw16n16c = dnnl_NCdhw16n16c,
        NCdhw32n32c = dnnl_NCdhw32n32c,
        NChw32n32c = dnnl_NChw32n32c,
        IOhw16i16o = dnnl_IOhw16i16o,
        Ohwi32o = dnnl_Ohwi32o,
        IOdhw16i16o = dnnl_IOdhw16i16o,
        gIOhw16i16o = dnnl_gIOhw16i16o,
        gOhwi32o = dnnl_gOhwi32o,
        Goidhw16g = dnnl_Goidhw16g,
        IOw16o16i = dnnl_IOw16o16i,
        OIw16i16o = dnnl_OIw16i16o,
        IOw16i16o = dnnl_IOw16i16o,
        gIOw16i16o = dnnl_gIOw16i16o,
        OIw16o16i = dnnl_OIw16o16i,
        Oiw16o = dnnl_Oiw16o,
        OIw4i16o4i = dnnl_OIw4i16o4i,
        OIw2i8o4i = dnnl_OIw2i8o4i,
        OIw4i4o = dnnl_OIw4i4o,
        OIw4o4i = dnnl_OIw4o4i,
        Oiw4o = dnnl_Oiw4o,
        OIw8i16o2i = dnnl_OIw8i16o2i,
        OIw8i8o = dnnl_OIw8i8o,
        OIw8o16i2o = dnnl_OIw8o16i2o,
        OIw8o8i = dnnl_OIw8o8i,
        OIw8o4i = dnnl_OIw8o4i,
        Owi16o = dnnl_Owi16o,
        OwI16o2i = dnnl_OwI16o2i,
        Owi4o = dnnl_Owi4o,
        Owi8o = dnnl_Owi8o,
        IOhw16o16i = dnnl_IOhw16o16i,
        Ohwi16o = dnnl_Ohwi16o,
        OhwI16o2i = dnnl_OhwI16o2i,
        Ohwi4o = dnnl_Ohwi4o,
        Ohwi8o = dnnl_Ohwi8o,
        OIhw16i16o = dnnl_OIhw16i16o,
        OIhw16o16i = dnnl_OIhw16o16i,
        Oihw16o = dnnl_Oihw16o,
        OIhw4i16o4i = dnnl_OIhw4i16o4i,
        OIhw4i4o = dnnl_OIhw4i4o,
        OIhw4o4i = dnnl_OIhw4o4i,
        Oihw4o = dnnl_Oihw4o,
        OIhw8i16o2i = dnnl_OIhw8i16o2i,
        OIhw8i8o = dnnl_OIhw8i8o,
        OIhw8o16i2o = dnnl_OIhw8o16i2o,
        OIhw8o8i = dnnl_OIhw8o8i,
        OIhw8o4i = dnnl_OIhw8o4i,
        OIhw2i8o4i = dnnl_OIhw2i8o4i,
        IOdhw16o16i = dnnl_IOdhw16o16i,
        Odhwi16o = dnnl_Odhwi16o,
        OdhwI16o2i = dnnl_OdhwI16o2i,
        Odhwi4o = dnnl_Odhwi4o,
        Odhwi8o = dnnl_Odhwi8o,
        OIdhw16i16o = dnnl_OIdhw16i16o,
        OIdhw16o16i = dnnl_OIdhw16o16i,
        Oidhw16o = dnnl_Oidhw16o,
        OIdhw4i4o = dnnl_OIdhw4i4o,
        OIdhw4o4i = dnnl_OIdhw4o4i,
        Oidhw4o = dnnl_Oidhw4o,
        OIdhw8i16o2i = dnnl_OIdhw8i16o2i,
        OIdhw4i16o4i = dnnl_OIdhw4i16o4i,
        OIdhw2i8o4i = dnnl_OIdhw2i8o4i,
        OIdhw8i8o = dnnl_OIdhw8i8o,
        OIdhw8o8i = dnnl_OIdhw8o8i,
        OIdhw8o4i = dnnl_OIdhw8o4i,
        gIOw16o16i = dnnl_gIOw16o16i,
        gOIw16i16o = dnnl_gOIw16i16o,
        gOIw16o16i = dnnl_gOIw16o16i,
        gOiw16o = dnnl_gOiw16o,
        gOIw4i16o4i = dnnl_gOIw4i16o4i,
        gOIw2i8o4i = dnnl_gOIw2i8o4i,
        gOIw4i4o = dnnl_gOIw4i4o,
        gOIw4o4i = dnnl_gOIw4o4i,
        gOiw4o = dnnl_gOiw4o,
        gOIw8i16o2i = dnnl_gOIw8i16o2i,
        gOIw8i8o = dnnl_gOIw8i8o,
        gOIw8o16i2o = dnnl_gOIw8o16i2o,
        gOIw8o8i = dnnl_gOIw8o8i,
        gOIw8o4i = dnnl_gOIw8o4i,
        gOwi16o = dnnl_gOwi16o,
        gOwI16o2i = dnnl_gOwI16o2i,
        gOwi4o = dnnl_gOwi4o,
        gOwi8o = dnnl_gOwi8o,
        Goiw8g = dnnl_Goiw8g,
        Goiw16g = dnnl_Goiw16g,
        gIOhw16o16i = dnnl_gIOhw16o16i,
        gOhwi16o = dnnl_gOhwi16o,
        gOhwI16o2i = dnnl_gOhwI16o2i,
        gOhwi4o = dnnl_gOhwi4o,
        gOhwi8o = dnnl_gOhwi8o,
        Goihw16g = dnnl_Goihw16g,
        gOIhw16i16o = dnnl_gOIhw16i16o,
        gOIhw16o16i = dnnl_gOIhw16o16i,
        gOihw16o = dnnl_gOihw16o,
        gOIhw4i16o4i = dnnl_gOIhw4i16o4i,
        gOIhw2i8o4i = dnnl_gOIhw2i8o4i,
        gOIhw4i4o = dnnl_gOIhw4i4o,
        gOIhw4o4i = dnnl_gOIhw4o4i,
        gOihw4o = dnnl_gOihw4o,
        Goihw8g = dnnl_Goihw8g,
        gOIhw8i16o2i = dnnl_gOIhw8i16o2i,
        gOIhw8i8o = dnnl_gOIhw8i8o,
        gOIhw8o16i2o = dnnl_gOIhw8o16i2o,
        OIw4o8i8o4i = dnnl_OIw4o8i8o4i,
        OIdhw4o8i8o4i = dnnl_OIdhw4o8i8o4i,
        OIhw4o8i8o4i = dnnl_OIhw4o8i8o4i,
        OIhw2o8i8o2i = dnnl_OIhw2o8i8o2i,
        gOIw4o8i8o4i = dnnl_gOIw4o8i8o4i,
        gOIdhw4o8i8o4i = dnnl_gOIdhw4o8i8o4i,
        gOIhw4o8i8o4i = dnnl_gOIhw4o8i8o4i,
        gOIhw2o8i8o2i = dnnl_gOIhw2o8i8o2i,
        OIhw16i16o4i = dnnl_OIhw16i16o4i,
        OIhw16i16o2i = dnnl_OIhw16i16o2i,
        gOIhw16i16o4i = dnnl_gOIhw16i16o4i,
        gOIhw16i16o2i = dnnl_gOIhw16i16o2i,
        gOIhw8o8i = dnnl_gOIhw8o8i,
        gOIhw8o4i = dnnl_gOIhw8o4i,
        gIOdhw16i16o = dnnl_gIOdhw16i16o,
        gIOdhw16o16i = dnnl_gIOdhw16o16i,
        gOdhwi16o = dnnl_gOdhwi16o,
        gOdhwI16o2i = dnnl_gOdhwI16o2i,
        gOdhwi4o = dnnl_gOdhwi4o,
        gOdhwi8o = dnnl_gOdhwi8o,
        gOIdhw16i16o = dnnl_gOIdhw16i16o,
        gOIdhw16o16i = dnnl_gOIdhw16o16i,
        gOidhw16o = dnnl_gOidhw16o,
        gOIdhw4i4o = dnnl_gOIdhw4i4o,
        gOIdhw4o4i = dnnl_gOIdhw4o4i,
        gOidhw4o = dnnl_gOidhw4o,
        gOIdhw8i16o2i = dnnl_gOIdhw8i16o2i,
        gOIdhw4i16o4i = dnnl_gOIdhw4i16o4i,
        gOIdhw2i8o4i = dnnl_gOIdhw2i8o4i,
        gOIdhw8i8o = dnnl_gOIdhw8i8o,
        gOIdhw8o8i = dnnl_gOIdhw8o8i,
        gOIdhw8o4i = dnnl_gOIdhw8o4i,
        gOIw2i4o2i = dnnl_gOIw2i4o2i,
        gOIhw2i4o2i = dnnl_gOIhw2i4o2i,
        gOIdhw2i4o2i = dnnl_gOIdhw2i4o2i,
        gOIw2o4i2o = dnnl_gOIw2o4i2o,
        gOIhw2o4i2o = dnnl_gOIhw2o4i2o,
        gOIdhw2o4i2o = dnnl_gOIdhw2o4i2o,
        gOIw4i8o2i = dnnl_gOIw4i8o2i,
        gOIhw4i8o2i = dnnl_gOIhw4i8o2i,
        gOIdhw4i8o2i = dnnl_gOIdhw4i8o2i,
        gOIw4o8i2o = dnnl_gOIw4o8i2o,
        gOIhw4o8i2o = dnnl_gOIhw4o8i2o,
        gOIdhw4o8i2o = dnnl_gOIdhw4o8i2o,
    };

    /// A memory descriptor.
    struct desc {
        friend struct memory;
        /// The underlying C API data structure.
        dnnl_memory_desc_t data;

        /// Constructs a zero (empty) memory descriptor. Such a memory
        /// descriptor can be used to indicate absence of an argument.
        desc() : data() {}

        /// Constructs a memory descriptor.
        ///
        /// @note
        ///     The logical order of dimensions corresponds to the `abc...`
        ///     format tag, and the physical meaning of the dimensions depends
        ///     both on the primitive that would operate on this memory and
        ///     the operation context.
        ///
        /// @param adims Tensor dimensions.
        /// @param adata_type Data precision/type.
        /// @param aformat_tag Memory format tag.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case a
        ///     zero memory descriptor will be constructed. This flag is
        ///     optional and defaults to false.
        desc(const dims &adims, data_type adata_type, format_tag aformat_tag,
                bool allow_empty = false)
            : data() {
            validate_dims(adims);
            dnnl_status_t status = dnnl_memory_desc_init_by_tag(&data,
                    (int)adims.size(), adims.data(), convert_to_c(adata_type),
                    convert_to_c(aformat_tag));
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not construct a memory descriptor using a "
                        "format tag");
        }

        /// Constructs a memory descriptor by strides.
        ///
        /// @note
        ///     The logical order of dimensions corresponds to the `abc...`
        ///     format tag, and the physical meaning of the dimensions depends
        ///     both on the primitive that would operate on this memory and
        ///     the operation context.
        ///
        /// @param adims Tensor dimensions.
        /// @param adata_type Data precision/type.
        /// @param strides Strides for each dimension.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case a
        ///     zero memory descriptor will be constructed. This flag is
        ///     optional and defaults to false.
        desc(const dims &adims, data_type adata_type, const dims &strides,
                bool allow_empty = false)
            : data() {
            validate_dims(adims);
            if (!strides.empty()) validate_dims(strides, (int)adims.size());
            dnnl_status_t status = dnnl_memory_desc_init_by_strides(&data,
                    (int)adims.size(), adims.data(), convert_to_c(adata_type),
                    strides.empty() ? nullptr : &strides[0]);
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not construct a memory descriptor using "
                        "strides");
        }

        /// Constructs a memory descriptor from a C API data structure.
        ///
        /// @param data A C API ::dnnl_memory_desc_t structure.
        desc(const dnnl_memory_desc_t &data) : data(data) {}

        /// Constructs a memory descriptor for a region inside an area
        /// described by this memory descriptor.
        //
        /// @param adims Sizes of the region.
        /// @param offsets Offsets to the region from the encompassing
        ///     memory object in each dimension.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case a
        ///     zero memory descriptor will be returned. This flag is optional
        ///     and defaults to false.
        /// @returns A memory descriptor for the region.
        desc submemory_desc(const dims &adims, const dims &offsets,
                bool allow_empty = false) const {
            validate_dims(adims, data.ndims);
            validate_dims(offsets, data.ndims);
            dnnl_memory_desc_t sub_md = dnnl_memory_desc_t();
            dnnl_status_t status = dnnl_memory_desc_init_submemory(
                    &sub_md, &data, adims.data(), offsets.data());
            if (!allow_empty)
                error::wrap_c_api(status, "could not construct a sub-memory");
            return desc(sub_md);
        }

        /// Constructs a memory descriptor by reshaping an existing one. The
        /// new memory descriptor inherits the data type. This operation is
        /// valid only for memory descriptors that have format_kind set to
        /// #dnnl::memory::format_kind::blocked or
        /// #dnnl::memory::format_kind::any.
        ///
        /// The operation ensures that the transformation of the physical memory
        /// format corresponds to the transformation of the logical dimensions.
        /// If such transformation is impossible, the function either throws an
        /// exception (default) or returns a zero memory descriptor depending on
        /// the `allow_empty` flag.
        ///
        /// The reshape operation can be described as a combination of the
        /// following basic operations:
        /// 1. Add a dimension of size `1`. This is always possible.
        /// 2. Remove a dimension of size `1`. This is possible only if the
        ///    dimension has no padding (i.e.
        ///    `padded_dims[dim] == dims[dim] && dims[dim] == 1`).
        /// 3. Split a dimension into multiple ones. This is possible only if
        ///    the product of all tensor dimensions stays constant and the
        ///    dimension being split does not have padding (i.e.
        ///    `padded_dims[dim] = dims[dim]`).
        /// 4. Join multiple consecutive dimensions into a single one. As in
        ///    the cases above, this requires that the dimensions do not have
        ///    padding and that the memory format is such that in physical
        ///    memory these dimensions are dense and have the same order as
        ///    their logical counterparts. This also assumes that these
        ///    dimensions are not blocked.
        ///    - Here, 'dense' means:
        ///      `stride for dim[i] == (stride for dim[i + 1]) * dim[i + 1]`;
        ///    - And 'same order' means:
        ///      `i < j` if and only if `stride for dim[j] <= stride for dim[i]`.
        ///
        /// @warning
        ///     Some combinations of physical memory layout and/or offsets or
        ///     dimensions may result in a failure to make a reshape.
        ///
        /// @param adims New dimensions. The product of dimensions must
        ///     remain constant.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case a
        ///     zero memory descriptor will be returned. This flag is optional
        ///     and defaults to false.
        /// @returns A new memory descriptor with new dimensions.
        desc reshape(const dims &adims, bool allow_empty = false) const {
            if (data.ndims) validate_dims(adims, 1);
            dnnl_memory_desc_t out_md = dnnl_memory_desc_t();
            dnnl_status_t status = dnnl_memory_desc_reshape(
                    &out_md, &data, (int)adims.size(), adims.data());
            if (!allow_empty)
                error::wrap_c_api(
                        status, "could not reshape a memory descriptor");
            return desc(out_md);
        }

        /// Constructs a memory descriptor by permuting axes in an existing
        /// one.
        ///
        /// The physical memory layout representation is adjusted accordingly
        /// to maintain the consistency between the logical and physical parts
        /// of the memory descriptor. The new memory descriptor inherits the
        /// data type.
        ///
        /// The new memory descriptor inherits the data type. This operation is
        /// valid only for memory descriptors that have format_kind set to
        /// #dnnl::memory::format_kind::blocked or
        /// #dnnl::memory::format_kind::any.
        ///
        /// The logical axes will be permuted in the following manner:
        /// @code
        /// for (i = 0; i < ndims(); i++)
        ///     new_desc.dims()[permutation[i]] = dims()[i];
        /// @endcode
        ///
        /// Example:
        /// @code
        ///     std::vector<int> permutation = {1, 0}; // swap the first and
        ///                                            // the second axes
        ///     dnnl::memory::desc in_md(
        ///             {2, 3}, data_type, memory::format_tag::ab);
        ///     dnnl::memory::desc expect_out_md(
        ///             {3, 2}, data_type, memory::format_tag::ba);
        ///
        ///     assert(in_md.permute_axes(permutation) == expect_out_md);
        /// @endcode
        ///
        /// @param permutation Axes permutation.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case a
        ///     zero memory descriptor will be returned. This flag is optional
        ///     and defaults to false.
        /// @returns A new memory descriptor with new dimensions.
        desc permute_axes(const std::vector<int> &permutation,
                bool allow_empty = false) const {
            validate_dims(permutation, data.ndims);
            dnnl_memory_desc_t out_md = dnnl_memory_desc_t();
            dnnl_status_t status = dnnl_memory_desc_permute_axes(
                    &out_md, &data, permutation.data());
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not permute axes of a memory descriptor");
            return desc(out_md);
        }

        /// Returns dimensions of the memory descriptor.
        ///
        /// Potentially expensive due to the data copy involved.
        /// @returns A copy of the dimensions vector.
        memory::dims dims() const {
            return memory::dims(data.dims, data.dims + data.ndims);
        }

        /// Returns the data type of the memory descriptor.
        /// @returns The data type.
        memory::data_type data_type() const {
            return static_cast<memory::data_type>(data.data_type);
        }

        /// Returns size of the memory descriptor in bytes.
        /// @returns The number of bytes required to allocate a memory buffer
        ///     for the memory object described by this memory descriptor
        ///     including the padding area.
        size_t get_size() const { return dnnl_memory_desc_get_size(&data); }

        /// Checks whether the memory descriptor is zero (empty).
        /// @returns @c true if the memory descriptor describes an empty
        ///     memory and @c false otherwise.
        bool is_zero() const { return data.ndims == 0; }

        /// An equality operator.
        /// @param other Another memory descriptor.
        /// @returns Whether this and the other memory descriptors have
        ///     the same format tag, dimensions, strides, blocking, etc.
        bool operator==(const desc &other) const {
            return dnnl_memory_desc_equal(&data, &other.data) != 0;
        }

        /// An inequality operator.
        /// @param other Another memory descriptor.
        /// @returns Whether this and the other memory descriptors describe
        ///     different memory.
        bool operator!=(const desc &other) const { return !operator==(other); }

        /// Checks whether the object is not empty.
        ///
        /// @returns Whether the object is not empty.
        explicit operator bool() const { return data.ndims != 0; }
    };

    /// Default constructor.
    ///
    /// Constructs an empty memory object, which can be used to indicate
    /// absence of a parameter.
    memory() = default;

    /// Constructs a memory object.
    ///
    /// Unless @p handle is equal to #DNNL_MEMORY_NONE, the constructed memory
    /// object will have the underlying buffer set. In this case, the buffer
    /// will be initialized as if #dnnl::memory::set_data_handle() had been
    /// called.
    ///
    /// @sa memory::set_data_handle()
    ///
    /// @param md Memory descriptor.
    /// @param aengine Engine to store the data on.
    /// @param handle Handle of the memory buffer to use.
    ///     - A pointer to the user-allocated buffer. In this case the library
    ///       doesn't own the buffer.
    ///     - The #DNNL_MEMORY_ALLOCATE special value. Instructs the library to
    ///       allocate the buffer for the memory object. In this case the
    ///       library owns the buffer.
    ///     - #DNNL_MEMORY_NONE to create dnnl::memory without an underlying
    ///       buffer.
    memory(const desc &md, const engine &aengine, void *handle) {
        dnnl_memory_t result;
        error::wrap_c_api(
                dnnl_memory_create(&result, &md.data, aengine.get(), handle),
                "could not create a memory object");
        reset(result);
    }

    /// Constructs a memory object.
    ///
    /// The underlying buffer for the memory will be allocated by the library.
    ///
    /// @param md Memory descriptor.
    /// @param aengine Engine to store the data on.
    memory(const desc &md, const engine &aengine)
        : memory(md, aengine, DNNL_MEMORY_ALLOCATE) {}

    /// Returns the associated memory descriptor.
    desc get_desc() const {
        const dnnl_memory_desc_t *cdesc;
        error::wrap_c_api(dnnl_memory_get_memory_desc(get(), &cdesc),
                "could not get a memory descriptor from a memory object");
        return desc(*cdesc);
    }

    /// Returns the associated engine.
    engine get_engine() const {
        dnnl_engine_t c_engine;
        error::wrap_c_api(dnnl_memory_get_engine(get(), &c_engine),
                "could not get an engine from a memory object");
        return engine(c_engine, true);
    }

    /// Returns the underlying memory buffer.
    ///
    /// On the CPU engine, or when using USM, this is a pointer to the
    /// allocated memory.
    void *get_data_handle() const {
        void *handle;
        error::wrap_c_api(dnnl_memory_get_data_handle(get(), &handle),
                "could not get a native handle from a memory object");
        return handle;
    }

    /// Sets the underlying memory buffer.
    ///
    /// This function may write zero values to the memory specified by the @p
    /// handle if the memory object has a zero padding area. This may be time
    /// consuming and happens each time this function is called. The
    /// operation is always blocking and the stream parameter is a hint.
    ///
    /// @note
    ///     The zero padding is required by memory objects created with
    ///     blocked memory format tags like #dnnl_aBcd8b when any of the
    ///     dimensions is not a multiple of the corresponding block size. For
    ///     "plain" formats like #dnnl::memory::format_tag::nchw or
    ///     #dnnl::memory::format_tag::nhwc zero padding area needs to be set
    ///     up explicitly when creating the corresponding memory descriptors.
    ///     See @ref dev_guide_understanding_memory_formats for more details.
    ///
    /// @note
    ///     Even when the memory object is used to hold values that stay
    ///     constant during the execution of the program (pre-packed weights
    ///     during inference, for example), the function will still write
    ///     zeroes to the padding area if it exists. Hence, the @p handle
    ///     parameter cannot and does not have a const qualifier.
    ///
    /// @param handle Memory buffer to use. On the CPU engine or when USM is
    ///     used, the memory buffer is a pointer to the actual data. For OpenCL
    ///     it is a cl_mem. It must have at least
    ///     #dnnl::memory::desc::get_size() bytes allocated.
    /// @param astream Stream to use to execute padding in.
    void set_data_handle(void *handle, const stream &astream) const {
        error::wrap_c_api(dnnl_memory_set_data_handle_v2(
                                  get(), handle, astream.get(true)),
                "could not set native handle of a memory object");
    }

    /// Sets the underlying memory buffer.
    ///
    /// See documentation for
    /// #dnnl::memory::set_data_handle(void *, const stream &) const
    /// for more information.
    ///
    /// @param handle Memory buffer to use. For the CPU engine, the memory
    ///     buffer is a pointer to the actual data. For OpenCL it is a cl_mem.
    ///     It must have at least #dnnl::memory::desc::get_size() bytes
    ///     allocated.
    void set_data_handle(void *handle) const {
        error::wrap_c_api(
                dnnl_memory_set_data_handle_v2(get(), handle, nullptr),
                "could not set native handle of a memory object");
    }

    /// Maps a memory object and returns a host-side pointer to a memory
    /// buffer with a copy of its contents.
    ///
    /// Mapping enables read/write directly from/to the memory contents for
    /// engines that do not support direct memory access.
    ///
    /// Mapping is an exclusive operation - a memory object cannot be used in
    /// other operations until it is unmapped via #dnnl::memory::unmap_data()
    /// call.
    ///
    /// @note
    ///     Any primitives working with the memory should be completed before
    ///     the memory is mapped. Use #dnnl::stream::wait() to synchronize the
    ///     corresponding execution stream.
    ///
    /// @note
    ///     The map_data and unmap_data functions are provided mainly for
    ///     debug and testing purposes and their performance may be suboptimal.
    ///
    /// @tparam T Data type to return a pointer to.
    /// @returns Pointer to the mapped memory.
    template <typename T = void>
    T *map_data() const {
        void *mapped_ptr;
        error::wrap_c_api(dnnl_memory_map_data(get(), &mapped_ptr),
                "could not map memory object data");
        return static_cast<T *>(mapped_ptr);
    }

    /// Unmaps a memory object and writes back any changes made to the
    /// previously mapped memory buffer.
    ///
    /// @note
    ///     The map_data and unmap_data functions are provided mainly for
    ///     debug and testing purposes and their performance may be
    ///     suboptimal.
    ///
    /// @param mapped_ptr A pointer previously returned by
    ///     #dnnl::memory::map_data().
    void unmap_data(void *mapped_ptr) const {
        error::wrap_c_api(dnnl_memory_unmap_data(get(), mapped_ptr),
                "could not unmap memory object data");
    }

    static dnnl_data_type_t convert_to_c(data_type adata_type) {
        return static_cast<dnnl_data_type_t>(adata_type);
    }
    static dnnl_format_tag_t convert_to_c(format_tag format) {
        return static_cast<dnnl_format_tag_t>(format);
    }
};

inline bool operator==(dnnl_data_type_t a, memory::data_type b) {
    return a == memory::convert_to_c(b);
}
inline bool operator!=(dnnl_data_type_t a, memory::data_type b) {
    return !(a == b);
}
inline bool operator==(memory::data_type a, dnnl_data_type_t b) {
    return b == a;
}
inline bool operator!=(memory::data_type a, dnnl_data_type_t b) {
    return !(a == b);
}

inline bool operator==(dnnl_format_tag_t a, memory::format_tag b) {
    return a == memory::convert_to_c(b);
}
inline bool operator!=(dnnl_format_tag_t a, memory::format_tag b) {
    return !(a == b);
}
inline bool operator==(memory::format_tag a, dnnl_format_tag_t b) {
    return b == a;
}
inline bool operator!=(memory::format_tag a, dnnl_format_tag_t b) {
    return !(a == b);
}

/// @} dnnl_api_memory

/// @addtogroup dnnl_api_primitives
/// @{
/// @addtogroup dnnl_api_attributes Attributes
///
/// A container for parameters that extend primitives behavior.
///
/// @{

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<dnnl_post_ops_t> {
    static dnnl_status_t destructor(dnnl_post_ops_t p) {
        return dnnl_post_ops_destroy(p);
    }
};
/// @endcond

/// Post-ops.
///
/// Post-ops are computations executed after the main primitive computations
/// and are attached to the primitive via primitive attributes.
///
/// @sa @ref dev_guide_attributes_post_ops
///
struct post_ops : public handle<dnnl_post_ops_t> {
    using handle<dnnl_post_ops_t>::handle;

    /// Constructs an empty sequence of post-ops.
    post_ops() {
        dnnl_post_ops_t result;
        error::wrap_c_api(
                dnnl_post_ops_create(&result), "could not create post-ops");
        reset(result);
    }

    /// Returns the number of post-ops entries.
    int len() const { return dnnl_post_ops_len(get()); }

    /// Returns the primitive kind of post-op at entry with a certain index.
    /// @param index Index of the post-op to return the kind for.
    /// @returns Primitive kind of the post-op at the specified index.
    primitive::kind kind(int index) const {
        error::wrap_c_api(index < len() ? dnnl_success : dnnl_invalid_arguments,
                "post-ops index is out of range");
        return static_cast<primitive::kind>(
                dnnl_post_ops_get_kind(get(), index));
    }

    /// Appends an accumulation (sum) post-op. Prior to accumulating the
    /// result, the previous value would be multiplied by a scaling factor
    /// @p scale.
    ///
    /// The kind of this post-op is #dnnl::primitive::kind::sum.
    ///
    /// This feature may improve performance for cases like residual learning
    /// blocks, where the result of convolution is accumulated to the
    /// previously computed activations. The parameter @p scale may be used
    /// for the integer-based computations when the result and previous
    /// activations have different logical scaling factors.
    ///
    /// In the simplest case when the accumulation is the only post-op,
    /// the computations would be `dst[:] := scale * dst[:] + op(...)`
    /// instead of `dst[:] := op(...)`.
    ///
    /// If @p data_type is specified, the original dst tensor will be
    /// reinterpreted as a tensor with the provided data type. Because it is a
    /// reinterpretation, data_type and dst data type should have the same size.
    /// As a result, computations would be `dst[:] <- scale *
    /// as_data_type(dst[:]) + op(...)` instead of `dst[:] <- op(...)`.
    ///
    /// @note
    ///     This post-op executes in-place and does not change the
    ///     destination layout.
    ///
    /// @param scale Scaling factor.
    /// @param data_type Data type.
    void append_sum(float scale = 1.f,
            memory::data_type data_type = memory::data_type::undef) {
        if (data_type == memory::data_type::undef)
            error::wrap_c_api(dnnl_post_ops_append_sum(get(), scale),
                    "could not append a sum post-op");
        else
            error::wrap_c_api(dnnl_post_ops_append_sum_v2(get(), scale,
                                      memory::convert_to_c(data_type)),
                    "could not append a sum post-op");
    }

    /// Returns the parameters of an accumulation (sum) post-op.
    ///
    /// @param index Index of the sum post-op.
    /// @param scale Scaling factor of the sum post-op.
    void get_params_sum(int index, float &scale) const {
        error::wrap_c_api(dnnl_post_ops_get_params_sum(get(), index, &scale),
                "could not get parameters of a sum post-op");
    }

    /// Returns the parameters of an accumulation (sum) post-op.
    ///
    /// @param index Index of the sum post-op.
    /// @param scale Scaling factor of the sum post-op.
    /// @param data_type Data type of the sum post-op.
    void get_params_sum(
            int index, float &scale, memory::data_type &data_type) const {
        dnnl_data_type_t c_data_type;
        error::wrap_c_api(dnnl_post_ops_get_params_sum_v2(
                                  get(), index, &scale, &c_data_type),
                "could not get parameters of a sum post-op");
        data_type = static_cast<memory::data_type>(c_data_type);
    }

    /// Appends an elementwise post-op.
    ///
    /// The kind of this post-op is #dnnl::primitive::kind::eltwise.
    ///
    /// In the simplest case when the elementwise is the only post-op, the
    /// computations would be `dst[:] := scale * eltwise_op (op(...))` instead
    /// of `dst[:] <- op(...)`, where eltwise_op is configured with the given
    /// parameters.
    ///
    /// @param scale Scaling factor.
    /// @param aalgorithm Elementwise algorithm.
    /// @param alpha Alpha parameter for the elementwise algorithm.
    /// @param beta Beta parameter for the elementwise algorithm.
    void append_eltwise(
            float scale, algorithm aalgorithm, float alpha, float beta) {
        error::wrap_c_api(dnnl_post_ops_append_eltwise(get(), scale,
                                  convert_to_c(aalgorithm), alpha, beta),
                "could not append an elementwise post-op");
    }

    /// Returns parameters of an elementwise post-op.
    ///
    /// @param index Index of the post-op.
    /// @param scale Output scaling factor.
    /// @param aalgorithm Output elementwise algorithm kind.
    /// @param alpha Output alpha parameter for the elementwise algorithm.
    /// @param beta Output beta parameter for the elementwise algorithm.
    void get_params_eltwise(int index, float &scale, algorithm &aalgorithm,
            float &alpha, float &beta) const {
        dnnl_alg_kind_t c_alg;
        error::wrap_c_api(dnnl_post_ops_get_params_eltwise(
                                  get(), index, &scale, &c_alg, &alpha, &beta),
                "could not get parameters of an elementwise post-op");
        aalgorithm = static_cast<dnnl::algorithm>(c_alg);
    }

    /// Appends a depthwise post-op convolution with stride 1.
    ///
    /// This post-op can only be fused with a 2D 1x1 convolution (convolution
    /// with weights spatial dimension equal to 1 i.e., kh=kw=1).
    ///
    /// The kind of this post-op is #dnnl_convolution.
    ///
    /// The number of outputs for primitive remain same as before. The output
    /// size remain same as the original primitive due to stride=1.
    ///
    /// The Post-op can be defined as:
    ///
    ///      dst[:] <- scales * (conv_dw(conv_1x1))
    ///
    /// See @ref dev_guide_attributes_post_ops_depthwise and
    /// @ref dev_guide_attributes_post_ops_depthwise_fusion for more info.
    ///
    /// @param weights_data_type Weights data type of depthwise post-op
    /// @param bias_data_type Bias data type of depthwise post-op
    /// @param dst_data_type Output data type of depthwise post-op
    /// @param mask Output scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the
    ///     @p scales array. The set i-th bit indicates that a dedicated output
    ///     scaling factor is used for each index along that dimension. The mask
    ///     value of 0 implies a common scaling factor for the whole output
    ///     tensor.
    /// @param scales Output pointer to a constant array of float scaling
    ///     factors.
    void append_dw_k3s1p1(memory::data_type weights_data_type,
            memory::data_type bias_data_type, memory::data_type dst_data_type,
            int mask, const std::vector<float> &scales) {

        error::wrap_c_api(dnnl_post_ops_append_dw_k3s1p1(get(),
                                  memory::convert_to_c(weights_data_type),
                                  memory::convert_to_c(bias_data_type),
                                  memory::convert_to_c(dst_data_type),
                                  scales.size(), mask, &scales[0]),
                "could not append depthwise post-op");
    }

    /// Returns the parameters of an depthwise post-op with stride 1.
    ///
    /// @param index Index of the elementwise post-op.
    /// @param weights_data_type Weights data type of depthwise post-op
    /// @param bias_data_type Bias data type of depthwise post-op
    /// @param dst_data_type Output data type of depthwise post-op
    /// @param mask Output scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the
    ///     @p scales array. The set i-th bit indicates that a dedicated output
    ///     scaling factor is used for each index along that dimension. The mask
    ///     value of 0 implies a common scaling factor for the whole output
    ///     tensor.
    /// @param scales Output pointer to a constant array of float scaling
    ///     factors.
    void get_params_dw_k3s1p1(int index, memory::data_type &weights_data_type,
            memory::data_type &bias_data_type, memory::data_type &dst_data_type,
            int &mask, std::vector<float> &scales) const {

        dnnl_data_type_t c_weights_data_type;
        dnnl_data_type_t c_bias_data_type;
        dnnl_data_type_t c_dst_data_type;
        dnnl_dim_t count;
        int c_mask;
        const float *c_scales;
        error::wrap_c_api(dnnl_post_ops_get_params_dw_k3s1p1(get(), index,
                                  &c_weights_data_type, &c_bias_data_type,
                                  &c_dst_data_type, &count, &c_mask, &c_scales),
                "could not get parameters of depthwise post-op");

        weights_data_type = static_cast<memory::data_type>(c_weights_data_type);
        bias_data_type = static_cast<memory::data_type>(c_bias_data_type);
        dst_data_type = static_cast<memory::data_type>(c_dst_data_type);
        scales.resize(count);

        mask = c_mask;
        for (dnnl_dim_t c = 0; c < count; ++c)
            scales[c] = c_scales[c];
        return;
    }

    /// Appends a depthwise post-op convolution with stride 2.
    ///
    /// This post-op can only be fused with a 2D 1x1 convolution (convolution
    /// with weights spatial dimension equal to 1 i.e., kh=kw=1).
    ///
    /// The kind of this post-op is #dnnl_convolution.
    ///
    /// The number of outputs for primitive remain same as before. The output
    /// spatial size can be derived as below:
    ///
    /// output_height = ceil(output_height_1x1_convolution, stride)
    /// output_width = ceil(output_width_1x1_convolution, stride)
    ///
    /// The Post-op can be defined as:
    ///
    ///      dst[:] <- scales * (conv_dw(conv_1x1))
    ///
    /// See @ref dev_guide_attributes_post_ops_depthwise and
    /// @ref dev_guide_attributes_post_ops_depthwise_fusion for more info.
    ///
    /// @param weights_data_type Weights data type of depthwise post-op
    /// @param bias_data_type Bias data type of depthwise post-op
    /// @param dst_data_type Output data type of depthwise post-op
    /// @param mask Output scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the
    ///     @p scales array. The set i-th bit indicates that a dedicated output
    ///     scaling factor is used for each index along that dimension. The mask
    ///     value of 0 implies a common scaling factor for the whole output
    ///     tensor.
    /// @param scales Output pointer to a constant array of float scaling
    ///     factors.
    /// @returns #dnnl_success on success and a status describing the error
    ///     otherwise
    void append_dw_k3s2p1(memory::data_type weights_data_type,
            memory::data_type bias_data_type, memory::data_type dst_data_type,
            int mask, const std::vector<float> &scales) {

        error::wrap_c_api(dnnl_post_ops_append_dw_k3s2p1(get(),
                                  memory::convert_to_c(weights_data_type),
                                  memory::convert_to_c(bias_data_type),
                                  memory::convert_to_c(dst_data_type),
                                  scales.size(), mask, &scales[0]),
                "could not append depthwise post-op");
    }

    /// Returns the parameters of an depthwise post-op with stride 2.
    ///
    /// @param index Index of the elementwise post-op.
    /// @param weights_data_type Weights data type of depthwise post-op
    /// @param bias_data_type Bias data type of depthwise post-op
    /// @param dst_data_type Output data type of depthwise post-op
    /// @param mask Output scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the
    ///     @p scales array. The set i-th bit indicates that a dedicated output
    ///     scaling factor is used for each index along that dimension. The mask
    ///     value of 0 implies a common scaling factor for the whole output
    ///     tensor.
    /// @param scales Output pointer to a constant array of float scaling
    ///     factors.
    void get_params_dw_k3s2p1(int index, memory::data_type &weights_data_type,
            memory::data_type &bias_data_type, memory::data_type &dst_data_type,
            int &mask, std::vector<float> &scales) const {

        dnnl_data_type_t c_weights_data_type;
        dnnl_data_type_t c_bias_data_type;
        dnnl_data_type_t c_dst_data_type;
        dnnl_dim_t count;
        int c_mask;
        const float *c_scales;
        error::wrap_c_api(dnnl_post_ops_get_params_dw_k3s2p1(get(), index,
                                  &c_weights_data_type, &c_bias_data_type,
                                  &c_dst_data_type, &count, &c_mask, &c_scales),
                "could not get parameters of depthwise post-op");

        weights_data_type = static_cast<memory::data_type>(c_weights_data_type);
        bias_data_type = static_cast<memory::data_type>(c_bias_data_type);
        dst_data_type = static_cast<memory::data_type>(c_dst_data_type);
        scales.resize(count);

        mask = c_mask;
        for (dnnl_dim_t c = 0; c < count; ++c)
            scales[c] = c_scales[c];
        return;
    }

    /// Appends a binary post-op.
    ///
    /// The kind of this post operation is #dnnl_binary.
    ///
    /// In the simplest case when the binary is the only post operation, the
    /// computations would be:
    ///
    ///     dst[:] <- binary_op (dst[:], another_input[:])
    ///
    /// where binary_op is configured with the given parameters. binary_op
    /// supports broadcast semantics for a second operand.
    ///
    /// @param aalgorithm Binary algorithm for the post-op.
    /// @param src1_desc Memory descriptor of a second operand.
    void append_binary(algorithm aalgorithm, const memory::desc &src1_desc) {
        error::wrap_c_api(dnnl_post_ops_append_binary(get(),
                                  convert_to_c(aalgorithm), &src1_desc.data),
                "could not append a binary post-op");
    }

    /// Returns the parameters of a binary post-op.
    ///
    /// @param index Index of the binary post-op.
    /// @param aalgorithm Output binary algorithm kind.
    /// @param src1_desc Output memory descriptor of a second operand.
    void get_params_binary(
            int index, algorithm &aalgorithm, memory::desc &src1_desc) const {
        dnnl_alg_kind_t c_alg;
        const dnnl_memory_desc_t *data;
        error::wrap_c_api(
                dnnl_post_ops_get_params_binary(get(), index, &c_alg, &data),
                "could not get parameters of a binary post-op");
        aalgorithm = static_cast<dnnl::algorithm>(c_alg);
        src1_desc.data = *data;
    }
};

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<dnnl_primitive_attr_t> {
    static dnnl_status_t destructor(dnnl_primitive_attr_t p) {
        return dnnl_primitive_attr_destroy(p);
    }
};
/// @endcond

/// Primitive attributes.
///
/// @sa @ref dev_guide_attributes
struct primitive_attr : public handle<dnnl_primitive_attr_t> {
    using handle<dnnl_primitive_attr_t>::handle;

    /// Constructs default (empty) primitive attributes.
    primitive_attr() {
        dnnl_primitive_attr_t result;
        error::wrap_c_api(dnnl_primitive_attr_create(&result),
                "could not create primitive attribute");
        reset(result);
    }

    /// Creates primitive attributes from a C API ::dnnl_primitive_attr_t
    /// handle. The resulting handle is not weak and the C handle will be
    /// destroyed during the destruction of the C++ object.
    ///
    /// @param attr The C API primitive attributes.
    primitive_attr(dnnl_primitive_attr_t attr)
        : handle<dnnl_primitive_attr_t>(attr) {}

    /// Returns the scratchpad mode.
    scratchpad_mode get_scratchpad_mode() const {
        dnnl_scratchpad_mode_t result;
        error::wrap_c_api(
                dnnl_primitive_attr_get_scratchpad_mode(get(), &result),
                "could not get scratchpad mode primitive attribute");
        return scratchpad_mode(result);
    }

    /// Sets scratchpad mode.
    ///
    /// @param mode Specified scratchpad mode.
    void set_scratchpad_mode(scratchpad_mode mode) {
        error::wrap_c_api(dnnl_primitive_attr_set_scratchpad_mode(
                                  get(), dnnl::convert_to_c(mode)),
                "could not set scratchpad mode primitive attribute");
    }

    /// Returns output scaling factors correspondence mask and values.
    ///
    /// @param mask Scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the @p
    ///     scales vector. The set i-th bit indicates that a dedicated output
    ///     scaling factor is used for each index along that dimension. The
    ///     mask value of 0 implies a common output scaling factor for the
    ///     whole output tensor.
    /// @param scales Vector of output scaling factors.
    void get_output_scales(int &mask, std::vector<float> &scales) const {
        dnnl_dim_t count;
        int c_mask;
        const float *c_scales;
        error::wrap_c_api(dnnl_primitive_attr_get_output_scales(
                                  get(), &count, &c_mask, &c_scales),
                "could not get output scales primitive attribute");
        scales.resize(count);

        mask = c_mask;
        for (dnnl_dim_t c = 0; c < count; ++c)
            scales[c] = c_scales[c];
    }

    /// Sets output scaling factors correspondence mask and values.
    ///
    /// Example usage:
    /// @code
    ///     int mb = 32, oc = 32,
    ///         oh = 14, ow = 14; // convolution output params
    ///     // unique output scales per output channel
    ///     vector<float> scales = { ... };
    ///     int oc_dim = 1; // mb_dim = 0, channel_dim = 1, height_dim = 2, ...
    ///
    ///     // construct a convolution descriptor
    ///     dnnl::convolution::desc conv_d;
    ///
    ///     dnnl::primitive_attr attr;
    ///     attr.set_output_scales(attr, oc, 1 << oc_dim, scales);
    ///
    ///     dnnl::primitive_desc conv_pd(conv_d, attr, engine);
    /// @endcode
    ///
    /// @note
    ///     The order of dimensions does not depend on how elements are laid
    ///     out in memory. For example:
    ///     - for a 2D CNN activations tensor the order is always (n, c)
    ///     - for a 4D CNN activations tensor the order is always (n, c, h, w)
    ///     - for a 5D CNN weights tensor the order is always
    ///        (g, oc, ic, kh, kw)
    ///
    /// @param mask Defines the correspondence between the output tensor
    ///     dimensions and the @p scales vector. The set i-th bit indicates
    ///     that a dedicated scaling factor is used for each index along that
    ///     dimension. Set the mask to 0 to use a common output scaling factor
    ///     for the whole output tensor.
    /// @param scales Constant vector of output scaling factors. If the
    ///     scaling factors are known at the time of this call, the following
    ///     equality must hold:
    ///     \f$scales.size() = \prod\limits_{d \in mask} output.dims[d].\f$
    ///     Violations can only be detected when the attributes
    ///     are used to create a primitive descriptor.
    ///     If the scaling factors are not known at the time of the call,
    ///     this vector must contain a single #DNNL_RUNTIME_F32_VAL value and
    ///     the output scaling factors must be passed at execution time as an
    ///     argument with index #DNNL_ARG_ATTR_OUTPUT_SCALES.
    void set_output_scales(int mask, const std::vector<float> &scales) {
        error::wrap_c_api(
                dnnl_primitive_attr_set_output_scales(
                        get(), (dnnl_dim_t)scales.size(), mask, scales.data()),
                "could not set output scales primitive attribute");
    }

    /// Returns scaling factors correspondence mask and values for a given
    /// memory argument.
    ///
    /// @param arg Parameter argument index as passed to the
    ///     primitive::execute() call.
    /// @param mask Scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the @p
    ///     scales vector. The set i-th bit indicates that a dedicated scaling
    ///     factor is used for each index along that dimension. Set the mask to
    ///     0 to use a common scaling factor for the whole output tensor.
    /// @param scales Output vector of scaling factors.
    void get_scales(int arg, int &mask, std::vector<float> &scales) const {
        dnnl_dim_t count;
        int c_mask;
        const float *c_scales;
        error::wrap_c_api(dnnl_primitive_attr_get_scales(
                                  get(), arg, &count, &c_mask, &c_scales),
                "could not get scales primitive attributes");
        scales.resize(count);

        mask = c_mask;
        for (dnnl_dim_t c = 0; c < count; ++c)
            scales[c] = c_scales[c];
    }

    /// Sets scaling factors for primitive operations for a given memory
    /// argument.
    ///
    /// @sa dnnl_primitive_attr_set_scales
    /// @sa dnnl::primitive_attr::set_output_scales
    ///
    /// @param arg Parameter argument index as passed to the
    ///     primitive::execute() call.
    /// @param mask Scaling factors correspondence mask that defines the
    ///     correspondence between the tensor dimensions and the @p scales
    ///     vector. The set i-th bit indicates that a dedicated scaling factor
    ///     is used for each index along that dimension. Set the mask to 0 to
    ///     use a common scaling factor for the whole output tensor.
    /// @param scales Constant vector of scaling factors. The following equality
    ///     must hold:
    ///     \f$scales.size() = \prod\limits_{d \in mask} argument.dims[d].\f$
    void set_scales(int arg, int mask, const std::vector<float> &scales) {
        error::wrap_c_api(
                dnnl_primitive_attr_set_scales(get(), arg,
                        (dnnl_dim_t)scales.size(), mask, scales.data()),
                "could not set scales primitive attribute");
    }

    /// Returns zero points correspondence mask and values.
    ///
    /// @param arg Parameter argument index as passed to the
    ///     primitive::execute() call.
    /// @param mask Zero points correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the @p
    ///     zero_points vector. The set i-th bit indicates that a dedicated
    ///     zero point is used for each index along that dimension. Set the
    ///     mask to 0 to use a common zero point for the whole output tensor.
    /// @param zero_points Output vector of zero points.
    void get_zero_points(
            int arg, int &mask, std::vector<int32_t> &zero_points) const {
        dnnl_dim_t count;
        int c_mask;
        const int32_t *c_zero_points;
        error::wrap_c_api(dnnl_primitive_attr_get_zero_points(
                                  get(), arg, &count, &c_mask, &c_zero_points),
                "could not get zero points primitive attribute");
        zero_points.resize(count);

        mask = c_mask;
        for (dnnl_dim_t c = 0; c < count; ++c)
            zero_points[c] = c_zero_points[c];
    }

    /// Sets zero points for primitive operations for a given memory argument.
    ///
    /// @sa dnnl_primitive_attr_set_zero_points
    /// @sa dnnl::primitive_attr::set_output_scales
    ///
    /// @param arg Parameter argument index as passed to the
    ///     primitive::execute() call.
    /// @param mask Zero point correspondence mask that defines the
    ///     correspondence between the tensor dimensions and the @p
    ///     zero_points vector. The set i-th bit indicates that a dedicated
    ///     zero point is used for each index along that dimension. Set the
    ///     mask to 0 to use a common zero point for the whole output tensor.
    /// @param zero_points Constant vector of zero points. If the zero points
    ///     are known at the time of this call, the following equality must
    ///     hold: \f$zero\_points.size() = \prod\limits_{d \in mask}
    ///     argument.dims[d].\f$ If the zero points are not known at the time
    ///     of the call, this vector must contain a single
    ///     #DNNL_RUNTIME_S32_VAL value and the zero points must be passed at
    ///     execution time as an argument with index
    ///     #DNNL_ARG_ATTR_ZERO_POINTS.
    void set_zero_points(
            int arg, int mask, const std::vector<int32_t> &zero_points) {
        error::wrap_c_api(dnnl_primitive_attr_set_zero_points(get(), arg,
                                  (dnnl_dim_t)zero_points.size(), mask,
                                  zero_points.data()),
                "could not set zero points primitive attribute");
    }

    /// Returns post-ops previously set via set_post_ops().
    ///
    /// @returns Post-ops.
    const post_ops get_post_ops() const {
        post_ops result;
        const_dnnl_post_ops_t c_result;
        error::wrap_c_api(dnnl_primitive_attr_get_post_ops(get(), &c_result),
                "could not get post-ops primitive attribute");
        result.reset(const_cast<dnnl_post_ops_t>(c_result), true);
        return result;
    }

    /// Sets post-ops.
    ///
    /// @note
    ///     There is no way to check whether the post-ops would be supported
    ///     by the target primitive. Any error will be reported
    ///     by the respective primitive descriptor constructor.
    ///
    /// @param ops Post-ops object to copy post-ops from.
    void set_post_ops(const post_ops ops) {
        error::wrap_c_api(dnnl_primitive_attr_set_post_ops(get(), ops.get()),
                "could not set post-ops primitive attribute");
    }

    /// Sets quantization scale and shift parameters for RNN data tensors.
    ///
    /// For performance reasons, the low-precision configuration of the RNN
    /// primitives expect input activations to have the unsigned 8-bit integer
    /// data type. The scale and shift parameters are used to quantize
    /// floating-point data to unsigned integer and must be passed to the RNN
    /// primitive using attributes.
    ///
    /// The quantization formula is `scale * data + shift`.
    ///
    /// Example usage:
    /// @code
    ///     // RNN parameters
    ///     int l = 2, t = 2, mb = 32, sic = 32, slc = 32, dic = 32, dlc = 32;
    ///     // Activations quantization parameters
    ///     float scale = 63.f, shift = 64.f;
    ///
    ///     primitive_attr attr;
    ///
    ///     // Set scale and shift for int8 quantization of activation
    ///     attr.set_rnn_data_qparams(scale, shift);
    ///
    ///     // Create and configure rnn op_desc
    ///     vanilla_rnn_forward::desc rnn_d(/* arguments */);
    ///     vanilla_rnn_forward::primitive_desc rnn_d(rnn_d, attr, engine);
    /// @endcode
    ///
    /// @note
    ///     Quantization scale and shift are common for src_layer, src_iter,
    ///     dst_iter, and dst_layer.
    ///
    /// @param scale The value to scale the data by.
    /// @param shift The value to shift the data by.
    void set_rnn_data_qparams(float scale, float shift) {
        error::wrap_c_api(
                dnnl_primitive_attr_set_rnn_data_qparams(get(), scale, shift),
                "could not set RNN data quantization parameters primitive "
                "attribute");
    }

    /// Sets quantization scaling factors for RNN weights tensors. The
    /// low-precision configuration of the RNN primitives expect input weights
    /// to use the signed 8-bit integer data type. The scaling factors are
    /// used to quantize floating-point data to signed integer and must be
    /// passed to RNN primitives using attributes.
    ///
    /// @note
    ///     The dimension order is always native and does not depend on the
    ///     actual layout used. For example, five-dimensional weights always
    ///     have (l, d, i, g, o) logical dimension ordering.
    ///
    /// @note
    ///     Quantization scales are common for weights_layer and
    ///     weights_iteration
    ///
    /// @param mask Scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the @p
    ///     scales vector. The set i-th bit indicates that a dedicated scaling
    ///     factor should be used each index along that dimension. Set the
    ///     mask to 0 to use a common scaling factor for the whole output
    ///     tensor.
    /// @param scales Constant vector of output scaling factors. The following
    ///     equality must hold:
    ///     \f$scales.size() = \prod\limits_{d \in mask} weights.dims[d].\f$
    ///     Violations can only be detected when the attributes are used to
    ///     create a primitive descriptor.
    void set_rnn_weights_qparams(int mask, const std::vector<float> &scales) {
        error::wrap_c_api(dnnl_primitive_attr_set_rnn_weights_qparams(get(),
                                  (int)scales.size(), mask, scales.data()),
                "could not set RNN weights quantization parameters primitive "
                "attribute");
    }
};

/// @} dnnl_api_attributes

/// @addtogroup dnnl_api_primitives_common
/// @{

/// Base class for all primitive descriptors.
struct primitive_desc_base : public handle<dnnl_primitive_desc_t> {
    using handle<dnnl_primitive_desc_t>::handle;

    /// Default constructor. Produces an empty object.
    primitive_desc_base() = default;

    /// Returns the engine of the primitive descriptor.
    /// @returns The engine of the primitive descriptor.
    engine get_engine() const { return engine::query(*this); }

    /// Returns implementation name.
    /// @returns The implementation name.
    const char *impl_info_str() const {
        const char *res;
        error::wrap_c_api(dnnl_primitive_desc_query(
                                  get(), dnnl_query_impl_info_str, 0, &res),
                "could not retrieve implementation info string from a "
                "primitive descriptor");
        return res;
    }

    /// Returns a memory::dim value (same as int64_t).
    /// @param what The value to query.
    /// @returns The result of the query.
    memory::dim query_s64(query what) const {
        memory::dim res;
        dnnl_status_t status = dnnl_primitive_desc_query(
                get(), dnnl::convert_to_c(what), 0, &res);
        return status == dnnl_success ? res : 0;
    }

    /// Returns a memory descriptor.
    ///
    /// @note
    ///     There are also convenience methods
    ///     #dnnl::primitive_desc_base::src_desc(),
    ///     #dnnl::primitive_desc_base::dst_desc(), and others.
    ///
    /// @param what The kind of parameter to query; can be
    ///     #dnnl::query::src_md, #dnnl::query::dst_md, etc.
    /// @param idx Index of the parameter. For example, convolution bias can
    ///     be queried with what = #dnnl::query::weights_md and idx = 1.
    /// @returns The requested memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     parameter of the specified kind or index.
    memory::desc query_md(query what, int idx = 0) const {
        std::vector<query> valid_q {query::src_md, query::diff_src_md,
                query::weights_md, query::diff_weights_md, query::dst_md,
                query::diff_dst_md, query::workspace_md, query::scratchpad_md,
                query::exec_arg_md};
        if (!std::any_of(valid_q.cbegin(), valid_q.cend(),
                    [=](query q) { return what == q; }))
            DNNL_THROW_ERROR(dnnl_invalid_arguments,
                    "memory descriptor query is invalid");

        const dnnl_memory_desc_t *cdesc = dnnl_primitive_desc_query_md(
                get(), dnnl::convert_to_c(what), idx);
        return cdesc ? memory::desc(*cdesc) : memory::desc();
    }

    /// Returns a source memory descriptor.
    /// @param idx Source index.
    /// @returns Source memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     source parameter with index @p idx.
    memory::desc src_desc(int idx) const {
        return query_md(query::src_md, idx);
    }

    /// Returns a destination memory descriptor.
    /// @param idx Destination index.
    /// @returns Destination memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     destination parameter with index @p idx.
    memory::desc dst_desc(int idx) const {
        return query_md(query::dst_md, idx);
    }

    /// Returns a weights memory descriptor.
    /// @param idx Weights index.
    /// @returns Weights memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     weights parameter with index @p idx.
    memory::desc weights_desc(int idx) const {
        return query_md(query::weights_md, idx);
    }

    /// Returns a diff source memory descriptor.
    /// @param idx Diff source index.
    /// @returns Diff source memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     diff source parameter with index @p idx.
    memory::desc diff_src_desc(int idx) const {
        return query_md(query::diff_src_md, idx);
    }

    /// Returns a diff destination memory descriptor.
    /// @param idx Diff destination index.
    /// @returns Diff destination memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     diff destination parameter with index @p idx.
    memory::desc diff_dst_desc(int idx) const {
        return query_md(query::diff_dst_md, idx);
    }

    /// Returns a diff weights memory descriptor.
    /// @param idx Diff weights index.
    /// @returns Diff weights memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     diff weights parameter with index @p idx.
    memory::desc diff_weights_desc(int idx) const {
        return query_md(query::diff_weights_md, idx);
    }

    // Separate versions without the index argument for documentation
    // purposes.

    /// Returns a source memory descriptor.
    /// @returns Source memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     source parameter.
    memory::desc src_desc() const { return src_desc(0); }

    /// Returns a destination memory descriptor.
    /// @returns Destination memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     destination parameter.
    memory::desc dst_desc() const { return dst_desc(0); }

    /// Returns a weights memory descriptor.
    /// @returns Weights memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     weights parameter.
    memory::desc weights_desc() const { return weights_desc(0); }

    /// Returns a diff source memory descriptor.
    /// @returns Diff source memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     diff source memory with.
    memory::desc diff_src_desc() const { return diff_src_desc(0); }

    /// Returns a diff destination memory descriptor.
    /// @returns Diff destination memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     diff destination parameter.
    memory::desc diff_dst_desc() const { return diff_dst_desc(0); }

    /// Returns a diff weights memory descriptor.
    /// @returns Diff weights memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     diff weights parameter.
    memory::desc diff_weights_desc() const { return diff_weights_desc(0); }

    /// Returns the workspace memory descriptor.
    /// @returns Workspace memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not require
    ///     workspace parameter.
    memory::desc workspace_desc() const {
        return query_md(query::workspace_md, 0);
    }

    /// Returns the scratchpad memory descriptor.
    /// @returns scratchpad memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not require
    ///     scratchpad parameter.
    /// @sa @ref dev_guide_attributes_scratchpad
    memory::desc scratchpad_desc() const {
        return query_md(query::scratchpad_md, 0);
    }

    /// Returns the engine on which the scratchpad memory is located.
    /// @returns The engine on which the scratchpad memory is located.
    engine scratchpad_engine() const {
        dnnl_engine_t c_engine;
        error::wrap_c_api(dnnl_primitive_desc_query(get(),
                                  dnnl::convert_to_c(query::scratchpad_engine),
                                  0, &c_engine),
                "could not retrieve scratchpad engine from a primitive "
                "descriptor");
        return engine(c_engine, true);
    }

    /// Returns the primitive attributes.
    /// @returns The primitive attributes.
    primitive_attr get_primitive_attr() const {
        const_dnnl_primitive_attr_t const_c_attr;
        error::wrap_c_api(dnnl_primitive_desc_get_attr(get(), &const_c_attr),
                "could not get attributes from a primitive descriptor");
        dnnl_primitive_attr_t c_attr;
        error::wrap_c_api(dnnl_primitive_attr_clone(&c_attr, const_c_attr),
                "could not clone primitive attributes");
        return primitive_attr(c_attr);
    }

    /// Returns the kind of the primitive descriptor.
    /// @returns The kind of the primitive descriptor.
    dnnl::primitive::kind get_kind() const {
        dnnl_primitive_kind_t kind;
        error::wrap_c_api(dnnl_primitive_desc_query(get(),
                                  dnnl_query_primitive_kind, 0, (void *)&kind),
                "could not get primitive kind from a primitive descriptor");
        return static_cast<dnnl::primitive::kind>(kind);
    }

protected:
    /// Resets the value of the handle to a clone of a C API primitive
    /// descriptor.
    /// @param pd A C API primitive descriptor to clone.
    void reset_with_clone(const_dnnl_primitive_desc_t pd) {
        dnnl_primitive_desc_t new_pd;
        error::wrap_c_api(dnnl_primitive_desc_clone(&new_pd, pd),
                "could not clone a primitive descriptor");
        reset(new_pd);
    }

    /// Constructs a primitive descriptor base object from a clone of a C API
    /// primitive descriptor after verifying that it is what the caller
    /// expects.
    ///
    /// @note
    ///     The @p prim_kind should map to a primitive that does not have
    ///     different values of propagation kind (e.g. #dnnl::binary).
    /// @note
    ///     Primitive descriptor base constructed this way does not support
    ///     next_impl() (will throw).
    ///
    /// @param pd C API primitive descriptor to clone.
    /// @param prim_kind Expected primitive kind.
    primitive_desc_base(
            dnnl_primitive_desc_t pd, dnnl::primitive::kind prim_kind)
        : primitive_desc_base(pd, prim_kind, dnnl::prop_kind::undef) {}

    /// Constructs a primitive descriptor base object from a clone of a C API
    /// primitive descriptor after verifying that it is what the caller
    /// expects.
    ///
    /// @note
    ///     Primitive descriptor base constructed this way does not support
    ///     next_impl() (will throw).
    ///
    /// @param pd C API primitive descriptor to clone.
    /// @param prim_kind Expected primitive kind.
    /// @param aprop_kind Expected propagation kind.
    primitive_desc_base(dnnl_primitive_desc_t pd,
            dnnl::primitive::kind prim_kind, dnnl::prop_kind aprop_kind)
        : primitive_desc_base(pd, prim_kind, aprop_kind, aprop_kind) {}

    /// Constructs a primitive descriptor base object from a clone of a C API
    /// primitive descriptor after verifying that it is what the caller
    /// expects.
    ///
    /// @note
    ///     Primitive descriptor base constructed this way does not support
    ///     next_impl() (will throw).
    ///
    /// @param pd C API primitive descriptor to clone.
    /// @param prim_kind Expected primitive kind.
    /// @param prop_kind1 Expected propagation kind (option 1).
    /// @param prop_kind2 Expected propagation kind (option 2). This value is
    ///     checked if the check with @p prop_kind1 fails.
    primitive_desc_base(dnnl_primitive_desc_t pd,
            dnnl::primitive::kind prim_kind, dnnl::prop_kind prop_kind1,
            dnnl::prop_kind prop_kind2) {
        // It is OK to pass an empty primitive descriptor
        if (pd == nullptr) return;

        dnnl_status_t rc;

        dnnl_primitive_kind_t c_prim_kind = convert_to_c(prim_kind);
        dnnl_prop_kind_t c_prop_kind1 = convert_to_c(prop_kind1);
        dnnl_prop_kind_t c_prop_kind2 = convert_to_c(prop_kind2);

        // Check that primitive kind matches
        dnnl_primitive_kind_t pd_kind;
        rc = dnnl_primitive_desc_query(
                pd, dnnl_query_primitive_kind, 0, (void *)&pd_kind);
        error::wrap_c_api(
                rc, "could not get primitive kind from a primitive descriptor");
        if (pd_kind != c_prim_kind)
            DNNL_THROW_ERROR(dnnl_invalid_arguments,
                    "primitive descriptor operation kind mismatch");

        // Check that propagation kind matches
        dnnl_prop_kind_t pd_prop_kind;
        rc = dnnl_primitive_desc_query(
                pd, dnnl_query_prop_kind, 0, (void *)&pd_prop_kind);

        // Something went wrong
        if (rc != dnnl_success && rc != dnnl_unimplemented)
            DNNL_THROW_ERROR(dnnl_invalid_arguments,
                    "could not get propagation kind from the primitive "
                    "descriptor");

        // Everything is fine
        if ((rc == dnnl_unimplemented && c_prop_kind1 == dnnl_prop_kind_undef)
                || (rc == dnnl_success
                        && (pd_prop_kind == c_prop_kind1
                                || pd_prop_kind == c_prop_kind2))) {
            reset_with_clone(pd);
            return;
        }

        // We could get the propagation kind but there is a mismatch
        DNNL_THROW_ERROR(dnnl_invalid_arguments,
                "primitive descriptor propagation kind mismatch");
    }

    using base = primitive_desc_base;
};

/// @} dnnl_api_primitives_common

/// @addtogroup dnnl_api_reorder Reorder
///
/// A primitive to copy data between two memory objects. This primitive is
/// typically used to change the way the data is laid out in memory.
///
/// @sa @ref dev_guide_reorder in developer guide
///
/// @{

/// Reorder primitive.
struct reorder : public primitive {
    /// Primitive descriptor for a reorder primitive.
    struct primitive_desc : public primitive_desc_base {
        using primitive_desc_base::primitive_desc_base;

        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for reorder primitive.
        ///
        /// @note
        ///     If @p allow_empty is true, the constructor does not throw if a
        ///     primitive descriptor cannot be created.
        ///
        /// @param src_engine Engine on which the source memory object will be
        ///     located.
        /// @param src_md Source memory descriptor.
        /// @param dst_engine Engine on which the destination memory object
        ///     will be located.
        /// @param dst_md Destination memory descriptor.
        /// @param attr Primitive attributes to use (optional).
        /// @param allow_empty A flag signifying whether construction is allowed
        ///     to fail without throwing an exception. In this case an empty
        ///     object will be produced. This flag is optional and defaults to
        ///     false.
        primitive_desc(const engine &src_engine, const memory::desc &src_md,
                const engine &dst_engine, const memory::desc &dst_md,
                const primitive_attr &attr = primitive_attr(),
                bool allow_empty = false) {
            dnnl_primitive_desc_t result;
            dnnl_status_t status = dnnl_reorder_primitive_desc_create(&result,
                    &src_md.data, src_engine.get(), &dst_md.data,
                    dst_engine.get(), attr.get());
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a reorder "
                        "primitive");
            reset(status == dnnl_success ? result : dnnl_primitive_desc_t());
        }

        /// Constructs a primitive descriptor for reorder primitive.
        ///
        /// @param src Source memory object. It is used to obtain the source
        ///     memory descriptor and engine.
        /// @param dst Destination memory object. It is used to obtain the
        ///     destination memory descriptor and engine.
        /// @param attr Primitive attributes to use (optional).
        /// @param allow_empty A flag signifying whether construction is allowed
        ///     to fail without throwing an exception. In this case an empty
        ///     object will be produced. This flag is optional and defaults to
        ///     false.
        primitive_desc(const memory &src, const memory &dst,
                const primitive_attr &attr = primitive_attr(),
                bool allow_empty = false) {
            dnnl_primitive_desc_t result;
            auto src_md = src.get_desc();
            auto dst_md = dst.get_desc();
            dnnl_status_t status = dnnl_reorder_primitive_desc_create(&result,
                    &src_md.data, src.get_engine().get(), &dst_md.data,
                    dst.get_engine().get(), attr.get());
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a reorder "
                        "primitive");
            reset(status == dnnl_success ? result : dnnl_primitive_desc_t());
        }

        /// Constructs a primitive descriptor for reorder primitive from a C
        /// API primitive descriptor which must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for reorder primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : primitive_desc_base(pd, dnnl::primitive::kind::reorder) {}

        /// Returns the engine on which the source memory is allocated.
        /// @returns The engine on which the source memory is allocated.
        engine get_src_engine() const {
            return engine::query(*this, dnnl::query::reorder_src_engine);
        }

        /// Returns the engine on which the destination memory is allocated.
        /// @returns The engine on which the destination memory is allocated.
        engine get_dst_engine() const {
            return engine::query(*this, dnnl::query::reorder_dst_engine);
        }

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    reorder() = default;

    /// Constructs a reorder primitive.
    /// @param pd Primitive descriptor for reorder primitive.
    reorder(const primitive_desc &pd) : primitive(pd.get()) {}

    /// Constructs a reorder primitive that would reorder data between memory
    /// objects having the same memory descriptors as memory objects @p src and
    /// @p dst.
    ///
    /// @param src Source memory object.
    /// @param dst Destination memory object.
    /// @param attr Primitive attributes to use (optional).
    reorder(const memory &src, const memory &dst,
            const primitive_attr &attr = primitive_attr())
        : primitive(primitive_desc(src, dst, attr).get()) {}

    using primitive::execute;

    /// Executes the reorder primitive.
    ///
    /// @param astream Stream object. The stream must belong to the same engine
    ///     as the primitive.
    /// @param src Source memory object.
    /// @param dst Destination memory object.
    void execute(const stream &astream, memory &src, memory &dst) const {
        primitive::execute(astream, {{DNNL_ARG_FROM, src}, {DNNL_ARG_TO, dst}});
    }
};

/// @} dnnl_api_reorder

/// @addtogroup dnnl_api_concat Concat
///
/// A primitive to concatenate data by arbitrary dimension.
///
/// @sa @ref dev_guide_concat in developer guide
///
/// @{

/// @cond DO_NOT_DOCUMENT_THIS
inline std::vector<dnnl_memory_desc_t> convert_to_c(
        const std::vector<memory::desc> &mems) {
    std::vector<dnnl_memory_desc_t> c_mems;
    c_mems.reserve(mems.size());
    for (const auto &s : mems)
        c_mems.push_back(s.data);
    return c_mems;
}
/// @endcond

/// Tensor concatenation (concat) primitive.
struct concat : public primitive {
    /// Primitive descriptor for a concat primitive.
    struct primitive_desc : public primitive_desc_base {
        using primitive_desc_base::primitive_desc_base;

        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an out-of-place concatenation
        /// primitive.
        ///
        /// @param dst Destination memory descriptor.
        /// @param concat_dimension Source tensors will be concatenated over
        ///     dimension with this index. Note that order of dimensions does
        ///     not depend on memory format.
        /// @param srcs Vector of source memory descriptors.
        /// @param aengine Engine to perform the operation on.
        /// @param attr Primitive attributes to use (optional).
        primitive_desc(const memory::desc &dst, int concat_dimension,
                const std::vector<memory::desc> &srcs, const engine &aengine,
                const primitive_attr &attr = primitive_attr()) {
            auto c_srcs = convert_to_c(srcs);

            dnnl_primitive_desc_t result;
            error::wrap_c_api(
                    dnnl_concat_primitive_desc_create(&result, &dst.data,
                            (int)c_srcs.size(), concat_dimension, c_srcs.data(),
                            attr.get(), aengine.get()),
                    "could not create a primitive descriptor for a concat "
                    "primitive");
            reset(result);
        }

        /// Constructs a primitive descriptor for an out-of-place concatenation
        /// primitive.
        ///
        /// This version derives the destination memory descriptor
        /// automatically.
        ///
        /// @param concat_dimension Source tensors will be concatenated over
        ///     dimension with this index. Note that order of dimensions does
        ///     not depend on memory format.
        /// @param srcs Vector of source memory descriptors.
        /// @param aengine Engine to perform the operation on.
        /// @param attr Primitive attributes to use (optional).
        primitive_desc(int concat_dimension,
                const std::vector<memory::desc> &srcs, const engine &aengine,
                const primitive_attr &attr = primitive_attr()) {
            auto c_api_srcs = convert_to_c(srcs);

            dnnl_primitive_desc_t result;
            error::wrap_c_api(
                    dnnl_concat_primitive_desc_create(&result, nullptr,
                            (int)c_api_srcs.size(), concat_dimension,
                            c_api_srcs.data(), attr.get(), aengine.get()),
                    "could not create a primitive descriptor for a concat "
                    "primitive");
            reset(result);
        }

        /// Constructs a primitive descriptor for concat primitive from a C
        /// API primitive descriptor which must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for concat primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : primitive_desc_base(pd, dnnl::primitive::kind::concat) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc(int)const
        memory::desc src_desc(int idx = 0) const { return base::src_desc(idx); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    concat() = default;

    /// Constructs a concatenation primitive.
    /// @param pd Primitive descriptor for concatenation primitive.
    concat(const primitive_desc &pd) : primitive(pd.get()) {}
};

/// @} dnnl_api_concat

/// @addtogroup dnnl_api_sum Sum
///
/// A primitive to sum multiple tensors.
///
/// @sa @ref dev_guide_sum in developer guide
///
/// @{

/// Out-of-place summation (sum) primitive.
struct sum : public primitive {
    /// Primitive descriptor for a sum primitive.
    struct primitive_desc : public primitive_desc_base {
        using primitive_desc_base::primitive_desc_base;

        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a sum primitive.
        ///
        /// @param dst Destination memory descriptor.
        /// @param scales Vector of scales to multiply data in each source
        ///     memory by.
        /// @param srcs Vector of source memory descriptors.
        /// @param aengine Engine to perform the operation on.
        /// @param attr Primitive attributes to use (optional).
        primitive_desc(const memory::desc &dst,
                const std::vector<float> &scales,
                const std::vector<memory::desc> &srcs, const engine &aengine,
                const primitive_attr &attr = primitive_attr()) {
            validate_container_size(scales,
                    "counts of scales and sources are not equal",
                    (int)srcs.size(), (int)srcs.size());

            auto c_api_srcs = convert_to_c(srcs);

            dnnl_primitive_desc_t result;
            error::wrap_c_api(
                    dnnl_sum_primitive_desc_create(&result, &dst.data,
                            (int)c_api_srcs.size(), scales.data(),
                            c_api_srcs.data(), attr.get(), aengine.get()),
                    "could not create a primitive descriptor for a sum "
                    "primitive");
            reset(result);
        }

        /// Constructs a primitive descriptor for a sum primitive.
        ///
        /// This version derives the destination memory descriptor
        /// automatically.
        ///
        /// @param scales Vector of scales by which to multiply data in each
        ///     source memory object.
        /// @param srcs Vector of source memory descriptors.
        /// @param aengine Engine on which to perform the operation.
        /// @param attr Primitive attributes to use (optional).
        primitive_desc(const std::vector<float> &scales,
                const std::vector<memory::desc> &srcs, const engine &aengine,
                const primitive_attr &attr = primitive_attr()) {
            validate_container_size(scales,
                    "counts of scales and sources are not equal",
                    (int)srcs.size(), (int)srcs.size());

            auto c_api_srcs = convert_to_c(srcs);
            dnnl_primitive_desc_t result;
            error::wrap_c_api(
                    dnnl_sum_primitive_desc_create(&result, nullptr,
                            (int)c_api_srcs.size(), scales.data(),
                            c_api_srcs.data(), attr.get(), aengine.get()),
                    "could not create a primitive descriptor for a sum "
                    "primitive");
            reset(result);
        }

        /// Constructs a primitive descriptor for sum primitive from a C API
        /// primitive descriptor which must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for reorder primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : primitive_desc_base(pd, dnnl::primitive::kind::sum) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc(int)const
        memory::desc src_desc(int idx = 0) const { return base::src_desc(idx); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    sum() = default;

    /// Constructs a sum primitive.
    /// @param pd Primitive descriptor for sum primitive.
    sum(const primitive_desc &pd) : primitive(pd.get()) {}
};

/// @} dnnl_api_sum

/// @addtogroup dnnl_api_primitives_common
/// @{

/// A base class for descriptors of all primitives that have an operation
/// descriptor and that support iteration over multiple implementations.
struct primitive_desc : public primitive_desc_base {
    using primitive_desc_base::primitive_desc_base;

    primitive_desc() = default;

    /// Constructs a primitive descriptor.
    ///
    /// @note
    ///     If @p allow_empty is true, the constructor does not throw if a
    ///     primitive descriptor cannot be created. But calling next_impl() in
    ///     this case will throw.
    ///
    /// @note
    ///     This is a low-level implementation detail that is typically not
    ///     needed in application code.
    ///
    /// @param desc Constant C API operation descriptor.
    /// @param attr Pointer to primitive attributes. It is safe to pass
    ///     nullptr to indicate absence of attributes.
    /// @param aengine Engine to use.
    /// @param hint_fwd_pd C API primitive descriptor for a forward
    ///     propagation primitive. It is used as a hint for deciding which
    ///     memory format to use for backward propagation or weights gradient.
    /// @param allow_empty A flag signifying whether construction is allowed
    ///     to fail without throwing an exception. In this case an empty
    ///     object will be produced. This flag is optional and defaults to
    ///     false.
    primitive_desc(const_dnnl_op_desc_t desc, const primitive_attr *attr,
            const engine &aengine, const_dnnl_primitive_desc_t hint_fwd_pd,
            bool allow_empty = false)
        : allow_empty_(allow_empty) {
        dnnl_primitive_desc_iterator_t iterator = nullptr;
        dnnl_status_t status = dnnl_primitive_desc_iterator_create(&iterator,
                desc, attr ? attr->get() : nullptr, aengine.get(), hint_fwd_pd);
        if (!allow_empty)
            error::wrap_c_api(
                    status, "could not create a primitive descriptor iterator");
        pd_iterator.reset(iterator);
        fetch_impl();
    }

    /// Advances the primitive iterator to the next implementation.
    ///
    /// @returns @c true on success, and @c false if the last implementation
    ///     reached, and the primitive descriptor itself is kept unchanged
    bool next_impl() {
        dnnl_status_t status
                = dnnl_primitive_desc_iterator_next(pd_iterator.get());
        if (status == dnnl_iterator_ends) return false;
        error::wrap_c_api(
                status, "could not advance a primitive descriptor iterator");
        fetch_impl();
        return true;
    }

private:
    bool allow_empty_ = false;
    handle<dnnl_primitive_desc_iterator_t> pd_iterator;
    void fetch_impl() {
        dnnl_primitive_desc_t pd = dnnl_primitive_desc_iterator_fetch(
                pd_iterator.get(allow_empty_));
        error::wrap_c_api(pd != nullptr || allow_empty_ ? dnnl_success
                                                        : dnnl_out_of_memory,
                "could not fetch a primitive descriptor from a primitive "
                "descriptor iterator");
        reset(pd);
    }
};

/// @} dnnl_api_primitives_common

/// @addtogroup dnnl_api_convolution Convolution
///
/// A primitive to perform 1D, 2D or 3D convolution. Supported variants are
/// forward propagation, backward propagation, and weights gradient with or
/// without bias.
///
/// @sa @ref dev_guide_convolution in developer guide
///
/// @{

/// Convolution forward propagation primitive.
struct convolution_forward : public primitive {
    /// Descriptor for a convolution forward propagation primitive.
    struct desc {
        dnnl_convolution_desc_t data;

        /// Constructs a descriptor for a convolution forward propagation
        /// primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param bias_desc Bias memory descriptor. Passing zero memory
        ///     descriptor disables the bias term.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &bias_desc, const memory::desc &dst_desc,
                const memory::dims &strides, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(
                    dnnl_convolution_forward_desc_init(&data,
                            dnnl::convert_to_c(aprop_kind),
                            convert_to_c(aalgorithm), &src_desc.data,
                            &weights_desc.data, &bias_desc.data, &dst_desc.data,
                            &strides[0], &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a convolution forward "
                    "propagation primitive");
        }

        /// Constructs a descriptor for a convolution forward propagation
        /// primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(
                    dnnl_convolution_forward_desc_init(&data,
                            dnnl::convert_to_c(aprop_kind),
                            convert_to_c(aalgorithm), &src_desc.data,
                            &weights_desc.data, nullptr, &dst_desc.data,
                            &strides[0], &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a convolution forward "
                    "propagation primitive");
        }

        /// Constructs a descriptor for a dilated convolution forward
        /// propagation primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param bias_desc Bias memory descriptor. Passing zero memory
        ///     descriptor disables the bias term.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &bias_desc, const memory::desc &dst_desc,
                const memory::dims &strides, const memory::dims &dilates,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(dilates, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(dnnl_dilated_convolution_forward_desc_init(&data,
                                      dnnl::convert_to_c(aprop_kind),
                                      convert_to_c(aalgorithm), &src_desc.data,
                                      &weights_desc.data, &bias_desc.data,
                                      &dst_desc.data, &strides[0], &dilates[0],
                                      &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a dilated convolution "
                    "forward propagation primitive");
        }

        /// Constructs a descriptor for a dilated convolution forward
        /// propagation primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(dilates, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(dnnl_dilated_convolution_forward_desc_init(&data,
                                      dnnl::convert_to_c(aprop_kind),
                                      convert_to_c(aalgorithm), &src_desc.data,
                                      &weights_desc.data, nullptr,
                                      &dst_desc.data, &strides[0], &dilates[0],
                                      &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a dilated convolution "
                    "forward propagation primitive");
        }
    };

    /// Primitive descriptor for a convolution forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a convolution forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a convolution forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case
        ///     an empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a convolution forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a convolution forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case
        ///     an empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a convolution forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a convolution forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::convolution,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// Returns the bias memory descriptor.
        /// @returns The bias memory descriptor.
        /// @returns A zero memory descriptor of the primitive does not have a
        ///     bias parameter.
        memory::desc bias_desc() const { return base::weights_desc(1); }
    };

    /// Default constructor. Produces an empty object.
    convolution_forward() = default;

    /// Constructs a convolution forward propagation primitive.
    /// @param pd Primitive descriptor for a convolution forward propagation
    ///     primitive.
    convolution_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Convolution backward propagation primitive.
struct convolution_backward_data : public primitive {

    /// Descriptor for a convolution backward propagation primitive.
    struct desc {
        dnnl_convolution_desc_t data;

        /// Constructs a descriptor for a convolution backward propagation
        /// primitive.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, diff_src_desc.data.ndims - 2);
            error::wrap_c_api(
                    dnnl_convolution_backward_data_desc_init(&data,
                            convert_to_c(aalgorithm), &diff_src_desc.data,
                            &weights_desc.data, &diff_dst_desc.data,
                            &strides[0], &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a convolution backward "
                    "propagation primitive");
        }

        /// Constructs a descriptor for dilated convolution backward
        /// propagation primitive.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides, diff_src_desc.data.ndims - 2);
            memory::validate_dims(dilates, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, diff_src_desc.data.ndims - 2);
            error::wrap_c_api(
                    dnnl_dilated_convolution_backward_data_desc_init(&data,
                            convert_to_c(aalgorithm), &diff_src_desc.data,
                            &weights_desc.data, &diff_dst_desc.data,
                            &strides[0], &dilates[0], &padding_l[0],
                            &padding_r[0]),
                    "could not create a descriptor for a dilated convolution "
                    "backward propagation primitive");
        }
    };

    /// Primitive descriptor for a convolution backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a convolution backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a convolution backward propagation
        ///     primitive.
        /// @param aengine Engine to perform the operation on.
        /// @param hint_fwd_pd Primitive descriptor for a convolution forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case
        ///     an empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const convolution_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, nullptr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a convolution backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a convolution backward propagation
        ///     primitive.
        /// @param aengine Engine to perform the operation on.
        /// @param attr Primitive attributes to use.
        /// @param hint_fwd_pd Primitive descriptor for a convolution forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case
        ///     an empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine,
                const convolution_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a convolution backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a convolution backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::convolution,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    convolution_backward_data() = default;

    /// Constructs a convolution backward propagation primitive.
    /// @param pd Primitive descriptor for a convolution backward propagation
    ///     primitive.
    convolution_backward_data(const primitive_desc &pd) : primitive(pd) {}
};

/// Convolution weights gradient primitive.
struct convolution_backward_weights : public primitive {
    /// Descriptor for a convolution weights gradient primitive.
    struct desc {
        dnnl_convolution_desc_t data;

        /// Constructs a descriptor for a convolution weights gradient primitive
        /// with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_bias_desc Diff bias memory descriptor. Passing zero
        ///     memory descriptor disables the bias term.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(
                    dnnl_convolution_backward_weights_desc_init(&data,
                            convert_to_c(aalgorithm), &src_desc.data,
                            &diff_weights_desc.data, &diff_bias_desc.data,
                            &diff_dst_desc.data, &strides[0], &padding_l[0],
                            &padding_r[0]),
                    "could not create a descriptor for a convolution weights "
                    "update primitive");
        }

        /// Constructs a descriptor for a convolution weights gradient primitive
        /// without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(dnnl_convolution_backward_weights_desc_init(&data,
                                      convert_to_c(aalgorithm), &src_desc.data,
                                      &diff_weights_desc.data, nullptr,
                                      &diff_dst_desc.data, &strides[0],
                                      &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a convolution weights "
                    "update primitive");
        }

        /// Constructs a descriptor for a dilated convolution weights gradient
        /// primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_bias_desc Diff bias memory descriptor. Passing zero
        ///     memory descriptor disables the bias term.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(dilates, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(
                    dnnl_dilated_convolution_backward_weights_desc_init(&data,
                            convert_to_c(aalgorithm), &src_desc.data,
                            &diff_weights_desc.data, &diff_bias_desc.data,
                            &diff_dst_desc.data, &strides[0], &dilates[0],
                            &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a dilated convolution "
                    "weights gradient primitive");
        }

        /// Constructs a descriptor for a dilated convolution weights gradient
        /// primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(dilates, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(
                    dnnl_dilated_convolution_backward_weights_desc_init(&data,
                            convert_to_c(aalgorithm), &src_desc.data,
                            &diff_weights_desc.data, nullptr,
                            &diff_dst_desc.data, &strides[0], &dilates[0],
                            &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a dilated convolution "
                    "weights gradient primitive");
        }
    };

    /// Primitive descriptor for a convolution weights gradient primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a convolution weights gradient
        /// primitive.
        ///
        /// @param adesc Descriptor for a convolution weights gradient primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a convolution forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case
        ///     an empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const convolution_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, nullptr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a convolution weights gradient
        /// primitive.
        ///
        /// @param adesc Descriptor for a convolution weights gradient primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a convolution forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case
        ///     an empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine,
                const convolution_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a convolution weights gradient
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a convolution weights
        ///     gradient primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::convolution,
                    dnnl::prop_kind::backward_weights) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_weights_desc()const
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// Returns the diff bias memory descriptor.
        /// @returns The diff bias memory descriptor.
        /// @returns A zero memory descriptor of the primitive does not have a
        ///          diff bias parameter.
        memory::desc diff_bias_desc() const {
            return base::diff_weights_desc(1);
        }
    };

    /// Default constructor. Produces an empty object.
    convolution_backward_weights() = default;

    /// Constructs a convolution weights gradient primitive.
    /// @param pd Primitive descriptor for a convolution weights gradient
    ///     primitive.
    convolution_backward_weights(const primitive_desc &pd) : primitive(pd) {}
};

/// @} dnnl_api_convolution
//
/// @addtogroup dnnl_api_deconvolution Deconvolution
///
/// A primitive to perform 1D, 2D or 3D deconvolution. Supported variants are
/// forward propagation, backward propagation, and weights gradient with or
/// without bias.
///
/// @{

/// Deconvolution forward propagation primitive.
struct deconvolution_forward : public primitive {
    /// Descriptor for a deconvolution forward propagation primitive.
    struct desc {
        dnnl_deconvolution_desc_t data;

        /// Constructs a descriptor for a deconvolution forward propagation
        /// primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Deconvolution algorithm:
        ///     #dnnl::algorithm::deconvolution_direct, and
        ///     #dnnl::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param bias_desc Bias memory descriptor. Passing zero memory
        ///     descriptor disables the bias term.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &bias_desc, const memory::desc &dst_desc,
                const memory::dims &strides, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(
                    dnnl_deconvolution_forward_desc_init(&data,
                            dnnl::convert_to_c(aprop_kind),
                            convert_to_c(aalgorithm), &src_desc.data,
                            &weights_desc.data, &bias_desc.data, &dst_desc.data,
                            &strides[0], &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a deconvolution forward "
                    "propagation primitive");
        }

        /// Constructs a descriptor for a deconvolution forward propagation
        /// primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Deconvolution algorithm:
        ///     #dnnl::algorithm::deconvolution_direct, and
        ///     #dnnl::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(
                    dnnl_deconvolution_forward_desc_init(&data,
                            dnnl::convert_to_c(aprop_kind),
                            convert_to_c(aalgorithm), &src_desc.data,
                            &weights_desc.data, nullptr, &dst_desc.data,
                            &strides[0], &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a deconvolution forward "
                    "propagation primitive");
        }

        /// Constructs a descriptor for a dilated deconvolution forward
        /// propagation primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Deconvolution algorithm:
        ///     #dnnl::algorithm::deconvolution_direct, and
        ///     #dnnl::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param bias_desc Bias memory descriptor. Passing zero memory
        ///     descriptor disables the bias term.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &bias_desc, const memory::desc &dst_desc,
                const memory::dims &strides, const memory::dims &dilates,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(dilates, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(dnnl_dilated_deconvolution_forward_desc_init(
                                      &data, dnnl::convert_to_c(aprop_kind),
                                      convert_to_c(aalgorithm), &src_desc.data,
                                      &weights_desc.data, &bias_desc.data,
                                      &dst_desc.data, &strides[0], &dilates[0],
                                      &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a dilated deconvolution "
                    "forward propagation primitive");
        }

        /// Constructs a descriptor for a dilated deconvolution forward
        /// propagation primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Deconvolution algorithm:
        ///     #dnnl::algorithm::deconvolution_direct, and
        ///     #dnnl::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(dilates, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(dnnl_dilated_deconvolution_forward_desc_init(
                                      &data, dnnl::convert_to_c(aprop_kind),
                                      convert_to_c(aalgorithm), &src_desc.data,
                                      &weights_desc.data, nullptr,
                                      &dst_desc.data, &strides[0], &dilates[0],
                                      &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a dilated deconvolution "
                    "forward propagation primitive");
        }
    };

    /// Primitive descriptor for a deconvolution forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a deconvolution forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a deconvolution forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a deconvolution forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a deconvolution forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::deconvolution,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::convolution_forward::primitive_desc::bias_desc()const
        memory::desc bias_desc() const { return base::weights_desc(1); }
    };

    /// Default constructor. Produces an empty object.
    deconvolution_forward() = default;

    /// Constructs a deconvolution forward propagation primitive.
    /// @param pd Primitive descriptor for a deconvolution forward propagation
    ///     primitive.
    deconvolution_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Deconvolution backward propagation primitive.
struct deconvolution_backward_data : public primitive {
    /// Descriptor for a deconvolution backward propagation primitive.
    struct desc {
        dnnl_deconvolution_desc_t data;

        /// Constructs a descriptor for a deconvolution backward propagation
        /// primitive.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aalgorithm Deconvolution algorithm
        ///     (#dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd).
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, diff_src_desc.data.ndims - 2);
            error::wrap_c_api(
                    dnnl_deconvolution_backward_data_desc_init(&data,
                            convert_to_c(aalgorithm), &diff_src_desc.data,
                            &weights_desc.data, &diff_dst_desc.data,
                            &strides[0], &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a deconvolution "
                    "backward propagation primitive");
        }

        /// Constructs a descriptor for a dilated deconvolution backward
        /// propagation primitive.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aalgorithm Deconvolution algorithm
        ///     (#dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd).
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides, diff_src_desc.data.ndims - 2);
            memory::validate_dims(dilates, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, diff_src_desc.data.ndims - 2);
            error::wrap_c_api(
                    dnnl_dilated_deconvolution_backward_data_desc_init(&data,
                            convert_to_c(aalgorithm), &diff_src_desc.data,
                            &weights_desc.data, &diff_dst_desc.data,
                            &strides[0], &dilates[0], &padding_l[0],
                            &padding_r[0]),
                    "could not create a descriptor for a dilated deconvolution "
                    "backward propagation primitive");
        }
    };

    /// Primitive descriptor for a deconvolution backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a deconvolution backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a deconvolution backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a deconvolution forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const deconvolution_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, nullptr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a deconvolution backward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a deconvolution forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine,
                const deconvolution_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a deconvolution backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::deconvolution,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    deconvolution_backward_data() = default;

    /// Constructs a deconvolution backward propagation primitive.
    /// @param pd Primitive descriptor for a deconvolution backward propagation
    ///     primitive.
    deconvolution_backward_data(const primitive_desc &pd) : primitive(pd) {}
};

/// Deconvolution weights gradient primitive.
struct deconvolution_backward_weights : public primitive {
    /// Descriptor for a deconvolution weights gradient primitive.
    struct desc {
        dnnl_deconvolution_desc_t data;

        /// Constructs a descriptor for a deconvolution weights gradient
        /// primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aalgorithm Deconvolution algorithm. Possible values are
        ///     #dnnl::algorithm::deconvolution_direct, and
        ///     #dnnl::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_bias_desc Diff bias memory descriptor. Passing zero
        ///     memory descriptor disables the bias term.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(
                    dnnl_deconvolution_backward_weights_desc_init(&data,
                            convert_to_c(aalgorithm), &src_desc.data,
                            &diff_weights_desc.data, &diff_bias_desc.data,
                            &diff_dst_desc.data, &strides[0], &padding_l[0],
                            &padding_r[0]),
                    "could not create a descriptor for a deconvolution weights "
                    "update primitive");
        }

        /// Constructs a descriptor for a deconvolution weights gradient
        /// primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aalgorithm Deconvolution algorithm. Possible values are
        ///     #dnnl::algorithm::deconvolution_direct, and
        ///     #dnnl::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(dnnl_deconvolution_backward_weights_desc_init(
                                      &data, convert_to_c(aalgorithm),
                                      &src_desc.data, &diff_weights_desc.data,
                                      nullptr, &diff_dst_desc.data, &strides[0],
                                      &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a deconvolution weights "
                    "update primitive");
        }

        /// Constructs a descriptor for a dilated deconvolution weights gradient
        /// primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aalgorithm Deconvolution algorithm. Possible values are
        ///     #dnnl::algorithm::deconvolution_direct, and
        ///     #dnnl::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_bias_desc Diff bias memory descriptor. Passing zero
        ///     memory descriptor disables the bias term.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(dilates, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(
                    dnnl_dilated_deconvolution_backward_weights_desc_init(&data,
                            convert_to_c(aalgorithm), &src_desc.data,
                            &diff_weights_desc.data, &diff_bias_desc.data,
                            &diff_dst_desc.data, &strides[0], &dilates[0],
                            &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a dilated deconvolution "
                    "weights gradient primitive");
        }

        /// Constructs a descriptor for a dilated deconvolution weights gradient
        /// primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aalgorithm Deconvolution algorithm. Possible values are
        ///     #dnnl::algorithm::deconvolution_direct, and
        ///     #dnnl::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(dilates, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(
                    dnnl_dilated_deconvolution_backward_weights_desc_init(&data,
                            convert_to_c(aalgorithm), &src_desc.data,
                            &diff_weights_desc.data, nullptr,
                            &diff_dst_desc.data, &strides[0], &dilates[0],
                            &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a dilated deconvolution "
                    "weights gradient primitive");
        }
    };

    /// Primitive descriptor for a deconvolution weights gradient primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a deconvolution weights
        /// update primitive.
        ///
        /// @param adesc Descriptor for a deconvolution weights gradient
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a deconvolution forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception.  In this case
        ///     an empty object will be produced.  This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const deconvolution_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, nullptr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution weights
        /// update primitive.
        ///
        /// @param adesc Descriptor for a deconvolution weights gradient
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a deconvolution forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine,
                const deconvolution_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution weights
        /// gradient primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a deconvolution weights
        ///     gradient primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::deconvolution,
                    dnnl::prop_kind::backward_weights) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_weights_desc()const
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::convolution_backward_weights::primitive_desc::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return base::diff_weights_desc(1);
        }
    };

    /// Default constructor. Produces an empty object.
    deconvolution_backward_weights() = default;

    /// Constructs a deconvolution weights gradient primitive.
    /// @param pd Primitive descriptor for a deconvolution weights gradient
    ///     primitive.
    deconvolution_backward_weights(const primitive_desc &pd) : primitive(pd) {}
};

/// @} dnnl_api_deconvolution

/// @addtogroup dnnl_api_lrn LRN
///
/// A primitive to perform local response normalization (LRN) across or within
/// channels.
///
/// @sa @ref dev_guide_lrn in developer guide
///
/// @{

/// Local response normalization (LRN) forward propagation primitive.
struct lrn_forward : public primitive {
    /// Descriptor for an LRN forward propagation primitive.
    struct desc {
        dnnl_lrn_desc_t data;

        /// Constructs a descriptor for a LRN forward propagation primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm LRN algorithm kind: either
        ///     #dnnl::algorithm::lrn_across_channels, or
        ///     #dnnl::algorithm::lrn_within_channel.
        /// @param data_desc Source and destination memory descriptors.
        /// @param local_size Regularization local size.
        /// @param alpha The alpha regularization parameter.
        /// @param beta The beta regularization parameter.
        /// @param k The k regularization parameter.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &data_desc, memory::dim local_size,
                float alpha, float beta, float k = 1.f) {
            error::wrap_c_api(dnnl_lrn_forward_desc_init(&data,
                                      dnnl::convert_to_c(aprop_kind),
                                      convert_to_c(aalgorithm), &data_desc.data,
                                      local_size, alpha, beta, k),
                    "could not create a descriptor for a lrn forward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for an LRN forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LRN forward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for an LRN forward propagation primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an LRN forward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for an LRN forward propagation primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an LRN forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for an LRN forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::lrn,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }
    };

    /// Default constructor. Produces an empty object.
    lrn_forward() = default;

    /// Constructs an LRN forward propagation primitive.
    /// @param pd Primitive descriptor for an LRN forward propagation
    ///     primitive.
    lrn_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Local response normalization (LRN) backward propagation primitive.
struct lrn_backward : public primitive {
    /// Descriptor for an LRN backward propagation primitive.
    struct desc {
        dnnl_lrn_desc_t data;

        /// Constructs a descriptor for an LRN backward propagation primitive.
        ///
        /// @param aalgorithm LRN algorithm kind: either
        ///     #dnnl::algorithm::lrn_across_channels, or
        ///     #dnnl::algorithm::lrn_within_channel.
        /// @param diff_data_desc Diff source and diff destination memory
        ///     descriptor.
        /// @param data_desc Source memory descriptor.
        /// @param local_size Regularization local size.
        /// @param alpha The alpha regularization parameter.
        /// @param beta The beta regularization parameter.
        /// @param k The k regularization parameter.
        desc(algorithm aalgorithm, const memory::desc &data_desc,
                const memory::desc &diff_data_desc, memory::dim local_size,
                float alpha, float beta, float k = 1.f) {
            error::wrap_c_api(
                    dnnl_lrn_backward_desc_init(&data, convert_to_c(aalgorithm),
                            &diff_data_desc.data, &data_desc.data, local_size,
                            alpha, beta, k),
                    "could not create a descriptor for a lrn backward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for an LRN backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LRN backward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for an LRN backward propagation primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an LRN forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const lrn_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, nullptr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an LRN backward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for an LRN backward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an LRN forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine,
                const lrn_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an LRN backward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for an LRN backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::lrn,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }
    };

    /// Default constructor. Produces an empty object.
    lrn_backward() = default;

    /// Constructs an LRN backward propagation primitive.
    /// @param pd Primitive descriptor for an LRN backward propagation
    ///     primitive.
    lrn_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @} dnnl_api_lrn

/// @addtogroup dnnl_api_pooling Pooling
///
/// A primitive to perform max or average pooling.
///
/// @sa @ref dev_guide_pooling in developer guide
///
/// @{

/// Pooling forward propagation primitive.
struct pooling_forward : public primitive {
    /// Descriptor for a pooling forward propagation primitive.
    struct desc {
        dnnl_pooling_desc_t data;

        /// Constructs a descriptor for pooling forward propagation primitive.
        ///
        /// Arrays @p strides, @p kernel, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Pooling algorithm kind: either
        ///     #dnnl::algorithm::pooling_max,
        ///     #dnnl::algorithm::pooling_avg_include_padding,
        ///     or #dnnl::algorithm::pooling_avg (same as
        ///     #dnnl::algorithm::pooling_avg_exclude_padding).
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param kernel Vector of kernel spatial dimensions.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc, const memory::desc &dst_desc,
                const memory::dims &strides, const memory::dims &kernel,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(kernel, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(dnnl_pooling_forward_desc_init(&data,
                                      dnnl::convert_to_c(aprop_kind),
                                      convert_to_c(aalgorithm), &src_desc.data,
                                      &dst_desc.data, &strides[0], &kernel[0],
                                      &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a pooling forward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for a pooling forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a pooling forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a pooling forward propagation primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a pooling forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a pooling forward propagation primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a pooling forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a pooling forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::pooling,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }
    };

    /// Default constructor. Produces an empty object.
    pooling_forward() = default;

    /// Constructs a pooling forward propagation primitive.
    /// @param pd Primitive descriptor for a pooling forward propagation
    ///     primitive.
    pooling_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Pooling backward propagation primitive.
struct pooling_backward : public primitive {
    /// Descriptor for a pooling backward propagation primitive.
    struct desc {
        dnnl_pooling_desc_t data;

        /// Constructs a descriptor for pooling backward propagation primitive.
        ///
        /// Arrays @p strides, @p kernel, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aalgorithm Pooling algorithm kind: either
        ///     #dnnl::algorithm::pooling_max,
        ///     #dnnl::algorithm::pooling_avg_include_padding,
        ///     or #dnnl::algorithm::pooling_avg (same as
        ///     #dnnl::algorithm::pooling_avg_exclude_padding).
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param kernel Vector of kernel spatial dimensions.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &kernel, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides, diff_src_desc.data.ndims - 2);
            memory::validate_dims(kernel, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, diff_src_desc.data.ndims - 2);
            error::wrap_c_api(
                    dnnl_pooling_backward_desc_init(&data,
                            convert_to_c(aalgorithm), &diff_src_desc.data,
                            &diff_dst_desc.data, &strides[0], &kernel[0],
                            &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a pooling backward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for a pooling backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a pooling backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a pooling backward propagation primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a pooling forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const pooling_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, nullptr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a pooling backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a pooling backward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a pooling forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine,
                const pooling_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a pooling backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a pooling backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::pooling,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }
    };

    /// Default constructor. Produces an empty object.
    pooling_backward() = default;

    /// Constructs a pooling backward propagation primitive.
    /// @param pd Primitive descriptor for a pooling backward propagation
    ///     primitive.
    pooling_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @} dnnl_api_pooling

/// @addtogroup dnnl_api_eltwise Eltwise
///
/// A primitive to perform elementwise operations such as the
/// rectifier linear unit (ReLU).
///
/// Both forward and backward propagation primitives support in-place
/// operation; that is, src and dst can refer to the same memory for forward
/// propagation, and diff_dst and diff_src can refer to the same memory for
/// backward propagation.
///
/// @warning
///     Because the original source data is required for backward propagation,
///     in-place forward propagation is not generally supported in the
///     training mode. However, for algorithms supporting destination as input
///     memory, dst can be used for the backward propagation, which makes it
///     possible to get performance benefit even in the training mode.
///
/// @sa @ref dev_guide_eltwise in developer guide
///
/// @{

/// Elementwise unary operation forward propagation primitive.
struct eltwise_forward : public primitive {
    /// Descriptor for an elementwise forward propagation primitive.
    struct desc {
        dnnl_eltwise_desc_t data;

        /// Constructs a descriptor for an elementwise forward propagation
        /// primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Elementwise algorithm kind.
        /// @param data_desc Source and destination memory descriptors.
        /// @param alpha The alpha parameter for the elementwise operation.
        ///     Specific meaning depends on the algorithm.
        /// @param beta The beta parameter for the elementwise operation.
        ///     Specific meaning depends on the algorithm.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &data_desc, float alpha = 0,
                float beta = 0) {
            error::wrap_c_api(dnnl_eltwise_forward_desc_init(&data,
                                      dnnl::convert_to_c(aprop_kind),
                                      dnnl::convert_to_c(aalgorithm),
                                      &data_desc.data, alpha, beta),
                    "could not create a descriptor for an eltwise forward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for an elementwise forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an elementwise forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an elementwise forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an elementwise forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an elementwise forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an eltwise forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for an eltwise forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::eltwise,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    eltwise_forward() = default;

    /// Constructs an eltwise forward propagation primitive.
    /// @param pd Primitive descriptor for an eltwise forward propagation
    ///     primitive.
    eltwise_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Elementwise unary operation backward propagation primitive.
struct eltwise_backward : public primitive {
    /// Descriptor for an elementwise backward propagation primitive.
    struct desc {
        dnnl_eltwise_desc_t data;

        /// Constructs a descriptor for an elementwise backward propagation
        /// primitive.
        ///
        /// @param aalgorithm Elementwise algorithm kind.
        /// @param diff_data_desc Diff source and destination memory
        ///     descriptors.
        /// @param data_desc Source memory descriptor.
        /// @param alpha The alpha parameter for the elementwise operation.
        ///     Specific meaning depends on the algorithm.
        /// @param beta The beta parameter for the elementwise operation.
        ///     Specific meaning depends on the algorithm.
        desc(algorithm aalgorithm, const memory::desc &diff_data_desc,
                const memory::desc &data_desc, float alpha = 0,
                float beta = 0) {
            error::wrap_c_api(
                    dnnl_eltwise_backward_desc_init(&data,
                            dnnl::convert_to_c(aalgorithm),
                            &diff_data_desc.data, &data_desc.data, alpha, beta),
                    "could not create a descriptor for an eltwise backward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for eltwise backward propagation.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an elementwise backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an elementwise backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an elementwise forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const eltwise_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, nullptr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an elementwise backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an elementwise backward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an elementwise forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine,
                const eltwise_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an eltwise backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for an eltwise backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::eltwise,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    eltwise_backward() = default;

    /// Constructs an eltwise backward propagation primitive.
    /// @param pd Primitive descriptor for an eltwise backward propagation
    ///     primitive.
    eltwise_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @} dnnl_api_eltwise

/// @addtogroup dnnl_api_softmax Softmax
///
/// A primitive to perform softmax.
///
/// @sa @ref dev_guide_softmax in developer guide
///
/// @{

/// Softmax forward propagation primitive.
struct softmax_forward : public primitive {
    /// Descriptor for a softmax forward propagation primitive.
    struct desc {
        dnnl_softmax_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Constructs a descriptor for a softmax forward propagation
        /// primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param data_desc Source and destination memory descriptor.
        /// @param softmax_axis Axis over which softmax is computed.
        desc(prop_kind aprop_kind, const memory::desc &data_desc,
                int softmax_axis) {
            error::wrap_c_api(dnnl_softmax_forward_desc_init(&data,
                                      dnnl::convert_to_c(aprop_kind),
                                      &data_desc.data, softmax_axis),
                    "could not create a descriptor for a softmax forward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for a softmax forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a softmax forward
        /// propagation primitive.
        ///
        /// @param adesc descriptor for a softmax forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a softmax forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a softmax forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a softmax forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a softmax forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::softmax,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    softmax_forward() = default;

    /// Constructs a softmax forward propagation primitive.
    /// @param pd Primitive descriptor for a softmax forward propagation
    ///     primitive.
    softmax_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Softmax backward propagation primitive.
struct softmax_backward : public primitive {
    /// Descriptor for a softmax backward propagation primitive.
    struct desc {
        dnnl_softmax_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Constructs a descriptor for a softmax backward propagation
        /// primitive.
        ///
        /// @param diff_data_desc Diff source and diff destination memory
        ///     descriptor.
        /// @param data_desc Destination memory descriptor.
        /// @param softmax_axis Axis over which softmax is computed.
        desc(const memory::desc &diff_data_desc, const memory::desc &data_desc,
                int softmax_axis) {
            error::wrap_c_api(
                    dnnl_softmax_backward_desc_init(&data, &diff_data_desc.data,
                            &data_desc.data, softmax_axis),
                    "could not create a descriptor for a softmax backward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for a softmax backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a softmax backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a softmax backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a softmax forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const softmax_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, nullptr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a softmax backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a softmax backward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a softmax forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine,
                const softmax_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a softmax backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a softmax backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::softmax,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    softmax_backward() = default;

    /// Constructs a softmax backward propagation primitive.
    /// @param pd Primitive descriptor for a softmax backward propagation
    ///     primitive.
    softmax_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @} dnnl_api_softmax

/// @addtogroup dnnl_api_logsoftmax LogSoftmax
///
/// A primitive to perform logsoftmax.
///
/// @sa @ref dev_guide_logsoftmax in developer guide
///
/// @{

/// Logsoftmax forward propagation primitive.
struct logsoftmax_forward : public primitive {
    /// Descriptor for a logsoftmax forward propagation primitive.
    struct desc {
        dnnl_logsoftmax_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Constructs a descriptor for a logsoftmax forward propagation
        /// primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param data_desc Source and destination memory descriptor.
        /// @param logsoftmax_axis Axis over which softmax is computed.
        desc(prop_kind aprop_kind, const memory::desc &data_desc,
                int logsoftmax_axis) {
            error::wrap_c_api(dnnl_logsoftmax_forward_desc_init(&data,
                                      dnnl::convert_to_c(aprop_kind),
                                      &data_desc.data, logsoftmax_axis),
                    "could not create a descriptor for a logsoftmax forward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for a logsoftmax forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a logsoftmax forward
        /// propagation primitive.
        ///
        /// @param adesc descriptor for a logsoftmax forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a logsoftmax forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a logsoftmax forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a logsoftmax forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a logsoftmax forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd,
                    // Logsoftmax and softmax share the implementation and
                    // currently report the same primitive kind. Hence this
                    // must be softmax and not logsoftmax.
                    dnnl::primitive::kind::softmax,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    logsoftmax_forward() = default;

    /// Constructs a logsoftmax forward propagation primitive.
    /// @param pd Primitive descriptor for a logsoftmax forward propagation
    ///     primitive.
    logsoftmax_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Logsoftmax backward propagation primitive.
struct logsoftmax_backward : public primitive {
    /// Descriptor for a logsoftmax backward propagation primitive.
    struct desc {
        dnnl_logsoftmax_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Constructs a descriptor for a logsoftmax backward propagation
        /// primitive.
        ///
        /// @param diff_data_desc Diff source and diff destination memory
        ///     descriptors.
        /// @param data_desc Destination memory descriptor.
        /// @param logsoftmax_axis Axis over which softmax is computed.
        desc(const memory::desc &diff_data_desc, const memory::desc &data_desc,
                int logsoftmax_axis) {
            error::wrap_c_api(dnnl_logsoftmax_backward_desc_init(&data,
                                      &diff_data_desc.data, &data_desc.data,
                                      logsoftmax_axis),
                    "could not create a descriptor for a logsoftmax backward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for a logsoftmax backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a logsoftmax backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a logsoftmax backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a logsoftmax forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const logsoftmax_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, nullptr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a logsoftmax backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a logsoftmax backward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a logsoftmax forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine,
                const logsoftmax_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a logsoftmax backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a logsoftmax backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd,
                    // Logsoftmax and softmax share the implementation and
                    // currently report the same primitive kind. Hence this
                    // must be softmax and not logsoftmax.
                    dnnl::primitive::kind::softmax,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    logsoftmax_backward() = default;

    /// Constructs a logsoftmax backward propagation primitive.
    /// @param pd Primitive descriptor for a logsoftmax backward propagation
    ///     primitive.
    logsoftmax_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @} dnnl_api_logsoftmax

/// @addtogroup dnnl_api_batch_normalization Batch Normalization
///
/// A primitive to perform batch normalization.
///
/// Both forward and backward propagation primitives support in-place
/// operation; that is, src and dst can refer to the same memory for forward
/// propagation, and diff_dst and diff_src can refer to the same memory for
/// backward propagation.
///
/// The batch normalization primitives computations can be controlled by
/// specifying different @ref dnnl::normalization_flags values. For example,
/// batch normalization can compute the mean and variance on its own or take
/// them as inputs.  It can either perform scaling and shifting using gamma
/// and beta parameters or not. Optionally, it can also perform a fused ReLU,
/// which in case of training would also require a workspace.
///
/// @sa @ref dev_guide_batch_normalization in developer guide
///
/// @{

/// Batch normalization forward propagation primitive.
struct batch_normalization_forward : public primitive {
    /// Descriptor for a batch normalization forward propagation primitive.
    struct desc {
        dnnl_batch_normalization_desc_t data;

        /// Constructs a batch normalization descriptor for forward
        /// propagation.
        ///
        /// @note
        ///     In-place operation is supported: the dst can refer to the same
        ///     memory as the src.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param data_desc Source and destination memory descriptors.
        /// @param epsilon Batch normalization epsilon parameter.
        /// @param flags Batch normalization flags (@ref
        ///     dnnl::normalization_flags).
        desc(prop_kind aprop_kind, const memory::desc &data_desc, float epsilon,
                normalization_flags flags) {
            error::wrap_c_api(
                    dnnl_batch_normalization_forward_desc_init(&data,
                            dnnl::convert_to_c(aprop_kind), &data_desc.data,
                            epsilon, convert_to_c(flags)),
                    "could not create a descriptor for a batch normalization "
                    "forward propagation primitive");
        }
    };

    /// Primitive descriptor for a batch normalization forward propagation
    /// primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a batch normalization forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a batch normalization forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a batch normalization forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a batch normalization forward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a batch normalization
        /// forward propagation primitive from a C API primitive descriptor
        /// that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a batch normalization
        ///     forward propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd,
                    dnnl::primitive::kind::batch_normalization,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }

        /// Returns memory descriptor for mean.
        /// @returns Memory descriptor for mean.
        memory::desc mean_desc() const { return stat_desc(mean); }

        /// Returns memory descriptor for variance.
        /// @returns Memory descriptor for variance.
        memory::desc variance_desc() const { return stat_desc(var); }

    private:
        enum {
            mean = 1,
            var = 2,
        };
        memory::desc stat_desc(int kind) const {
            dnnl_batch_normalization_desc_t *p;
            error::wrap_c_api(
                    dnnl_primitive_desc_query(get(),
                            dnnl::convert_to_c(query::batch_normalization_d), 0,
                            &p),
                    "could not retrieve a descriptor from a primitive "
                    "descriptor for batch normalization forward propagation "
                    "primitive");
            return query_md(p->flags & dnnl_use_global_stats ? query::src_md
                                                             : query::dst_md,
                    kind);
        }
    };

    /// Default constructor. Produces an empty object.
    batch_normalization_forward() = default;

    /// Constructs a batch normalization forward propagation primitive.
    /// @param pd Primitive descriptor for a batch normalization forward
    ///     propagation primitive.
    batch_normalization_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Batch normalization backward propagation primitive.
struct batch_normalization_backward : public primitive {
    /// Descriptor for a batch normalization backward propagation primitive.
    struct desc {
        dnnl_batch_normalization_desc_t data;

        /// Constructs a batch normalization descriptor for backward
        /// propagation.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::backward_data and #dnnl::prop_kind::backward
        ///     (diffs for all parameters are computed in this case).
        /// @param diff_data_desc Diff source and diff destination memory
        ///     descriptor.
        /// @param data_desc Source memory descriptor.
        /// @param epsilon Batch normalization epsilon parameter.
        /// @param flags Batch normalization flags (@ref
        ///     dnnl::normalization_flags).
        desc(prop_kind aprop_kind, const memory::desc &diff_data_desc,
                const memory::desc &data_desc, float epsilon,
                normalization_flags flags) {
            error::wrap_c_api(dnnl_batch_normalization_backward_desc_init(&data,
                                      dnnl::convert_to_c(aprop_kind),
                                      &diff_data_desc.data, &data_desc.data,
                                      epsilon, convert_to_c(flags)),
                    "could not create a descriptor for a batch normalization "
                    "backward propagation primitive");
        }
    };

    /// Primitive descriptor for a batch normalization backward propagation
    /// primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a batch normalization backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a batch normalization backward
        ///     propagation primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a batch normalization
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const batch_normalization_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, nullptr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a batch normalization backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a batch normalization backward
        ///     propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a batch normalization
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine,
                const batch_normalization_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a batch normalization
        /// backward propagation primitive from a C API primitive descriptor
        /// that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a batch normalization
        ///     backward propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd,
                    dnnl::primitive::kind::batch_normalization,
                    dnnl::prop_kind::backward, dnnl::prop_kind::backward_data) {
        }

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_weights_desc()const
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc dnnl::batch_normalization_forward::primitive_desc::mean_desc()const
        memory::desc mean_desc() const { return query_md(query::src_md, 1); }

        /// @copydoc dnnl::batch_normalization_forward::primitive_desc::variance_desc()const
        memory::desc variance_desc() const {
            return query_md(query::src_md, 2);
        }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }
    };

    /// Default constructor. Produces an empty object.
    batch_normalization_backward() = default;

    /// Constructs a batch normalization backward propagation primitive.
    /// @param pd Primitive descriptor for a batch normalization backward
    ///     propagation primitive.
    batch_normalization_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @} dnnl_api_batch_normalization

/// @addtogroup dnnl_api_layer_normalization Layer Normalization
///
/// A primitive to perform layer normalization. Normalization is performed
/// within the last logical dimension of data tensor.
///
/// Both forward and backward propagation primitives support in-place
/// operation; that is, src and dst can refer to the same memory for forward
/// propagation, and diff_dst and diff_src can refer to the same memory for
/// backward propagation.
///
/// The layer normalization primitives computations can be controlled by
/// specifying different dnnl::normalization_flags values. For example,
/// layer normalization forward propagation can be configured to either
/// compute the mean and variance or take them as arguments. It can either
/// perform scaling and shifting using gamma and beta parameters or not.
/// Optionally, it can also perform a fused ReLU, which in case of training
/// would also require a workspace.
///
/// @sa @ref dev_guide_layer_normalization in developer guide
///
/// @{

/// Layer normalization forward propagation primitive.
struct layer_normalization_forward : public primitive {
    /// Descriptor for a layer normalization forward propagation primitive.
    struct desc {
        dnnl_layer_normalization_desc_t data;

        /// Constructs a descriptor for layer normalization forward
        /// propagation primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param data_desc Source and destination memory descriptor.
        /// @param stat_desc Statistics memory descriptors.
        /// @param epsilon Layer normalization epsilon parameter.
        /// @param flags Layer normalization flags (@ref
        ///     dnnl::normalization_flags).
        desc(prop_kind aprop_kind, const memory::desc &data_desc,
                const memory::desc &stat_desc, float epsilon,
                normalization_flags flags) {
            error::wrap_c_api(
                    dnnl_layer_normalization_forward_desc_init(&data,
                            dnnl::convert_to_c(aprop_kind), &data_desc.data,
                            &stat_desc.data, epsilon, convert_to_c(flags)),
                    "could not create a descriptor for a layer normalization "
                    "forward propagation primitive");
        }

        /// Constructs a descriptor for layer normalization forward
        /// propagation primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param data_desc Source and destination memory descriptor.
        /// @param epsilon Layer normalization epsilon parameter.
        /// @param flags Layer normalization flags (@ref
        ///     dnnl::normalization_flags).
        desc(prop_kind aprop_kind, const memory::desc &data_desc, float epsilon,
                normalization_flags flags) {
            error::wrap_c_api(
                    dnnl_layer_normalization_forward_desc_init(&data,
                            dnnl::convert_to_c(aprop_kind), &data_desc.data,
                            nullptr, epsilon, convert_to_c(flags)),
                    "could not create a descriptor for a layer normalization "
                    "forward propagation primitive");
        }
    };

    /// Primitive descriptor for a layer normalization forward propagation
    /// primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a layer normalization forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a layer normalization forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a layer normalization forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a layer normalization forward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a layer normalization
        /// forward propagation primitive from a C API primitive descriptor
        /// that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a layer normalization
        ///     forward propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd,
                    dnnl::primitive::kind::layer_normalization,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }

        /// @copydoc dnnl::batch_normalization_forward::primitive_desc::mean_desc()const
        memory::desc mean_desc() const { return stat_desc(mean); }

        /// @copydoc dnnl::batch_normalization_forward::primitive_desc::variance_desc()const
        memory::desc variance_desc() const { return stat_desc(var); }

    private:
        enum {
            mean = 1,
            var = 2,
        };
        memory::desc stat_desc(int kind) const {
            dnnl_layer_normalization_desc_t *p;
            error::wrap_c_api(
                    dnnl_primitive_desc_query(get(),
                            dnnl::convert_to_c(query::layer_normalization_d), 0,
                            &p),
                    "could not retrieve a descriptor from a primitive "
                    "descriptor for layer normalization forward propagation "
                    "primitive");
            return query_md(p->flags & dnnl_use_global_stats ? query::src_md
                                                             : query::dst_md,
                    kind);
        }
    };

    /// Default constructor. Produces an empty object.
    layer_normalization_forward() = default;

    /// Constructs a layer normalization forward propagation primitive.
    /// @param pd Primitive descriptor for a layer normalization forward
    ///     propagation primitive.
    layer_normalization_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Layer normalization backward propagation primitive.
struct layer_normalization_backward : public primitive {
    /// Descriptor for a layer normalization backward propagation primitive.
    struct desc {
        dnnl_layer_normalization_desc_t data;

        /// Constructs a descriptor for layer normalization backward
        /// propagation primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::backward_data and #dnnl::prop_kind::backward
        ///     (diffs for all parameters are computed in this case).
        /// @param diff_data_desc Diff source and diff destination memory
        ///     descriptor.
        /// @param data_desc Source memory descriptor.
        /// @param stat_desc Statistics memory descriptors.
        /// @param epsilon Layer normalization epsilon parameter.
        /// @param flags Layer normalization flags (@ref
        ///     dnnl::normalization_flags).
        desc(prop_kind aprop_kind, const memory::desc &diff_data_desc,
                const memory::desc &data_desc, const memory::desc &stat_desc,
                float epsilon, normalization_flags flags) {
            error::wrap_c_api(
                    dnnl_layer_normalization_backward_desc_init(&data,
                            dnnl::convert_to_c(aprop_kind),
                            &diff_data_desc.data, &data_desc.data,
                            &stat_desc.data, epsilon, convert_to_c(flags)),
                    "could not create a descriptor for a batch normalization "
                    "backward propagation primitive");
        }

        /// Constructs a descriptor for layer normalization backward
        /// propagation primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::backward_data and #dnnl::prop_kind::backward
        ///     (diffs for all parameters are computed in this case).
        /// @param diff_data_desc Diff source and diff destination memory
        ///     descriptor.
        /// @param data_desc Source memory descriptor.
        /// @param epsilon Layer normalization epsilon parameter.
        /// @param flags Layer normalization flags (@ref
        ///     dnnl::normalization_flags).
        desc(prop_kind aprop_kind, const memory::desc &diff_data_desc,
                const memory::desc &data_desc, float epsilon,
                normalization_flags flags) {
            error::wrap_c_api(dnnl_layer_normalization_backward_desc_init(&data,
                                      dnnl::convert_to_c(aprop_kind),
                                      &diff_data_desc.data, &data_desc.data,
                                      nullptr, epsilon, convert_to_c(flags)),
                    "could not create a descriptor for a batch normalization "
                    "backward propagation primitive");
        }
    };

    /// Primitive descriptor for a layer normalization backward propagation
    /// primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a layer normalization backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a layer normalization backward
        ///     propagation primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a layer normalization
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const layer_normalization_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, nullptr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a layer normalization backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a layer normalization backward
        ///     propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a layer normalization
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine,
                const layer_normalization_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a layer normalization
        /// backward propagation primitive from a C API primitive descriptor
        /// that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a layer normalization
        ///     backward propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd,
                    dnnl::primitive::kind::layer_normalization,
                    dnnl::prop_kind::backward, dnnl::prop_kind::backward_data) {
        }

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_weights_desc()const
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc dnnl::batch_normalization_forward::primitive_desc::mean_desc()const
        memory::desc mean_desc() const { return query_md(query::src_md, 1); }

        /// @copydoc dnnl::batch_normalization_forward::primitive_desc::variance_desc()const
        memory::desc variance_desc() const {
            return query_md(query::src_md, 2);
        }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }
    };

    /// Default constructor. Produces an empty object.
    layer_normalization_backward() = default;

    /// Constructs a layer normalization backward propagation primitive.
    /// @param pd Primitive descriptor for a layer normalization backward
    ///     propagation primitive.
    layer_normalization_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @} dnnl_api_layer_normalization

/// @addtogroup dnnl_api_inner_product Inner Product
///
/// A primitive to compute an inner product.
///
/// @sa @ref dev_guide_inner_product in developer guide
///
/// @{

/// Inner product forward propagation primitive.
struct inner_product_forward : public primitive {
    /// Descriptor for an inner product forward propagation primitive.
    struct desc {
        dnnl_inner_product_desc_t data;

        /// Constructs a descriptor for an inner product forward propagation
        /// primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param src_desc Memory descriptor for src.
        /// @param weights_desc Memory descriptor for diff weights.
        /// @param bias_desc Memory descriptor for diff bias.
        /// @param dst_desc Memory descriptor for diff dst.
        desc(prop_kind aprop_kind, const memory::desc &src_desc,
                const memory::desc &weights_desc, const memory::desc &bias_desc,
                const memory::desc &dst_desc) {
            error::wrap_c_api(dnnl_inner_product_forward_desc_init(&data,
                                      dnnl::convert_to_c(aprop_kind),
                                      &src_desc.data, &weights_desc.data,
                                      &bias_desc.data, &dst_desc.data),
                    "could not create a descriptor for an inner product "
                    "forward propagation primitive");
        }

        /// Constructs a descriptor for an inner product forward propagation
        /// primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param src_desc Memory descriptor for src.
        /// @param weights_desc Memory descriptor for diff weights.
        /// @param dst_desc Memory descriptor for dst.
        desc(prop_kind aprop_kind, const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc) {
            error::wrap_c_api(
                    dnnl_inner_product_forward_desc_init(&data,
                            dnnl::convert_to_c(aprop_kind), &src_desc.data,
                            &weights_desc.data, nullptr, &dst_desc.data),
                    "could not create a descriptor for an inner product "
                    "forward propagation primitive");
        }
    };

    /// Primitive descriptor for an inner product forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an inner product forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an inner product forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an inner product forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an inner product forward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an inner product forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for an inner product forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::inner_product,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::convolution_forward::primitive_desc::bias_desc()const
        memory::desc bias_desc() const { return base::weights_desc(1); }
    };

    /// Default constructor. Produces an empty object.
    inner_product_forward() = default;

    /// Constructs an inner product forward propagation primitive.
    /// @param pd Primitive descriptor for an inner product forward
    ///     propagation primitive.
    inner_product_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Inner product backward propagation primitive.
struct inner_product_backward_data : public primitive {
    /// Descriptor for an inner product backward propagation primitive.
    struct desc {
        dnnl_inner_product_desc_t data;

        /// Constructs a descriptor for an inner product backward propagation
        /// primitive.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param diff_src_desc Memory descriptor for diff src.
        /// @param weights_desc Memory descriptor for weights.
        /// @param diff_dst_desc Memory descriptor for diff dst.
        desc(const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc) {
            error::wrap_c_api(dnnl_inner_product_backward_data_desc_init(&data,
                                      &diff_src_desc.data, &weights_desc.data,
                                      &diff_dst_desc.data),
                    "could not create a descriptor for an inner product "
                    "backward propagation primitive");
        }
    };

    /// Primitive descriptor for an inner product backward propagation
    /// primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an inner product backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an inner product backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an inner product
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const inner_product_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, nullptr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an inner product backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an inner product backward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an inner product
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine,
                const inner_product_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an inner product backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for an inner product backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::inner_product,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    inner_product_backward_data() = default;

    /// Constructs an inner product backward propagation primitive.
    /// @param pd Primitive descriptor for an inner product backward
    ///     propagation primitive.
    inner_product_backward_data(const primitive_desc &pd) : primitive(pd) {}
};

/// Inner product weights gradient primitive.
struct inner_product_backward_weights : public primitive {
    /// Descriptor for an inner product weights gradient primitive.
    struct desc {
        dnnl_inner_product_desc_t data;

        /// Constructs a descriptor for an inner product descriptor weights
        /// update primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param src_desc Memory descriptor for src.
        /// @param diff_weights_desc Memory descriptor for diff weights.
        /// @param diff_bias_desc Memory descriptor for diff bias.
        /// @param diff_dst_desc Memory descriptor for diff dst.
        desc(const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc) {
            error::wrap_c_api(
                    dnnl_inner_product_backward_weights_desc_init(&data,
                            &src_desc.data, &diff_weights_desc.data,
                            &diff_bias_desc.data, &diff_dst_desc.data),
                    "could not create a descriptor for an inner product "
                    "weights gradient primitive");
        }

        /// Constructs a descriptor for an inner product descriptor weights
        /// update primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param src_desc Memory descriptor for src.
        /// @param diff_weights_desc Memory descriptor for diff weights.
        /// @param diff_dst_desc Memory descriptor for diff dst.
        desc(const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc) {
            error::wrap_c_api(
                    dnnl_inner_product_backward_weights_desc_init(&data,
                            &src_desc.data, &diff_weights_desc.data, nullptr,
                            &diff_dst_desc.data),
                    "could not create a descriptor for an inner product "
                    "weights gradient primitive");
        }
    };

    /// Primitive descriptor for an inner product weights gradient primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an inner product weights
        /// update primitive.
        ///
        /// @param adesc Descriptor for an inner product weights gradient
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an inner product
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const inner_product_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, nullptr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an inner product weights
        /// update primitive.
        ///
        /// @param adesc Descriptor for an inner product weights gradient
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an inner product
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine,
                const inner_product_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an inner product weights
        /// update primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for an inner product weights
        ///     gradient primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::inner_product,
                    dnnl::prop_kind::backward_weights) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_weights_desc()const
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::convolution_backward_weights::primitive_desc::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return base::diff_weights_desc(1);
        }
    };

    /// Default constructor. Produces an empty object.
    inner_product_backward_weights() = default;

    /// Constructs an inner product weights gradient primitive.
    /// @param pd Primitive descriptor for an inner product weights gradient
    ///     primitive.
    inner_product_backward_weights(const primitive_desc &pd) : primitive(pd) {}
};

/// @} dnnl_api_inner_product

/// @addtogroup dnnl_api_rnn RNN
///
/// A primitive to compute recurrent neural network layers.
///
/// @sa @ref dev_guide_rnn in developer guide
///
/// @{

/// Base class for primitive descriptors for RNN primitives.
struct rnn_primitive_desc_base : public primitive_desc {
    using primitive_desc::primitive_desc;

    /// Default constructor. Produces an empty object.
    rnn_primitive_desc_base() = default;

    /// Constructs an RNN primitive descriptor base from a C API primitive
    /// descriptor while checking that it actually describes the expected
    /// primitive by comparing propagation and primitive kinds.
    ///
    /// @param pd C API primitive descriptor.
    /// @param aprop_kind Expected propagation kind.
    /// @param cell_kind Expected cell kind.
    rnn_primitive_desc_base(dnnl_primitive_desc_t pd,
            dnnl::prop_kind aprop_kind, dnnl::algorithm cell_kind)
        : rnn_primitive_desc_base(pd, aprop_kind, aprop_kind, cell_kind) {}

    /// Returns source layer memory descriptor.
    /// @returns Source layer memory descriptor.
    memory::desc src_layer_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_SRC_LAYER);
    }

    /// Returns source iteration memory descriptor.
    /// @returns Source iteration memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///          source iteration parameter.
    memory::desc src_iter_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_SRC_ITER);
    }

    /// Returns source recurrent cell state memory descriptor.
    /// @returns Source recurrent cell state memory descriptor.
    memory::desc src_iter_c_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_SRC_ITER_C);
    }

    /// Returns weights layer memory descriptor.
    /// @returns Weights layer memory descriptor.
    memory::desc weights_layer_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_WEIGHTS_LAYER);
    }

    /// Returns weights iteration memory descriptor.
    /// @returns Weights iteration memory descriptor.
    memory::desc weights_iter_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_WEIGHTS_ITER);
    }

    /// Returns weights peephole memory descriptor.
    /// @returns Weights peephole memory descriptor.
    memory::desc weights_peephole_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_WEIGHTS_PEEPHOLE);
    }

    /// Returns weights projection memory descriptor.
    /// @returns Weights projection memory descriptor.
    memory::desc weights_projection_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_WEIGHTS_PROJECTION);
    }

    /// Returns bias memory descriptor.
    /// @returns Bias memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///          bias parameter.
    memory::desc bias_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_BIAS);
    }

    /// Returns destination layer memory descriptor.
    /// @returns Destination layer memory descriptor.
    memory::desc dst_layer_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DST_LAYER);
    }

    /// Returns destination iteration memory descriptor.
    /// @returns Destination iteration memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///          destination iteration parameter.
    memory::desc dst_iter_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DST_ITER);
    }

    /// Returns destination recurrent cell state memory descriptor.
    /// @returns Destination recurrent cell state memory descriptor.
    memory::desc dst_iter_c_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DST_ITER_C);
    }

    /// Returns diff source layer memory descriptor.
    /// @returns Diff source layer memory descriptor.
    memory::desc diff_src_layer_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DIFF_SRC_LAYER);
    }

    /// Returns diff source iteration memory descriptor.
    /// @returns Diff source iteration memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///          diff source iteration parameter.
    memory::desc diff_src_iter_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DIFF_SRC_ITER);
    }

    /// Returns diff source recurrent cell state memory descriptor.
    /// @returns Diff source recurrent cell state memory descriptor.
    memory::desc diff_src_iter_c_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DIFF_SRC_ITER_C);
    }

    /// Returns diff weights layer memory descriptor.
    /// @returns Diff weights layer memory descriptor.
    memory::desc diff_weights_layer_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DIFF_WEIGHTS_LAYER);
    }

    /// Returns diff weights iteration memory descriptor.
    /// @returns Diff weights iteration memory descriptor.
    memory::desc diff_weights_iter_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DIFF_WEIGHTS_ITER);
    }

    /// Returns diff weights peephole memory descriptor.
    /// @returns Diff weights peephole memory descriptor.
    memory::desc diff_weights_peephole_desc() const {
        return base::query_md(
                query::exec_arg_md, DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE);
    }

    /// Returns diff weights projection memory descriptor.
    /// @returns Diff weights projection memory descriptor.
    memory::desc diff_weights_projection_desc() const {
        return base::query_md(
                query::exec_arg_md, DNNL_ARG_DIFF_WEIGHTS_PROJECTION);
    }

    /// Returns diff bias memory descriptor.
    /// @returns Diff bias memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///          diff bias parameter.
    memory::desc diff_bias_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DIFF_BIAS);
    }

    /// Returns diff destination layer memory descriptor.
    /// @returns Diff destination layer memory descriptor.
    memory::desc diff_dst_layer_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DIFF_DST_LAYER);
    }

    /// Returns diff destination iteration memory descriptor.
    /// @returns Diff destination iteration memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///          diff destination iteration parameter.
    memory::desc diff_dst_iter_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DIFF_DST_ITER);
    }

    /// Returns diff destination recurrent cell state memory descriptor.
    /// @returns Diff destination recurrent cell state memory descriptor.
    memory::desc diff_dst_iter_c_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DIFF_DST_ITER_C);
    }

protected:
    using rnn_base = rnn_primitive_desc_base;

    // (Deliberately not using doxygen comments)
    //
    // Constructs an RNN primitive descriptor base from a C API primitive
    // descriptor while checking that it actually describes the expected
    // primitive by comparing propagation and primitive kinds. Caller can
    // pass two options propagation kinds. This is typically used to check
    // that propagation kind is inference or training forward propagation.
    //
    // @param pd C API primitive descriptor.
    // @param prop_kind1 Expected propagation kind.
    // @param prop_kind2 Expected propagation kind.
    // @param cell_kind Expected cell kind.
    rnn_primitive_desc_base(dnnl_primitive_desc_t pd,
            dnnl::prop_kind prop_kind1, dnnl::prop_kind prop_kind2,
            dnnl::algorithm cell_kind) {
        dnnl_rnn_desc_t *rnn_d;
        dnnl_status_t rc;
        rc = dnnl_primitive_desc_query(pd, dnnl_query_rnn_d, 0, &rnn_d);
        error::wrap_c_api(rc,
                "could not retrieve a descriptor from a primitive descriptor "
                "for an RNN primitive");

        dnnl_prop_kind_t c_prop_kind1 = convert_to_c(prop_kind1);
        dnnl_prop_kind_t c_prop_kind2 = convert_to_c(prop_kind2);
        dnnl_alg_kind_t c_cell_kind = convert_to_c(cell_kind);

        bool ok = rnn_d->primitive_kind == dnnl_rnn
                && (rnn_d->prop_kind == c_prop_kind1
                        || rnn_d->prop_kind == c_prop_kind2)
                && rnn_d->cell_kind == c_cell_kind;

        if (!ok)
            DNNL_THROW_ERROR(dnnl_invalid_arguments,
                    "mismatch between expected and provided descriptors for an "
                    "RNN primitive");

        reset_with_clone(pd);
    }
};

/// Vanilla RNN forward propagation primitive.
struct vanilla_rnn_forward : public primitive {
    /// Descriptor for a vanilla RNN forward propagation primitive.
    struct desc {
        dnnl_rnn_desc_t data;

        /// Constructs a descriptor for a vanilla RNN forward propagation
        /// primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc.
        ///
        /// This would then indicate that the RNN forward propagation primitive
        /// should not use them and should default to zero values instead.
        ///
        /// @note
        ///     All memory descriptors except @p src_iter_desc can be
        ///     initialized with an #dnnl::memory::format_tag::any value of @p
        ///     format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param activation Activation kind. Possible values are
        ///     #dnnl::algorithm::eltwise_relu,
        ///     #dnnl::algorithm::eltwise_tanh, or
        ///     #dnnl::algorithm::eltwise_logistic.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param flags Unused.
        /// @param alpha Negative slope if activation is
        ///     #dnnl::algorithm::eltwise_relu.
        /// @param beta Unused.
        desc(prop_kind aprop_kind, algorithm activation,
                rnn_direction direction, const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                rnn_flags flags = rnn_flags::undef, float alpha = 0.0f,
                float beta = 0.0f) {
            error::wrap_c_api(
                    dnnl_vanilla_rnn_forward_desc_init(&data,
                            dnnl::convert_to_c(aprop_kind),
                            dnnl::convert_to_c(activation),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &weights_layer_desc.data,
                            &weights_iter_desc.data, &bias_desc.data,
                            &dst_layer_desc.data, &dst_iter_desc.data,
                            dnnl::convert_to_c(flags), alpha, beta),
                    "could not create a descriptor for a vanilla RNN forward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for a vanilla RNN forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a vanilla RNN forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a vanilla RNN forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                bool allow_empty = false)
            : rnn_primitive_desc_base(
                    &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a vanilla RNN forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a vanilla RNN forward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine, bool allow_empty = false)
            : rnn_primitive_desc_base(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a vanilla RNN forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a vanilla RNN forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference,
                    dnnl::algorithm::vanilla_rnn) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    vanilla_rnn_forward() = default;

    /// Constructs a vanilla RNN forward propagation primitive.
    /// @param pd Primitive descriptor for a vanilla RNN forward
    ///     propagation primitive.
    vanilla_rnn_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Vanilla RNN backward propagation primitive.
struct vanilla_rnn_backward : public primitive {
    /// Descriptor for a vanilla RNN backward propagation primitive.
    struct desc {
        dnnl_rnn_desc_t data;

        /// Constructs a descriptor for a vanilla RNN backward propagation
        /// primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p diff_src_iter_desc,
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p diff_dst_iter_desc.
        ///
        /// This would then indicate that the RNN backward propagation
        /// primitive should not use the respective data and should use zero
        /// values instead.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Must be
        ///     #dnnl::prop_kind::backward.
        /// @param activation Activation kind. Possible values are
        ///     #dnnl::algorithm::eltwise_relu,
        ///     #dnnl::algorithm::eltwise_tanh, or
        ///     #dnnl::algorithm::eltwise_logistic.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param flags Unused.
        /// @param alpha Negative slope if activation is
        ///     #dnnl::algorithm::eltwise_relu.
        /// @param beta Unused.
        desc(prop_kind aprop_kind, algorithm activation,
                rnn_direction direction, const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc,
                rnn_flags flags = rnn_flags::undef, float alpha = 0.0f,
                float beta = 0.0f) {
            error::wrap_c_api(
                    dnnl_vanilla_rnn_backward_desc_init(&data,
                            dnnl::convert_to_c(aprop_kind),
                            dnnl::convert_to_c(activation),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &weights_layer_desc.data,
                            &weights_iter_desc.data, &bias_desc.data,
                            &dst_layer_desc.data, &dst_iter_desc.data,
                            &diff_src_layer_desc.data, &diff_src_iter_desc.data,
                            &diff_weights_layer_desc.data,
                            &diff_weights_iter_desc.data, &diff_bias_desc.data,
                            &diff_dst_layer_desc.data, &diff_dst_iter_desc.data,
                            dnnl::convert_to_c(flags), alpha, beta),
                    "could not create a descriptor for a vanilla RNN backward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for an RNN backward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a vanilla RNN backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a vanilla RNN backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a vanilla RNN
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const vanilla_rnn_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, nullptr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a vanilla RNN backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a vanilla RNN backward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a vanilla RNN
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine,
                const vanilla_rnn_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a vanilla RNN backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a vanilla RNN backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::backward,
                    dnnl::algorithm::vanilla_rnn) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_layer_desc()const
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_iter_desc()const
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_layer_desc()const
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_iter_desc()const
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_layer_desc()const
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_iter_desc()const
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    vanilla_rnn_backward() = default;

    /// Constructs a vanilla RNN backward propagation primitive.
    /// @param pd Primitive descriptor for a vanilla RNN backward
    ///     propagation primitive.
    vanilla_rnn_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// LSTM forward propagation primitive.
struct lstm_forward : public primitive {
    /// Descriptor for an LSTM forward propagation primitive.
    struct desc {
        dnnl_rnn_desc_t data;

        /// Constructs a descriptor for an LSTM (with or without peephole and
        /// with or without projection) forward propagation primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p src_iter_c_desc,
        /// - @p weights_peephole_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc together with @p dst_iter_c_desc.
        ///
        /// This would then indicate that the LSTM forward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// The @p weights_projection_desc may point to a zero memory
        /// descriptor. This would then indicate that the LSTM doesn't have
        /// recurrent projection layer.
        ///
        /// @note
        ///     All memory descriptors can be initialized with an
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param src_iter_c_desc Memory descriptor for the input recurrent
        ///     cell state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param weights_peephole_desc Memory descriptor for the weights
        ///     applied to the cell states (according to the Peephole LSTM
        ///     formula).
        /// @param weights_projection_desc Memory descriptor for the weights
        ///     applied to the hidden states to get the recurrent projection
        ///     (according to the Projection LSTM formula).
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param dst_iter_c_desc Memory descriptor for the output recurrent
        ///     cell state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &src_iter_c_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &weights_peephole_desc,
                const memory::desc &weights_projection_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &dst_iter_c_desc,
                rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                    dnnl_lstm_forward_desc_init_v3(&data,
                            dnnl::convert_to_c(aprop_kind),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &src_iter_c_desc.data,
                            &weights_layer_desc.data, &weights_iter_desc.data,
                            &weights_peephole_desc.data,
                            &weights_projection_desc.data, &bias_desc.data,
                            &dst_layer_desc.data, &dst_iter_desc.data,
                            &dst_iter_c_desc.data, dnnl::convert_to_c(flags)),
                    "could not create a descriptor for an LSTM forward "
                    "propagation primitive");
        }

        /// Constructs a descriptor for an LSTM (with or without peephole)
        /// forward propagation primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p src_iter_c_desc,
        /// - @p weights_peephole_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc together with @p dst_iter_c_desc.
        ///
        /// This would then indicate that the LSTM forward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors can be initialized with an
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param src_iter_c_desc Memory descriptor for the input recurrent
        ///     cell state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param weights_peephole_desc Memory descriptor for the weights
        ///     applied to the cell states (according to the Peephole LSTM
        ///     formula).
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param dst_iter_c_desc Memory descriptor for the output recurrent
        ///     cell state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &src_iter_c_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &weights_peephole_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &dst_iter_c_desc,
                rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                    dnnl_lstm_forward_desc_init_v2(&data,
                            dnnl::convert_to_c(aprop_kind),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &src_iter_c_desc.data,
                            &weights_layer_desc.data, &weights_iter_desc.data,
                            &weights_peephole_desc.data, &bias_desc.data,
                            &dst_layer_desc.data, &dst_iter_desc.data,
                            &dst_iter_c_desc.data, dnnl::convert_to_c(flags)),
                    "could not create a descriptor for an LSTM forward "
                    "propagation primitive");
        }

        /// Constructs a descriptor for an LSTM forward propagation primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p src_iter_c_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc together with @p dst_iter_c_desc.
        ///
        /// This would then indicate that the LSTM forward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors can be initialized with an
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param src_iter_c_desc Memory descriptor for the input recurrent
        ///     cell state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param dst_iter_c_desc Memory descriptor for the output recurrent
        ///     cell state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &src_iter_c_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &dst_iter_c_desc,
                rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                    dnnl_lstm_forward_desc_init(&data,
                            dnnl::convert_to_c(aprop_kind),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &src_iter_c_desc.data,
                            &weights_layer_desc.data, &weights_iter_desc.data,
                            &bias_desc.data, &dst_layer_desc.data,
                            &dst_iter_desc.data, &dst_iter_c_desc.data,
                            dnnl::convert_to_c(flags)),
                    "could not create a descriptor for an LSTM forward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for an LSTM forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LSTM forward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for an LSTM forward propagation primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                bool allow_empty = false)
            : rnn_primitive_desc_base(
                    &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an LSTM forward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for an LSTM forward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine, bool allow_empty = false)
            : rnn_primitive_desc_base(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an LSTM forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for an LSTM forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference,
                    dnnl::algorithm::vanilla_lstm) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_c_desc() const {
            return rnn_base::src_iter_c_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_peephole_desc()const
        memory::desc weights_peephole_desc() const {
            return rnn_base::weights_peephole_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_projection_desc()const
        memory::desc weights_projection_desc() const {
            return rnn_base::weights_projection_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc dst_iter_c_desc() const {
            return rnn_base::dst_iter_c_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    lstm_forward() = default;

    /// Constructs an LSTM forward propagation primitive.
    /// @param pd Primitive descriptor for an LSTM forward propagation
    ///     primitive.
    lstm_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// LSTM backward propagation primitive.
struct lstm_backward : public primitive {
    /// Descriptor for an LSTM backward propagation primitive.
    struct desc {
        dnnl_rnn_desc_t data;

        /// Constructs an LSTM (with or without peephole and with or without
        /// projection) descriptor for backward propagation using @p prop_kind,
        /// @p direction, and memory descriptors.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p src_iter_c_desc,
        ///   @p diff_src_iter_desc, and @p diff_src_iter_c_desc,
        /// - @p weights_peephole_desc together with
        ///   @p diff_weights_peephole_desc
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p dst_iter_c_desc,
        ///   @p diff_dst_iter_desc, and @p diff_dst_iter_c_desc.
        ///
        /// This would then indicate that the LSTM backward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// The @p weights_projection_desc together with @p
        /// diff_weights_projection_desc may point to a zero memory descriptor.
        /// This would then indicate that the LSTM doesn't have recurrent
        /// projection layer.
        ///
        /// @note
        ///     All memory descriptors can be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Must be
        ///     #dnnl::prop_kind::backward.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param src_iter_c_desc Memory descriptor for the input recurrent
        ///     cell state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param weights_peephole_desc Memory descriptor for the weights
        ///     applied to the cell states (according to the Peephole LSTM
        ///     formula).
        /// @param weights_projection_desc Memory descriptor for the weights
        ///     applied to the hidden states to get the recurrent projection
        ///     (according to the Projection LSTM formula).
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param dst_iter_c_desc Memory descriptor for the output recurrent
        ///     cell state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_src_iter_c_desc Memory descriptor for the diff of
        ///     input recurrent cell state vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_weights_peephole_desc Memory descriptor for the diff of
        ///     weights applied to the cell states (according to the Peephole
        ///     LSTM formula).
        /// @param diff_weights_projection_desc Memory descriptor for the diff
        ///     of weights applied to the hidden states to get the recurrent
        ///     projection (according to the Projection LSTM formula).
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param diff_dst_iter_c_desc Memory descriptor for the diff of
        ///     output recurrent cell state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &src_iter_c_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &weights_peephole_desc,
                const memory::desc &weights_projection_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &dst_iter_c_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_src_iter_c_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_weights_peephole_desc,
                const memory::desc &diff_weights_projection_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc,
                const memory::desc &diff_dst_iter_c_desc,
                rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                    dnnl_lstm_backward_desc_init_v3(&data,
                            dnnl::convert_to_c(aprop_kind),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &src_iter_c_desc.data,
                            &weights_layer_desc.data, &weights_iter_desc.data,
                            &weights_peephole_desc.data,
                            &weights_projection_desc.data, &bias_desc.data,
                            &dst_layer_desc.data, &dst_iter_desc.data,
                            &dst_iter_c_desc.data, &diff_src_layer_desc.data,
                            &diff_src_iter_desc.data,
                            &diff_src_iter_c_desc.data,
                            &diff_weights_layer_desc.data,
                            &diff_weights_iter_desc.data,
                            &diff_weights_peephole_desc.data,
                            &diff_weights_projection_desc.data,
                            &diff_bias_desc.data, &diff_dst_layer_desc.data,
                            &diff_dst_iter_desc.data,
                            &diff_dst_iter_c_desc.data,
                            dnnl::convert_to_c(flags)),
                    "could not create a descriptor for an LSTM backward "
                    "propagation primitive");
        }

        /// Constructs an LSTM (with or without peephole) descriptor for
        /// backward propagation using @p prop_kind, @p direction, and memory
        /// descriptors.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p src_iter_c_desc,
        ///   @p diff_src_iter_desc, and @p diff_src_iter_c_desc,
        /// - @p weights_peephole_desc together with
        ///   @p diff_weights_peephole_desc
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p dst_iter_c_desc,
        ///   @p diff_dst_iter_desc, and @p diff_dst_iter_c_desc.
        ///
        /// This would then indicate that the LSTM backward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors may be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Must be
        ///     #dnnl::prop_kind::backward.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param src_iter_c_desc Memory descriptor for the input recurrent
        ///     cell state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param weights_peephole_desc Memory descriptor for the weights
        ///     applied to the cell states (according to the Peephole LSTM
        ///     formula).
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param dst_iter_c_desc Memory descriptor for the output recurrent
        ///     cell state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_src_iter_c_desc Memory descriptor for the diff of
        ///     input recurrent cell state vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_weights_peephole_desc Memory descriptor for the diff of
        ///     weights applied to the cell states (according to the Peephole
        ///     LSTM formula).
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param diff_dst_iter_c_desc Memory descriptor for the diff of
        ///     output recurrent cell state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &src_iter_c_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &weights_peephole_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &dst_iter_c_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_src_iter_c_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_weights_peephole_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc,
                const memory::desc &diff_dst_iter_c_desc,
                rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                    dnnl_lstm_backward_desc_init_v2(&data,
                            dnnl::convert_to_c(aprop_kind),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &src_iter_c_desc.data,
                            &weights_layer_desc.data, &weights_iter_desc.data,
                            &weights_peephole_desc.data, &bias_desc.data,
                            &dst_layer_desc.data, &dst_iter_desc.data,
                            &dst_iter_c_desc.data, &diff_src_layer_desc.data,
                            &diff_src_iter_desc.data,
                            &diff_src_iter_c_desc.data,
                            &diff_weights_layer_desc.data,
                            &diff_weights_iter_desc.data,
                            &diff_weights_peephole_desc.data,
                            &diff_bias_desc.data, &diff_dst_layer_desc.data,
                            &diff_dst_iter_desc.data,
                            &diff_dst_iter_c_desc.data,
                            dnnl::convert_to_c(flags)),
                    "could not create a descriptor for an LSTM backward "
                    "propagation primitive");
        }

        /// Constructs an LSTM descriptor for backward propagation using @p
        /// prop_kind, @p direction, and memory descriptors.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p src_iter_c_desc,
        ///   @p diff_src_iter_desc, and @p diff_src_iter_c_desc,
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p dst_iter_c_desc,
        ///   @p diff_dst_iter_desc, and @p diff_dst_iter_c_desc.
        ///
        /// This would then indicate that the LSTM backward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors may be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Must be
        ///     #dnnl::prop_kind::backward.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param src_iter_c_desc Memory descriptor for the input recurrent
        ///     cell state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param dst_iter_c_desc Memory descriptor for the output recurrent
        ///     cell state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_src_iter_c_desc Memory descriptor for the diff of
        ///     input recurrent cell state vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param diff_dst_iter_c_desc Memory descriptor for the diff of
        ///     output recurrent cell state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &src_iter_c_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &dst_iter_c_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_src_iter_c_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc,
                const memory::desc &diff_dst_iter_c_desc,
                rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                    dnnl_lstm_backward_desc_init(&data,
                            dnnl::convert_to_c(aprop_kind),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &src_iter_c_desc.data,
                            &weights_layer_desc.data, &weights_iter_desc.data,
                            &bias_desc.data, &dst_layer_desc.data,
                            &dst_iter_desc.data, &dst_iter_c_desc.data,
                            &diff_src_layer_desc.data, &diff_src_iter_desc.data,
                            &diff_src_iter_c_desc.data,
                            &diff_weights_layer_desc.data,
                            &diff_weights_iter_desc.data, &diff_bias_desc.data,
                            &diff_dst_layer_desc.data, &diff_dst_iter_desc.data,
                            &diff_dst_iter_c_desc.data,
                            dnnl::convert_to_c(flags)),
                    "could not create a descriptor for an LSTM backward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for an LSTM backward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LSTM backward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for LSTM backward propagation primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an LSTM
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const lstm_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, nullptr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an LSTM backward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for an LSTM backward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an LSTM
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine,
                const lstm_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an LSTM backward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for an LSTM backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::backward,
                    dnnl::algorithm::vanilla_lstm) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_c_desc() const {
            return rnn_base::src_iter_c_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_peephole_desc()const
        memory::desc weights_peephole_desc() const {
            return rnn_base::weights_peephole_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_projection_desc()const
        memory::desc weights_projection_desc() const {
            return rnn_base::weights_projection_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc dst_iter_c_desc() const {
            return rnn_base::dst_iter_c_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_layer_desc()const
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_iter_desc()const
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_iter_c_desc()const
        memory::desc diff_src_iter_c_desc() const {
            return rnn_base::diff_src_iter_c_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_layer_desc()const
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_iter_desc()const
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_peephole_desc()const
        memory::desc diff_weights_peephole_desc() const {
            return rnn_base::diff_weights_peephole_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_projection_desc()const
        memory::desc diff_weights_projection_desc() const {
            return rnn_base::diff_weights_projection_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_layer_desc()const
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_iter_desc()const
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_iter_c_desc()const
        memory::desc diff_dst_iter_c_desc() const {
            return rnn_base::diff_dst_iter_c_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    lstm_backward() = default;

    /// Constructs an LSTM backward propagation primitive.
    /// @param pd Primitive descriptor for an LSTM backward propagation
    ///     primitive.
    lstm_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// GRU forward propagation primitive.
struct gru_forward : public primitive {
    /// Descriptor for a GRU forward propagation primitive.
    struct desc {
        dnnl_rnn_desc_t data;

        /// Constructs a descriptor for a GRU forward propagation primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc.
        ///
        /// This would then indicate that the GRU forward propagation primitive
        /// should not use them and should default to zero values instead.
        ///
        /// @note
        ///     All memory descriptors except @p src_iter_desc may be
        ///     initialized with an #dnnl::memory::format_tag::any value of @p
        ///     format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                    dnnl_gru_forward_desc_init(&data,
                            dnnl::convert_to_c(aprop_kind),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &weights_layer_desc.data,
                            &weights_iter_desc.data, &bias_desc.data,
                            &dst_layer_desc.data, &dst_iter_desc.data,
                            dnnl::convert_to_c(flags)),
                    "could not create a descriptor for a GRU forward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for a GRU forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a GRU forward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for a GRU forward propagation primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                bool allow_empty = false)
            : rnn_primitive_desc_base(
                    &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a GRU forward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for a GRU forward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine, bool allow_empty = false)
            : rnn_primitive_desc_base(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a GRU forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a GRU forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference,
                    dnnl::algorithm::vanilla_gru) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    gru_forward() = default;

    /// Constructs a GRU forward propagation primitive.
    /// @param pd Primitive descriptor for a GRU forward propagation
    ///     primitive.
    gru_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// GRU backward propagation primitive.
struct gru_backward : public primitive {
    /// Descriptor for a GRU backward propagation primitive.
    struct desc {
        dnnl_rnn_desc_t data;

        /// Constructs a descriptor for a GRU backward propagation primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p diff_src_iter_desc,
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p diff_dst_iter_desc.
        ///
        /// This would then indicate that the GRU backward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors may be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Must be
        ///     #dnnl::prop_kind::backward.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc,
                rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                    dnnl_gru_backward_desc_init(&data,
                            dnnl::convert_to_c(aprop_kind),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &weights_layer_desc.data,
                            &weights_iter_desc.data, &bias_desc.data,
                            &dst_layer_desc.data, &dst_iter_desc.data,
                            &diff_src_layer_desc.data, &diff_src_iter_desc.data,
                            &diff_weights_layer_desc.data,
                            &diff_weights_iter_desc.data, &diff_bias_desc.data,
                            &diff_dst_layer_desc.data, &diff_dst_iter_desc.data,
                            dnnl::convert_to_c(flags)),
                    "could not create a descriptor for a GRU backward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for a GRU backward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a GRU backward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for a GRU backward propagation primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a GRU
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const gru_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, nullptr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a GRU backward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for a GRU backward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a GRU
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine,
                const gru_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a GRU backward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a GRU backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::backward,
                    dnnl::algorithm::vanilla_gru) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_layer_desc()const
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_iter_desc()const
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_layer_desc()const
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_iter_desc()const
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_layer_desc()const
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_iter_desc()const
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    gru_backward() = default;

    /// Constructs a GRU backward propagation primitive.
    /// @param pd Primitive descriptor for a GRU backward propagation
    ///     primitive.
    gru_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// LBR GRU forward propagation primitive.
struct lbr_gru_forward : public primitive {
    /// Descriptor for an LBR GRU forward propagation primitive.
    struct desc {
        dnnl_rnn_desc_t data;

        /// Constructs a descriptor for LBR GRU forward propagation primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc.
        ///
        /// This would then indicate that the LBR GRU forward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors except @p src_iter_desc may be
        ///     initialized with an #dnnl::memory::format_tag::any value of @p
        ///     format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                    dnnl_lbr_gru_forward_desc_init(&data,
                            dnnl::convert_to_c(aprop_kind),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &weights_layer_desc.data,
                            &weights_iter_desc.data, &bias_desc.data,
                            &dst_layer_desc.data, &dst_iter_desc.data,
                            dnnl::convert_to_c(flags)),
                    "could not create a descriptor for an LBR GRU forward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for an LBR GRU forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a LBR GRU forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a LBR GRU forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                bool allow_empty = false)
            : rnn_primitive_desc_base(
                    &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a LBR GRU forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a LBR GRU forward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine, bool allow_empty = false)
            : rnn_primitive_desc_base(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a LBR GRU forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a LBR GRU forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference,
                    dnnl::algorithm::lbr_gru) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    lbr_gru_forward() = default;

    /// Constructs an LBR GRU forward propagation primitive.
    /// @param pd Primitive descriptor for an LBR GRU forward propagation
    ///     primitive.
    lbr_gru_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// LBR GRU backward propagation primitive.
struct lbr_gru_backward : public primitive {
    /// Descriptor for a LBR GRU backward propagation primitive.
    struct desc {
        dnnl_rnn_desc_t data;

        /// Constructs a descriptor for LBR GRU backward propagation
        /// primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p diff_src_iter_desc,
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p diff_dst_iter_desc.
        ///
        /// This would then indicate that the LBR GRU backward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors may be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Must be
        ///     #dnnl::prop_kind::backward.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc,
                rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                    dnnl_lbr_gru_backward_desc_init(&data,
                            dnnl::convert_to_c(aprop_kind),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &weights_layer_desc.data,
                            &weights_iter_desc.data, &bias_desc.data,
                            &dst_layer_desc.data, &dst_iter_desc.data,
                            &diff_src_layer_desc.data, &diff_src_iter_desc.data,
                            &diff_weights_layer_desc.data,
                            &diff_weights_iter_desc.data, &diff_bias_desc.data,
                            &diff_dst_layer_desc.data, &diff_dst_iter_desc.data,
                            dnnl::convert_to_c(flags)),
                    "could not create a descriptor for an LBR GRU backward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for an LBR GRU backward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LBR GRU backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an LBR GRU backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an LBR GRU
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const lbr_gru_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, nullptr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an LBR GRU backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an LBR GRU backward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an LBR GRU
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine,
                const lbr_gru_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a LBR GRU backward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a LBR GRU backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(
                    pd, dnnl::prop_kind::backward, dnnl::algorithm::lbr_gru) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_layer_desc()const
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_iter_desc()const
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_layer_desc()const
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_iter_desc()const
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_layer_desc()const
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_iter_desc()const
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    lbr_gru_backward() = default;

    /// Constructs an LBR GRU backward propagation primitive.
    /// @param pd Primitive descriptor for an LBR GRU backward propagation
    ///     primitive.
    lbr_gru_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @} dnnl_api_rnn

/// @addtogroup dnnl_api_shuffle Shuffle
///
/// A primitive to shuffle tensor data along an axis.
///
/// @sa @ref dev_guide_shuffle in developer guide
///
/// @{

/// Shuffle forward propagation primitive.
struct shuffle_forward : public primitive {
    /// Descriptor for a shuffle forward propagation primitive.
    struct desc {
        dnnl_shuffle_desc_t data;

        /// Constructs a descriptor for a shuffle forward propagation
        /// primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param data_desc Source and destination memory descriptor.
        /// @param axis The axis along which the data is shuffled.
        /// @param group_size Shuffle group size.
        desc(prop_kind aprop_kind, const memory::desc &data_desc, int axis,
                int group_size) {
            error::wrap_c_api(dnnl_shuffle_forward_desc_init(&data,
                                      dnnl::convert_to_c(aprop_kind),
                                      &data_desc.data, axis, group_size),
                    "could not create a descriptor for a shuffle forward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for a shuffle forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a shuffle forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a shuffle forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const primitive_attr &attr = primitive_attr(),
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a shuffle forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a shuffle forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::shuffle,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    shuffle_forward() = default;

    /// Constructs a shuffle forward propagation primitive.
    /// @param pd Primitive descriptor for a shuffle forward propagation
    ///     primitive.
    shuffle_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Shuffle backward propagation primitive.
struct shuffle_backward : public primitive {
    /// Descriptor for a shuffle primitive backward propagation
    /// primitive.
    struct desc {
        dnnl_shuffle_desc_t data;

        /// Constructs a descriptor for a shuffle backward propagation
        /// primitive.
        ///
        /// @param diff_data_desc Diff source and diff destination memory
        ///     descriptor.
        /// @param axis The axis along which the data is shuffled.
        /// @param group_size Shuffle group size.
        desc(const memory::desc &diff_data_desc, int axis, int group_size) {
            error::wrap_c_api(dnnl_shuffle_backward_desc_init(&data,
                                      &diff_data_desc.data, axis, group_size),
                    "could not create a descriptor for a shuffle backward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for a shuffle backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a shuffle backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a shuffle backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param hint_fwd_pd Primitive descriptor for a shuffle
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const shuffle_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = primitive_attr(),
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a shuffle backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a shuffle backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::shuffle,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    shuffle_backward() = default;

    /// Constructs a shuffle backward propagation primitive.
    /// @param pd Primitive descriptor for a shuffle backward propagation
    ///     primitive.
    shuffle_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @} dnnl_api_shuffle

/// @addtogroup dnnl_api_binary Binary
///
/// A primitive to perform tensor operations over two tensors.
///
/// @sa @ref dev_guide_binary in developer guide
///
/// @{

/// Elementwise binary operator primitive.
struct binary : public primitive {
    /// Descriptor for an elementwise binary operator primitive.
    struct desc {
        /// Underlying C operation descriptor.
        dnnl_binary_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Constructs a descriptor for an elementwise binary operator
        /// primitive.
        ///
        /// @param aalgorithm Elementwise binary algorithm.
        /// @param src0 Memory descriptor for source tensor #0.
        /// @param src1 Memory descriptor for source tensor #1.
        /// @param dst Memory descriptor for destination tensor.
        desc(algorithm aalgorithm, const memory::desc &src0,
                const memory::desc &src1, const memory::desc &dst) {
            error::wrap_c_api(
                    dnnl_binary_desc_init(&data, dnnl::convert_to_c(aalgorithm),
                            &src0.data, &src1.data, &dst.data),
                    "could not create a descriptor for a binary operation "
                    "primitive");
        }
    };

    /// Primitive descriptor for an elementwise binary operator primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an elementwise binary operator
        /// primitive.
        ///
        /// @param adesc Descriptor for an elementwise binary operator primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an elementwise binary operator
        /// primitive.
        ///
        /// @param adesc Descriptor for an elementwise binary operator primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a binary primitive from a C
        /// API primitive descriptor that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a binary primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::binary) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc(int)const
        memory::desc src_desc(int idx = 0) const { return base::src_desc(idx); }

        /// Returns the memory descriptor for source #0.
        memory::desc src0_desc() const { return base::src_desc(0); }

        /// Returns the memory descriptor for source #1.
        memory::desc src1_desc() const { return base::src_desc(1); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    binary() = default;

    /// Constructs an elementwise binary operation primitive.
    /// @param pd Primitive descriptor for an elementwise binary operation
    ///     primitive.
    binary(const primitive_desc &pd) : primitive(pd) {}
};

/// @} dnnl_api_binary

/// @addtogroup dnnl_api_matmul Matrix Multiplication
///
/// A primitive to perform matrix-matrix multiplication. The batched mode
/// is supported with 3D tensors.
///
/// @sa @ref dev_guide_matmul in developer guide
///
///
/// @{

/// Matrix multiplication (matmul) primitive.
struct matmul : public primitive {
    /// Descriptor for a matmul primitive.
    struct desc {
        dnnl_matmul_desc_t data;

        /// Constructs a descriptor for a matmul primitive.
        ///
        /// @param src_desc Memory descriptor for source (matrix A).
        /// @param weights_desc Memory descriptor for weights (matrix B).
        /// @param dst_desc Memory descriptor for destination (matrix C).
        desc(const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &dst_desc) {
            error::wrap_c_api(
                    dnnl_matmul_desc_init(&data, &src_desc.data,
                            &weights_desc.data, nullptr, &dst_desc.data),
                    "could not create a descriptor for a matmul primitive");
        }

        /// Constructs a descriptor for a matmul primitive.
        ///
        /// @param src_desc Memory descriptor for source (matrix A).
        /// @param weights_desc Memory descriptor for weights (matrix B).
        /// @param dst_desc Memory descriptor for destination (matrix C).
        /// @param bias_desc Memory descriptor for bias.
        desc(const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &bias_desc, const memory::desc &dst_desc) {
            error::wrap_c_api(dnnl_matmul_desc_init(&data, &src_desc.data,
                                      &weights_desc.data, &bias_desc.data,
                                      &dst_desc.data),
                    "could not create a descriptor for a matmul primitive");
        }
    };

    /// Primitive descriptor for a matmul primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a matmul primitive.
        ///
        /// @param adesc Descriptor for a matmul primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a matmul primitive.
        ///
        /// @param adesc Descriptor for a matmul primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a matmul primitive from a C
        /// API primitive descriptor that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a matmul primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::matmul) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return query_md(query::src_md, 0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const {
            return query_md(query::weights_md, 0);
        }

        /// @copydoc dnnl::convolution_forward::primitive_desc::bias_desc()const
        memory::desc bias_desc() const {
            return query_md(query::weights_md, 1);
        }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return query_md(query::dst_md, 0); }
    };

    /// Default constructor. Produces an empty object.
    matmul() = default;

    /// Constructs a matmul primitive.
    /// @param pd Primitive descriptor for a matmul primitive.
    matmul(const primitive_desc &pd) : primitive(pd) {}
};

/// @} dnnl_api_matmul

/// @addtogroup dnnl_api_resampling Resampling
///
/// A primitive to compute resampling operation on 1D, 2D or 3D data tensor
/// using Nearest Neighbor, or Linear (Bilinear, Trilinear) interpolation
/// method.
///
/// @sa @ref dev_guide_resampling in developer guide
///
/// @{

/// Resampling forward propagation.
struct resampling_forward : public primitive {
    /// Descriptor for resampling forward propagation.
    struct desc {
        dnnl_resampling_desc_t data;

        /// Constructs a descriptor for a resampling forward propagation
        /// primitive using source and destination memory descriptors.
        ///
        /// @note
        ///     Destination memory descriptor may be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm resampling algorithm kind: either
        ///     #dnnl::algorithm::resampling_nearest, or
        ///     #dnnl::algorithm::resampling_linear
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc, const memory::desc &dst_desc) {
            error::wrap_c_api(dnnl_resampling_forward_desc_init(&data,
                                      dnnl::convert_to_c(aprop_kind),
                                      convert_to_c(aalgorithm), nullptr,
                                      &src_desc.data, &dst_desc.data),
                    "could not create a resampling forward descriptor");
        }

        /// Constructs a descriptor for a resampling forward propagation
        /// primitive using source memory descriptor and factors.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm resampling algorithm kind: either
        ///     #dnnl::algorithm::resampling_nearest, or
        ///     #dnnl::algorithm::resampling_linear
        /// @param factors Vector of scaling factors for spatial dimension.
        /// @param src_desc Source memory descriptor.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const std::vector<float> &factors,
                const memory::desc &src_desc) {
            memory::validate_dims(factors, src_desc.data.ndims - 2);
            error::wrap_c_api(dnnl_resampling_forward_desc_init(&data,
                                      dnnl::convert_to_c(aprop_kind),
                                      convert_to_c(aalgorithm), &factors[0],
                                      &src_desc.data, nullptr),
                    "could not create a resampling forward descriptor");
        }

        /// Constructs a descriptor for a resampling forward propagation
        /// primitive.
        ///
        /// @note
        ///     The destination memory descriptor may be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm resampling algorithm kind: either
        ///     #dnnl::algorithm::resampling_nearest, or
        ///     #dnnl::algorithm::resampling_linear
        /// @param factors Vector of scaling factors for spatial dimension.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const std::vector<float> &factors, const memory::desc &src_desc,
                const memory::desc &dst_desc) {
            if (!factors.empty())
                memory::validate_dims(factors, src_desc.data.ndims - 2);
            error::wrap_c_api(dnnl_resampling_forward_desc_init(&data,
                                      dnnl::convert_to_c(aprop_kind),
                                      convert_to_c(aalgorithm), factors.data(),
                                      &src_desc.data, &dst_desc.data),
                    "could not create a resampling forward descriptor");
        }
    };

    /// Primitive descriptor for a resampling forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a resampling forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a resampling forward propagation
        /// primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a resampling forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a resampling forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a resampling forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a resampling forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::resampling,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    resampling_forward() = default;

    /// Constructs a resampling forward propagation primitive.
    /// @param pd Primitive descriptor for a resampling forward propagation
    ///     primitive.
    resampling_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Resampling backward propagation primitive.
struct resampling_backward : public primitive {
    /// Descriptor for a resampling backward propagation primitive.
    struct desc {
        dnnl_resampling_desc_t data;

        /// Constructs a descriptor for a resampling backward propagation
        /// primitive using source and destination memory descriptors.
        ///
        /// @param aalgorithm resampling algorithm kind: either
        ///     #dnnl::algorithm::resampling_nearest, or
        ///     #dnnl::algorithm::resampling_linear
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        desc(algorithm aalgorithm, const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc) {
            error::wrap_c_api(dnnl_resampling_backward_desc_init(&data,
                                      convert_to_c(aalgorithm), nullptr,
                                      &diff_src_desc.data, &diff_dst_desc.data),
                    "could not create a resampling backward data descriptor");
        }

        /// Constructs a descriptor for resampling backward propagation
        /// primitive.
        ///
        /// @param aalgorithm resampling algorithm kind: either
        ///     #dnnl::algorithm::resampling_nearest, or
        ///     #dnnl::algorithm::resampling_linear
        /// @param factors Vector of scaling factors for spatial dimension.
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        desc(algorithm aalgorithm, const std::vector<float> &factors,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc) {
            if (!factors.empty())
                memory::validate_dims(factors, diff_src_desc.data.ndims - 2);
            error::wrap_c_api(dnnl_resampling_backward_desc_init(&data,
                                      convert_to_c(aalgorithm), factors.data(),
                                      &diff_src_desc.data, &diff_dst_desc.data),
                    "could not create a resampling backward data descriptor");
        }
    };

    /// Primitive descriptor for resampling backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a resampling backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a resampling backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a resampling forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const resampling_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, nullptr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a resampling backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a resampling backward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a resampling forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine,
                const resampling_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a resampling backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a resampling backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::resampling,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    resampling_backward() = default;

    /// Constructs a resampling backward propagation primitive.
    /// @param pd Primitive descriptor for a resampling backward propagation
    ///     primitive.
    resampling_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @} dnnl_api_resampling

/// @addtogroup dnnl_api_pooling
/// @{

/// Pooling v2 (dilated pooling) forward propagation primitive.
struct pooling_v2_forward : public primitive {
    /// Descriptor for a pooling forward propagation primitive.
    struct desc {
        dnnl_pooling_v2_desc_t data;

        /// Constructs a descriptor for pooling v2
        /// (dilated pooling) forward propagation primitive.
        ///
        /// Arrays @p strides, @p kernel, @p dilation, @p padding_l
        /// and @p padding_r contain values for spatial dimensions only and
        /// hence must have the same number of elements as there are spatial
        /// dimensions. The order of values is the same as in the tensor:
        /// depth (for 3D tensors), height (for 3D and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Pooling algorithm kind: either
        ///     #dnnl::algorithm::pooling_max,
        ///     #dnnl::algorithm::pooling_avg_include_padding,
        ///     or #dnnl::algorithm::pooling_avg (same as
        ///     #dnnl::algorithm::pooling_avg_exclude_padding).
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param kernel Vector of kernel spatial dimensions.
        /// @param dilation Array of dilations for spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc, const memory::desc &dst_desc,
                const memory::dims &strides, const memory::dims &kernel,
                const memory::dims &dilation, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(kernel, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            memory::validate_dims(dilation, src_desc.data.ndims - 2);
            error::wrap_c_api(
                    dnnl_pooling_v2_forward_desc_init(&data,
                            dnnl::convert_to_c(aprop_kind),
                            convert_to_c(aalgorithm), &src_desc.data,
                            &dst_desc.data, &strides[0], &kernel[0],
                            &dilation[0], &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a pooling forward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for a pooling forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a pooling v2
        /// (dilated pooling) forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a pooling forward propagation primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a pooling v2
        /// (dilated pooling) forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a pooling forward propagation primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a pooling v2
        /// (dilated pooling) forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a pooling forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::pooling_v2,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }
    };

    /// Default constructor. Produces an empty object.
    pooling_v2_forward() = default;

    /// Constructs a pooling v2 (dilated pooling) forward
    /// propagation primitive.
    /// @param pd Primitive descriptor for a pooling v2
    /// (dilated pooling) forward propagation primitive.
    pooling_v2_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Pooling v2 (dilated pooling) backward propagation primitive.
struct pooling_v2_backward : public primitive {
    /// Descriptor for a pooling backward propagation primitive.
    struct desc {
        dnnl_pooling_v2_desc_t data;

        /// Constructs a descriptor for pooling v2 (dilated pooling) backward
        /// propagation primitive.
        ///
        /// Arrays @p strides, @p kernel, @p dilation, @p padding_l
        /// and @p padding_r contain values for spatial dimensions only and
        /// hence must have the same number of elements as there are spatial
        /// dimensions. The order of values is the same as in the tensor:
        /// depth (for 3D tensors), height (for 3D and 2D tensors), and width.
        ///
        /// @param aalgorithm Pooling algorithm kind: either
        ///     #dnnl::algorithm::pooling_max,
        ///     #dnnl::algorithm::pooling_avg_include_padding,
        ///     or #dnnl::algorithm::pooling_avg (same as
        ///     #dnnl::algorithm::pooling_avg_exclude_padding).
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param kernel Vector of kernel spatial dimensions.
        /// @param dilation Array of dilations for spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &kernel, const memory::dims &dilation,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, diff_src_desc.data.ndims - 2);
            memory::validate_dims(kernel, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, diff_src_desc.data.ndims - 2);
            memory::validate_dims(dilation, diff_src_desc.data.ndims - 2);
            error::wrap_c_api(
                    dnnl_pooling_v2_backward_desc_init(&data,
                            convert_to_c(aalgorithm), &diff_src_desc.data,
                            &diff_dst_desc.data, &strides[0], &kernel[0],
                            &dilation[0], &padding_l[0], &padding_r[0]),
                    "could not create a descriptor for a pooling backward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for a pooling v2 (dilated pooling) backward
    /// propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a pooling v2
        /// (dilated pooling) backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a pooling backward propagation primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a pooling forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                const pooling_v2_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, nullptr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a pooling v2
        /// (dilated pooling) backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a pooling backward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a pooling forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine,
                const pooling_v2_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&adesc.data, &attr, aengine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a pooling v2
        /// (dilated pooling) backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a pooling backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::pooling_v2,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }
    };

    /// Default constructor. Produces an empty object.
    pooling_v2_backward() = default;

    /// Constructs a pooling v2 (dilated pooling) backward
    /// propagation primitive.
    /// @param pd Primitive descriptor for a pooling backward propagation
    ///     primitive.
    pooling_v2_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @} dnnl_api_pooling

/// @addtogroup dnnl_api_reduction Reduction
///
/// A primitive to compute reduction operation on data tensor
/// using min, max, mul, sum, mean and norm_lp operations.
///
/// @sa @ref dev_guide_reduction in developer guide
///
/// @{

/// Reduction.
struct reduction : public primitive {
    /// Descriptor for reduction.
    struct desc {
        dnnl_reduction_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Constructs a descriptor for a reduction primitive using algorithm
        /// specific parameters, source and destination memory descriptors.
        ///
        /// @note
        ///     Destination memory descriptor may be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aalgorithm reduction algorithm kind. Possible values:
        ///     #dnnl_reduction_max, #dnnl_reduction_min, #dnnl_reduction_sum,
        ///     #dnnl_reduction_mul, #dnnl_reduction_mean,
        ///     #dnnl_reduction_norm_lp_max, #dnnl_reduction_norm_lp_sum,
        ///     #dnnl_reduction_norm_lp_power_p_max,
        ///     #dnnl_reduction_norm_lp_power_p_sum.
        /// @param p algorithm specific parameter.
        /// @param eps algorithm specific parameter.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        desc(algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &dst_desc, float p, float eps) {
            error::wrap_c_api(
                    dnnl_reduction_desc_init(&data, convert_to_c(aalgorithm),
                            &src_desc.data, &dst_desc.data, p, eps),
                    "could not create a reduction descriptor");
        }
    };

    /// Primitive descriptor for a reduction primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a reduction primitive.
        ///
        /// @param adesc Descriptor for a reduction primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a reduction primitive.
        ///
        /// @param adesc Descriptor for a reduction primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                const engine &aengine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a reduction primitive from a C
        /// API primitive descriptor that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a reduction primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::reduction) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    reduction() = default;

    /// Constructs a reduction primitive.
    /// @param pd Primitive descriptor for a reduction primitive.
    reduction(const primitive_desc &pd) : primitive(pd) {}
};

/// @} dnnl_api_reduction

/// @} dnnl_api_primitives

/// @addtogroup dnnl_api_service Service
///
/// A set of functions that aid in oneDNN debugging and profiling.
///
/// @{

/// @copydoc dnnl_version_t
using version_t = dnnl_version_t;

/// Status values returned by the library functions.
enum class status {
    /// @copydoc dnnl_success
    success = dnnl_success,
    /// @copydoc dnnl_out_of_memory
    out_of_memory = dnnl_out_of_memory,
    /// @copydoc dnnl_invalid_arguments
    invalid_arguments = dnnl_invalid_arguments,
    /// @copydoc dnnl_unimplemented
    unimplemented = dnnl_unimplemented,
    /// @copydoc dnnl_iterator_ends
    iterator_ends = dnnl_iterator_ends,
    /// @copydoc dnnl_runtime_error
    runtime_error = dnnl_runtime_error,
    /// @copydoc dnnl_not_required
    not_required = dnnl_not_required,
};

/// @copydoc dnnl_set_verbose()
inline status set_verbose(int level) {
    return static_cast<status>(dnnl_set_verbose(level));
}

/// @copydoc dnnl_version()
inline const version_t *version() {
    return dnnl_version();
}

/// @copydoc dnnl_set_jit_dump()
inline status set_jit_dump(int enable) {
    return static_cast<status>(dnnl_set_jit_dump(enable));
}

/// @copydoc dnnl_set_jit_profiling_flags()
inline status set_jit_profiling_flags(unsigned flags) {
    return static_cast<status>(dnnl_set_jit_profiling_flags(flags));
}

/// @copydoc dnnl_set_jit_profiling_jitdumpdir()
inline status set_jit_profiling_jitdumpdir(const std::string &dir) {
    return static_cast<status>(dnnl_set_jit_profiling_jitdumpdir(dir.c_str()));
}

/// @copydoc dnnl_cpu_isa_t
enum class cpu_isa {
    /// @copydoc dnnl_cpu_isa_all
    all = dnnl_cpu_isa_all,
    /// @copydoc dnnl_cpu_isa_sse41
    sse41 = dnnl_cpu_isa_sse41,
    /// @copydoc dnnl_cpu_isa_avx
    avx = dnnl_cpu_isa_avx,
    /// @copydoc dnnl_cpu_isa_avx2
    avx2 = dnnl_cpu_isa_avx2,
    /// @copydoc dnnl_cpu_isa_avx512_mic
    avx512_mic = dnnl_cpu_isa_avx512_mic,
    /// @copydoc dnnl_cpu_isa_avx512_mic_4ops
    avx512_mic_4ops = dnnl_cpu_isa_avx512_mic_4ops,
    /// @copydoc dnnl_cpu_isa_avx512_core
    avx512_core = dnnl_cpu_isa_avx512_core,
    /// @copydoc dnnl_cpu_isa_avx512_core_vnni
    avx512_core_vnni = dnnl_cpu_isa_avx512_core_vnni,
    /// @copydoc dnnl_cpu_isa_avx512_core_bf16
    avx512_core_bf16 = dnnl_cpu_isa_avx512_core_bf16,
    /// @copydoc dnnl_cpu_isa_avx512_core_amx
    avx512_core_amx = dnnl_cpu_isa_avx512_core_amx,
};

/// @copydoc dnnl_set_max_cpu_isa()
inline status set_max_cpu_isa(cpu_isa isa) {
    return static_cast<status>(
            dnnl_set_max_cpu_isa(static_cast<dnnl_cpu_isa_t>(isa)));
}

/// @copydoc dnnl_get_effective_cpu_isa()
inline cpu_isa get_effective_cpu_isa() {
    return static_cast<cpu_isa>(dnnl_get_effective_cpu_isa());
}

/// @} dnnl_api_service

/// @addtogroup dnnl_api_primitive_cache Primitive Cache
///
/// A set of functions that provide primitive cache control.
///
/// @{

/// Returns the number of primitives that can be held in the primitive cache
/// at the same time.
inline int get_primitive_cache_capacity() {
    int result = 0;
    error::wrap_c_api(dnnl_get_primitive_cache_capacity(&result),
            "could not get primitive cache capacity");
    return result;
}

/// @copydoc dnnl_set_primitive_cache_capacity(int capacity)
inline void set_primitive_cache_capacity(int capacity) {
    error::wrap_c_api(dnnl_set_primitive_cache_capacity(capacity),
            "could not set primitive cache capacity");
}

/// @} dnnl_api_primitive_cache

/// @addtogroup dnnl_api_blas BLAS functions
///
/// A subset of Basic Linear Algebra (BLAS) functions that perform
/// matrix-matrix multiplication.
///
/// @{

/// @copydoc dnnl_sgemm()
inline status sgemm(char transa, char transb, dnnl_dim_t M, dnnl_dim_t N,
        dnnl_dim_t K, float alpha, const float *A, dnnl_dim_t lda,
        const float *B, dnnl_dim_t ldb, float beta, float *C, dnnl_dim_t ldc) {
    return static_cast<status>(dnnl_sgemm(
            transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc));
}

/// @copydoc dnnl_gemm_u8s8s32()
inline status gemm_u8s8s32(char transa, char transb, char offsetc, dnnl_dim_t M,
        dnnl_dim_t N, dnnl_dim_t K, float alpha, const uint8_t *A,
        dnnl_dim_t lda, uint8_t ao, const int8_t *B, dnnl_dim_t ldb, int8_t bo,
        float beta, int32_t *C, dnnl_dim_t ldc, const int32_t *co) {
    return static_cast<status>(dnnl_gemm_u8s8s32(transa, transb, offsetc, M, N,
            K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co));
}

/// @copydoc dnnl_gemm_s8s8s32()
inline status gemm_s8s8s32(char transa, char transb, char offsetc, dnnl_dim_t M,
        dnnl_dim_t N, dnnl_dim_t K, float alpha, const int8_t *A,
        dnnl_dim_t lda, int8_t ao, const int8_t *B, dnnl_dim_t ldb, int8_t bo,
        float beta, int32_t *C, dnnl_dim_t ldc, const int32_t *co) {
    return static_cast<status>(dnnl_gemm_s8s8s32(transa, transb, offsetc, M, N,
            K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co));
}

/// @} dnnl_api_blas

// implementation section

/// @cond DO_NOT_DOCUMENT_THIS
inline primitive::primitive(const_dnnl_primitive_desc_t c_pd) {
    dnnl_primitive_t result;
    error::wrap_c_api(dnnl_primitive_create(&result, c_pd),
            "could not create a primitive");
    reset(result);
}

inline primitive::primitive(const primitive_desc &pd) : primitive(pd.get()) {}

inline void primitive::execute(const stream &astream,
        const std::unordered_map<int, memory> &args) const {
    std::vector<dnnl_exec_arg_t> c_args;
    c_args.reserve(args.size());
    for (const auto &a : args)
        c_args.push_back({a.first, a.second.get(true)});

    error::wrap_c_api(dnnl_primitive_execute(get(), astream.get(),
                              (int)c_args.size(), c_args.data()),
            "could not execute a primitive");
}
/// @endcond

#undef DNNL_DEFINE_BITMASK_OPS

} // namespace dnnl

/// oneAPI namespace
namespace oneapi {
/// oneDNN alias namespace
namespace dnnl = ::dnnl;
} // namespace oneapi

/// @} dnnl_api

#endif
